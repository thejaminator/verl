from typing import Callable, Sequence

import torch

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing
from verl.workers.fsdp_workers import ActorRolloutRefWorker, logger
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout


def get_feature_vector(prompts: DataProto) -> Sequence[Sequence[float]]:
    """
    Get the feature vector from the prompts.
    """
    output: list[Sequence[float]] = []
    for item in prompts.non_tensor_batch["sae"]:
        output.append(item["feature_vector"])  # type: ignore
    return output


def get_activation_steering_hook(
    vectors: list[list[torch.Tensor]],  # [B][K, d_model]  or [K, d_model] if B==1
    positions: list[list[int]],  # [B][K]
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    K = number of feature/steering vectors per batch item
    Returns a forward hook that *replaces* specified residual-stream activations
    during the initial prompt pass of `model.generate`.

    • vectors[b][k]  – feature vector to inject for batch b, slot k
    • positions[b][k]– token index (0-based, within prompt only)
    """

    # ---- pack Python lists → torch tensors once, outside the hook ----
    vec_BKD = torch.stack([torch.stack(v) for v in vectors])  # (B, K, d)
    pos_BK = torch.tensor(positions, dtype=torch.long)  # (B, K)

    B, K, d_model = vec_BKD.shape
    assert pos_BK.shape == (B, K)

    vec_BKD = vec_BKD.to(device, dtype)
    pos_BK = pos_BK.to(device)

    def hook_fn(module, _input, output):
        resid_BLD, *rest = output  # Gemma returns (resid, hidden_states, ...)
        L = resid_BLD.shape[1]

        # Only touch the *prompt* forward pass (sequence length > 1)
        if L <= 1:
            return (resid_BLD, *rest)

        print(
            f"Applying feature vector on module {type(module).__name__}. Sequence length: {L}, Batch size: {resid_BLD.shape[0]}"
        )

        # Safety: make sure every position is inside current sequence
        if (pos_BK >= L).any():
            bad = pos_BK[pos_BK >= L].min().item()
            raise IndexError(f"position {bad} is out of bounds for length {L}")

        # ---- compute norms of original activations at the target slots ----
        batch_idx_B1 = torch.arange(B, device=device).unsqueeze(1)  # (B, 1) → (B, K)
        orig_BKD = resid_BLD[batch_idx_B1, pos_BK]  # (B, K, d)
        norms_BK1 = orig_BKD.norm(dim=-1, keepdim=True)  # (B, K, 1)

        # ---- build steered vectors ----
        steered_BKD = torch.nn.functional.normalize(vec_BKD, dim=-1) * norms_BK1 * steering_coefficient  # (B, K, d)

        # ---- in-place replacement via advanced indexing ----
        resid_BLD[batch_idx_B1, pos_BK] = steered_BKD

        return (resid_BLD, *rest)

    return hook_fn


class FeatureVectorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        """Place where we do the hooking.
        DataProto is
        batch: TensorDict = None
        non_tensor_batch: dict = field(default_factory=dict)
        meta_info: dict = field(default_factory=dict)

        non_tensor_batch should  have "sae": {"feature_vector": feature_vector}
        Where feature_vector is a list of floats.

        This is the class we use to pass the sae information to the rollout worker.
        class SAE(BaseModel):
            sae_id: int
            feature_vector: Sequence[float]
            activations: SAEActivations
            # Sentences that do not activate for the given sae_id. But come from a similar SAE
            # Here the sae_id correspond to different similar SAEs.
            # The activations are the activations w.r.t this SAE. And should be low.
            hard_negatives: list[SAEActivations]
        """
        # print(
        #     f"FeatureVectorRolloutRefWorker: Calling generate_sequences. prompts non_tensor_batch: {prompts.non_tensor_batch}"
        # )
        # Support all hardwares
        device: int = get_device_id()
        prompts = prompts.to(device)

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)
            rollout: vLLMRollout = self.rollout  # type: ignore
            """Hook logic begin"""
            layer = 9  # todo: DataProto may define this
            inference_model = rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            dtype = inference_model.dtype
    
            # This should get Gemma2DecoderLayer
            module_to_target = inference_model.model.layers[layer]

            # DataProto should contain
            try:
                all_feature_vectors: Sequence[Sequence[float]] = get_feature_vector(prompts)
                # print(f"First feature vector: {all_feature_vectors[0]}")
                print(f"Got feature vector for {len(all_feature_vectors)} prompts")
            except Exception as e:
                print(f"Error getting feature vector: {e} in prompts: {prompts}")
                raise ValueError("Feature vector not found in prompts")

            # Found X token at position: 11/33
            # Hardcoded for "Can you explain to me what 'X' means? Format your final answer with <explanation>" in chat format.
            x_position: int = 11  # TODO: I forgot to add this to DataProto.
            steering_coefficient = 2.0  # TODO: Let DataProto define this.

            # Convert feature vectors to tensor format expected by the hook
            # [B][K, d_model] - assuming d_model matches the feature vector dimension
            batch_size = len(all_feature_vectors)
            vectors = []
            positions = []

            for batch_idx in range(batch_size):
                # Each feature vector becomes a single K=1 entry
                feature_vec = torch.tensor(all_feature_vectors[batch_idx], dtype=dtype, device=device)
                vectors.append([feature_vec])  # K=1, so wrap in list
                positions.append([x_position])  # K=1, so wrap in list

            hook = get_activation_steering_hook(vectors, positions, steering_coefficient, device, dtype)
            module_to_target.register_forward_hook(hook)

            """hook logic end"""

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                # TODO(james): Add feature vector steering here.
                output = rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output
