import contextlib
from typing import Callable, Sequence

import torch

from detection_eval.steering_hooks import get_vllm_steering_hook
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


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    except Exception as e:
        print(f"Error adding hook: {e}")
        raise e
    finally:
        handle.remove()


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
            # Try the path from vllm_hook_demo.py
            inference_model = rollout.inference_engine.llm_engine.model_executor.driver_worker.model_runner.model
            dtype = torch.bfloat16

            # Debug model structure
            print(f"🔍 Inference model type: {type(inference_model)}")
            print(f"🔍 Model layers type: {type(inference_model.model.layers)}")
            print(f"🔍 Total layers: {len(inference_model.model.layers)}")
            print(f"🔍 Target layer type: {type(inference_model.model.layers[layer])}")

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
            batch_size = len(all_feature_vectors)
            vectors = []
            positions = []

            for batch_idx in range(batch_size):
                # Each feature vector becomes a single entry per batch
                feature_vec = torch.tensor(all_feature_vectors[batch_idx], dtype=dtype, device=device)
                vectors.append(feature_vec)
                positions.append(x_position)

            # input_ids
            input_ids = prompts.batch["input_ids"]
            prompt_lengths = [len(input_id) for input_id in input_ids]
            hook = get_vllm_steering_hook(
                vectors=vectors,
                positions=positions,
                prompt_lengths=prompt_lengths,
                steering_coefficient=steering_coefficient,
                device=device,
                dtype=dtype,
            )

            processed_prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            # NOTE: Need to enforce eager
            with simple_timer("generate_sequences", timing_generate):
                with add_hook(module_to_target, hook):
                    print("Hook registered, starting generation...")
                    output = rollout.generate_sequences(prompts=processed_prompts)
                    print("Generation completed with hook")

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
