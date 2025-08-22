from typing import Sequence

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
        device: torch.device = torch.device(get_device_id())
        prompts = prompts.to(get_device_id())

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
            assert "sae" in prompts.non_tensor_batch, f"sae not in prompts: {prompts.non_tensor_batch.keys()}"

            processed_prompts = self.rollout_sharding_manager.preprocess_data(prompts)

            # need to pass sae rollout.generate_sequences. then in vllm_rollout_spmd.py you have the true lengths.
            # Here verl for some reason pads all the input ids to max input length. But vllm removes it later.
            # NOTE: Need to enforce eager
            processed_prompts.non_tensor_batch["sae"] = prompts.non_tensor_batch["sae"]
            with simple_timer("generate_sequences", timing_generate):
                output = rollout.generate_sequences(prompts=processed_prompts)

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
