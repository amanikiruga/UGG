import torch
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, CPUOffload
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from algorithms.wan import WanTextToVideoRayDepthMoTConcat5B
from datasets.mixture import MixtureDataset
from datasets.dummy import DummyVideoDataset
from .exp_base import BaseLightningExperiment


class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        ugg=WanTextToVideoRayDepthMoTConcat5B,  # the release model (configurations/algorithm/ugg.yaml)
        ugg_toy=WanTextToVideoRayDepthMoTConcat5B,  # 2-block debug version
    )

    compatible_datasets = dict(
        mixture_parallax14=MixtureDataset,  # the release training mixture (14 subsets)
        dummy=DummyVideoDataset,            # random tensors, for smoke tests
    )

    def _build_strategy(self):
        from lightning.pytorch.strategies.fsdp import FSDPStrategy

        if self.cfg.strategy == "ddp":
            return super()._build_strategy()
        elif self.cfg.strategy == "fsdp":
            gpus_per_node = max(torch.cuda.device_count(), 1)
            # Configurable sharding: HYBRID_SHARD (default) all-gathers params every
            # layer (heavy comm at small batch); SHARD_GRAD_OP keeps params resident
            # (no per-layer all-gather, only grad reduce-scatter) -> much less exposed
            # comm for models that fit. Override with +experiment.fsdp_sharding_strategy=...
            sharding = getattr(self.cfg, "fsdp_sharding_strategy", "HYBRID_SHARD")
            # Process-group timeout (default NCCL watchdog is 10 min, which a slow first step with
            # compilation can exceed). Override via experiment.pg_timeout_min.
            from datetime import timedelta
            pg_timeout_min = getattr(self.cfg, "pg_timeout_min", 30)
            fsdp_kwargs = dict(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                ),
                auto_wrap_policy=ModuleWrapPolicy(self.algo.classes_to_shard()),
                sharding_strategy=sharding,
                timeout=timedelta(minutes=pg_timeout_min),
                # Lower peak memory (math-preserving): BACKWARD_POST + limit_all_gathers keeps ~1 unit's full
                # params resident instead of ~2. Override via +experiment.fsdp_backward_prefetch_post=false.
                backward_prefetch=(
                    BackwardPrefetch.BACKWARD_POST
                    if getattr(self.cfg, "fsdp_backward_prefetch_post", True)
                    else BackwardPrefetch.BACKWARD_PRE
                ),
                limit_all_gathers=True,
                # "full" rather than "sharded": sharded checkpoints omit the frozen VAE, and the loader fails
                # on the missing keys before `strict` applies. "full" needs host RAM for the gathered state.
                # Override: +experiment.fsdp_state_dict_type=...
                state_dict_type=getattr(self.cfg, "fsdp_state_dict_type", "full"),
            )
            # Optional CPU offload of sharded params, grads and the optimizer step: frees GPU memory at the cost
            # of speed (math-preserving). Enable with +experiment.fsdp_cpu_offload=true.
            if getattr(self.cfg, "fsdp_cpu_offload", False):
                fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
            # HYBRID variants need a 2D device mesh; FULL_SHARD/SHARD_GRAD_OP/NO_SHARD use 1D world.
            if "HYBRID" in sharding:
                if getattr(self.cfg, "fsdp_shard_within_node", False):
                    # Shard within each node, replicate across nodes (only a gradient all-reduce
                    # crosses nodes). mesh=(replicate=num_nodes, shard=gpus_per_node).
                    device_mesh = (self.cfg.num_nodes, gpus_per_node)
                elif self.cfg.num_nodes >= 8:
                    device_mesh = (self.cfg.num_nodes // 8, 8 * gpus_per_node)
                else:
                    device_mesh = (1, self.cfg.num_nodes * gpus_per_node)
                fsdp_kwargs["device_mesh"] = device_mesh
                print(f"[FSDP] sharding={sharding} device_mesh=(replicate={device_mesh[0]}, "
                      f"shard={device_mesh[1]}) num_nodes={self.cfg.num_nodes} "
                      f"gpus_per_node={gpus_per_node}", flush=True)
            # Freezing the depth expert (gf_freeze_depth) mixes requires_grad inside one
            # WanAttentionBlock, which FSDP only allows with use_orig_params=True.
            if getattr(self.cfg, "fsdp_use_orig_params", False):
                fsdp_kwargs["use_orig_params"] = True
                print("[FSDP] use_orig_params=True (mixed requires_grad / expert freeze)", flush=True)

            return FSDPStrategy(**fsdp_kwargs)

        else:
            return self.cfg.strategy

    def download_dataset(self):
        dataset = self._build_dataset("training")
