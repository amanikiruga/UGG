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
        wan_t2v_ray_depth_mot_concat_5b=WanTextToVideoRayDepthMoTConcat5B,
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
            # PG timeout: default NCCL watchdog is 10min, which a slow 14B first step
            # (compile / inductor+Triton compilation, large gckpt activations, rank-0
            # logging) can exceed -> false collective-timeout abort (saw this at 14B,
            # a rank-0 straggler that never reached the first-step allreduce).
            # Raise to 30min. Override via +experiment.pg_timeout_min=N.
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
                # Peak-memory reduction (math-preserving): default BACKWARD_PRE prefetches the NEXT unit's full
                # params while the current backward still holds its own -> ~2 units resident. BACKWARD_POST +
                # limit_all_gathers keeps ~1 unit resident, saving ~one layer of full params (~0.5GB at 14B).
                # 14B b=1 @ n_frames=50 OOM'd by only 134MiB on 1 node (4-way shard) -> this closes it.
                # Override via +experiment.fsdp_backward_prefetch_post=false.
                backward_prefetch=(
                    BackwardPrefetch.BACKWARD_POST
                    if getattr(self.cfg, "fsdp_backward_prefetch_post", True)
                    else BackwardPrefetch.BACKWARD_PRE
                ),
                limit_all_gathers=True,
                # "full" (not "sharded"): sharded checkpoints omit the frozen VAE and DCP's
                # load planner dies on the missing keys BEFORE Lightning consults `strict`,
                # so no config knob rescues a resume. "full" gathers the whole module state
                # and is self-consistent; its host-RAM cost is a memory budget problem,
                # addressed by --mem=1400G. Override: +experiment.fsdp_state_dict_type=...
                state_dict_type=getattr(self.cfg, "fsdp_state_dict_type", "full"),
            )
            # CPUOffload (opt-in): 14B b=1 @ 50 frames is ~100-166MiB over even on 8xH200 -- the peak (~137GB/rank)
            # is non-shardable (activations + grad/optim-step transient) and allocator tuning (expandable_segments,
            # garbage_collection_threshold) can't reclaim enough. CPUOffload keeps sharded params+grads on CPU and
            # runs the optimizer step on CPU -> frees ~16-21GB/rank on 8-way -> reliable fit. Math-preserving (same
            # fp32 AdamW, just on CPU); slower (param H2D per unit). Enable with +experiment.fsdp_cpu_offload=true.
            if getattr(self.cfg, "fsdp_cpu_offload", False):
                fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
            # HYBRID variants need a 2D device mesh; FULL_SHARD/SHARD_GRAD_OP/NO_SHARD use 1D world.
            if "HYBRID" in sharding:
                if getattr(self.cfg, "fsdp_shard_within_node", False):
                    # True hybrid for models that fit in one node: shard within each
                    # node (intra-node NVLink all-gather), replicate across nodes (only
                    # a grad all-reduce between nodes). mesh=(replicate=num_nodes,
                    # shard=gpus_per_node). Opt-in via +experiment.fsdp_shard_within_node=true.
                    device_mesh = (self.cfg.num_nodes, gpus_per_node)
                elif self.cfg.num_nodes >= 8:
                    device_mesh = (self.cfg.num_nodes // 8, 8 * gpus_per_node)
                else:
                    device_mesh = (1, self.cfg.num_nodes * gpus_per_node)
                fsdp_kwargs["device_mesh"] = device_mesh
                print(f"[FSDP] sharding={sharding} device_mesh=(replicate={device_mesh[0]}, "
                      f"shard={device_mesh[1]}) num_nodes={self.cfg.num_nodes} "
                      f"gpus_per_node={gpus_per_node}", flush=True)
            # GF (freeze depth, train rgb/ray) puts MIXED requires_grad inside a single
            # WanAttentionBlock. With the PyTorch default (use_orig_params=False) FSDP
            # flattens each block into one FlatParameter and requires uniform requires_grad,
            # so wrapping raises outright. Passed ONLY when enabled, so every existing run
            # (F included) is bit-identical.
            if getattr(self.cfg, "fsdp_use_orig_params", False):
                fsdp_kwargs["use_orig_params"] = True
                print("[FSDP] use_orig_params=True (mixed requires_grad / expert freeze)", flush=True)

            return FSDPStrategy(**fsdp_kwargs)

        else:
            return self.cfg.strategy

    def download_dataset(self):
        dataset = self._build_dataset("training")
