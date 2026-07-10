"""
Experiment configuration for pose-conditioned post-training.
"""

from hydra.core.config_store import ConfigStore
from cosmos_predict2.data.pose_conditioned import PoseConditionedDataset
from torch.utils.data import DataLoader, DistributedSampler
from megatron.core import parallel_state
from imaginaire.lazy_config import LazyCall as L

cs = ConfigStore.instance()

DATA_ROOT = "/PATH/TO/DATA" # containing train and val folders
BATCH_SIZE = 4 # xGPU

NUM_FRAMES=45
COND_FRAMES=13
RES = (480, 480) # 1:1 aspect ratio

train_dataset = PoseConditionedDataset(
    video_root=DATA_ROOT,
    split="train",
    num_frames=NUM_FRAMES,
    num_conditioning_frames=COND_FRAMES,
    video_size=RES,
    use_precomputed_latents=True,
    latents_fps=16,
    mode="train"
)

val_dataset = PoseConditionedDataset(
    video_root=DATA_ROOT,
    split="val",
    num_frames=NUM_FRAMES,
    num_conditioning_frames=COND_FRAMES,
    video_size=RES,
    use_precomputed_latents=True,
    latents_fps=16,
    mode="val"
)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


dataloader_train = L(DataLoader)(
    dataset=train_dataset,
    sampler=L(get_sampler)(dataset=train_dataset),
    batch_size=BATCH_SIZE,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

dataloader_val = L(DataLoader)(
    dataset=val_dataset,
    sampler=L(get_sampler)(dataset=val_dataset),
    batch_size=BATCH_SIZE,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

predict2_video2world_2b_pose_control = dict(
    defaults=[
        {"override /model": "predict2_v2w_2b_pose_conditioned_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=-1,
            # train_architecture="lora",  # Use "lora" for LoRA training
            pipe_config=dict(
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
            ),
            # model_manager_config=dict( # Restart from given checkpoint
            #     dit_path="/path/to/ckpt.pt"
            # ),
        )
    ),
    job=dict(project="posttraining", group="pose-control", name="exp_1"),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train,
    dataloader_val=dataloader_val,
    trainer=dict(
        distributed_parallelism="fsdp",
        run_validation=False,
        logging_iter=100,
        validation_iter=500,
        max_iter=30000,
        grad_accum_iter=1,
    ),
    optimizer=dict(
        lr=1.1e-5,
    ),
    scheduler=dict(
        f_max=[1.0],
        f_min=[0.7],
        warm_up_steps=[250],
        cycle_lengths=[30_000],
    ),
    checkpoint=dict(
        save_iter=1500,
        load_ema_to_reg=True,
        load_training_state=True,
    )
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_2b_pose_control", # Global experiment name identifier (use it as EXP in the launch script)
    node=predict2_video2world_2b_pose_control,
)
