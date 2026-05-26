from cosmos_predict2.configs.pose_conditioned.pipeline import get_cosmos_predict2_pose_conditioned_pipeline

from cosmos_predict2.models.video2world_pose_model import Predict2Video2WorldPoseConditionedModel

from hydra.core.config_store import ConfigStore

from cosmos_predict2.models.video2world_model import Predict2ModelManagerConfig, Predict2Video2WorldModelConfig
from imaginaire.constants import get_cosmos_predict2_pose_conditioned_checkpoint
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.lazy_config import LazyCall as L


_PREDICT2_V2W_2B_POSE_CONDITIONED_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldPoseConditionedModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=get_cosmos_predict2_pose_conditioned_pipeline(model_size="2B", resolution="480", fps=16),
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path=get_cosmos_predict2_video2world_checkpoint(model_size="2B", resolution="480", fps=16),
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=-1,
        ),
        _recursive_=False,
    ),
)


def register_model_pose_conditioned() -> None:
    cs = ConfigStore.instance()
    # predict2 v2w 2b model
    cs.store(
        group="model",
        package="_global_",
        name="predict2_v2w_2b_pose_conditioned_fsdp",
        node=_PREDICT2_V2W_2B_POSE_CONDITIONED_FSDP_CONFIG,
    )