from yacs.config import CfgNode as CN


_C = CN()
_C.config_name = 'OADIS'

# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
_C.DATASET = CN(new_allowed=True)
# _C.DATASET.name = 'vaw-czsl'
# _C.DATASET.root_dir = '/vaw-czsl'
_C.DATASET.name = 'mitstates'
_C.DATASET.root_dir = '/mit-states'
_C.DATASET.splitname = 'compositional-split-natural'
_C.DATASET.feasibility_threshold=0.05
# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------
_C.MODEL = CN(new_allowed=True)
_C.MODEL.name = 'OADIS'
_C.MODEL.load_checkpoint = False
_C.MODEL.weights = ''
_C.MODEL.optim_weights = ''

# -----------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------
_C.TRAIN = CN(new_allowed=True)

_C.TRAIN.log_dir = "/OADis"
_C.TRAIN.checkpoint_dir = '/OADis/checkpoints'
_C.TRAIN.seed = 124
_C.TRAIN.num_workers = 4

_C.TRAIN.test_batch_size = 32
_C.TRAIN.batch_size = 72
_C.TRAIN.max_epoch = 15
_C.TRAIN.warmup_steps = 0.3
_C.TRAIN.end_lr = 1e-5
_C.TRAIN.decay_power = 1
_C.TRAIN.lr_transformer=1e-5
_C.TRAIN.lr=1e-3
_C.TRAIN.lr_cross=1e-2

_C.TRAIN.disp_interval = 1000
_C.TRAIN.save_every_epoch = 1
_C.TRAIN.eval_every_epoch = 4

# -----------------------------------------------------------------------
# Eval
# -----------------------------------------------------------------------
_C.EVAL = CN(new_allowed=True)
_C.EVAL.topk = 1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()