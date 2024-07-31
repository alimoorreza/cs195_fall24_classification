# my_project/config.py
# please visit for details: https://github.com/rbgirshick/yacs

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.BASE_DIR = ''
_C.GPUS = (0,)
_C.DEVICE = 'cuda'
_C.WORKERS = 4
_C.CLASSES = 2

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'alexnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''


# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN_SET = ''
_C.DATASET.TEST_SET = ''
_C.DATASET.NUM_CLASSES = -1
_C.DATASET.IMAGE_H = 227
_C.DATASET.IMAGE_H = 227


# train
_C.TRAIN = CN()

_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

_C.TRAIN.ARCH = 'alexnet'
_C.TRAIN.TRAIN_H = 227
_C.TRAIN.TRAIN_W = 227
_C.TRAIN.IMAGE_SIZE = [227, 227]
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.BATCH_SIZE_VAL = 1

_C.TRAIN.END_EPOCH = 10
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.OPTIMIZER = "adam"
_C.TRAIN.BASE_LR = 0.001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.POWER = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0005
_C.TRAIN.GAMMA = 0.1
_C.TRAIN.NESTEROV = False

_C.TRAIN.PRINT_FREQ = 100
_C.TRAIN.SAVE_FREQ = 1
_C.TRAIN.SAVE_PATH = ''
_C.TRAIN.RESUME = True
_C.TRAIN.AUTO_RESUME = None
_C.TRAIN.EVALUATE = True

_C.TRAIN.MODEL_SAVE_DIR = "../models/"

_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 1
_C.TEST.MODEL_FILE = ''
_C.TEST.IOU = 0.5
_C.TEST.IMAGE_SIZE = [227, 227]
_C.TEST.BATCH_SIZE = 32


_C.TRAIN.EPISODE = 1000

_C.TRAIN.LAYERS = 50
_C.TRAIN.SYNC_BN = False
_C.TRAIN.SCALE_MIN = 0.8
_C.TRAIN.SCALE_MAX = 1.25
_C.TRAIN.ROTATE_MIN = -10
_C.TRAIN.ROTATE_MAX = 10
_C.TRAIN.ZOOM_FACTOR = 8
_C.TRAIN.IGNORE_LABEL = 255
_C.TRAIN.PADDING_LABEL = 255
_C.TRAIN.AUX_WEIGHT = 1.0
_C.TRAIN.WORKERS = 2
_C.TRAIN.WEIGHT = None
_C.TRAIN.MAX_SP = 5
_C.TRAIN.TRAIN_ITER = 10
_C.TRAIN.EVAL_ITER = 5
_C.TRAIN.PYRAMID = True
_C.TRAIN.PPM_SCALES = [1.0, 0.5, 0.25, 0.125]
_C.TRAIN.WARMUP = False
_C.TRAIN.ORI_RESIZE = True

_C.TRAIN.FIX_RANDOM_SEED_VAL = True
_C.TRAIN.MANUAL_SEED = 123

_C.TRAIN.RESNET_PRETRAINED_MODEL = ''

_C.TRAIN.SCALE_VAL = 1
_C.TRAIN.VGG = False
_C.TRAIN.PRETRAINED_MODEL = ''
_C.TRAIN.PA_NET_TYPE = ''
_C.TRAIN.NEW_NET_BACKBONE_TYPE = ''
_C.TRAIN.RANDOM_SPLIT_TRAIN = True
_C.TRAIN.RANDOM_SPLIT_EVAL = False
_C.TRAIN.HSNET_BB = ''
_C.TRAIN.ASNET_BB = ''


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`