from mmengine.registry import Registry

OPENOCC_LOSS = Registry("openocc_loss")

from .ce_loss import CeLoss
from .emb_loss import VQVAEEmbedLoss
from .multi_loss import MultiLoss
from .plan_reg_loss_lidar import PlanRegLossLidar
from .recon_loss import LovaszLoss, ReconLoss
