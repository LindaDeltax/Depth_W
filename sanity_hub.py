import torch
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import get_image_from_url, colorize

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from pprint import pprint

# Trigger reload of MiDaS
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 

model = torch.hub.load(".", "ZoeD_K", source="local", pretrained=True)
