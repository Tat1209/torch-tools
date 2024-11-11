import sys
import random
from pathlib import Path

import torch
import torchvision


work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

root = f"{work_path}/assets/datasets/"

# torchvision.datasets.CIFAR10(root=root, train=True, download=True)
torchvision.datasets.STL10(root=root, download=True)
# torchvision.datasets.Flowers102(root=root, download=True)

