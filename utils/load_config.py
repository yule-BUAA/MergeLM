import torch

if torch.cuda.is_available():
    cache_dir = "/mnt/data/yule/.cache"
else:
    cache_dir = "/Users/yule/.cache"
