import torch
import numpy as np
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Torch version:", torch.__version__)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Num GPUs:", torch.cuda.device_count())
print(torch.cuda.memory_allocated())