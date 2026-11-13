"""Minimal environment control, not a K1 performance measurement."""
import torch
print('TORCH_IMPORTED',flush=True)
x=torch.arange(32,device='cuda',dtype=torch.float32)
torch.cuda.synchronize()
assert x.sum().item()==496
print('MINIMAL_TORCH_CUDA_PASS',flush=True)
