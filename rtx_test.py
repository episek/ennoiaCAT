import torch, platform

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("built with CUDA:", torch.backends.cuda.is_built())
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("OS:", platform.system(), platform.release())
