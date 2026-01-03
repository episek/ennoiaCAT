import torch

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.rand(4000, 4000, device=device)
y = torch.mm(x, x)
print("Result shape:", y.shape)
print("Result device:", y.device)
