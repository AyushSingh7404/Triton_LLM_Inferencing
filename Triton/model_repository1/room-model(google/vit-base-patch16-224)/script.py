import torch

# Load TorchScript model
model_path = r"C:\Users\rajpu\Desktop\Trinton\model_repository\room_model\1\model.pt"
model = torch.jit.load(model_path)

print(model.graph)
