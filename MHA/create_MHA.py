import torch
import torch.nn as nn 

model = nn.MultiheadAttention(embed_dim=512, num_heads=8)
query = torch.rand(200,1,512)
key = torch.rand(200,1,512)
value = torch.rand(200,1,512)
torch.onnx.export(model, (query, key, value), 'MHA.onnx')