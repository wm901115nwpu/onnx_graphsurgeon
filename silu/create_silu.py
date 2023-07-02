import torch
import torch.nn as nn 

model = nn.SiLU()

torch.onnx.export(model, torch.rand(1,3,224,224), 'silu.onnx')