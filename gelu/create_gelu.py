import torch
import torch.nn as nn 

model = nn.GELU()

torch.onnx.export(model, torch.rand(1,3,224,224), 'gelu.onnx')