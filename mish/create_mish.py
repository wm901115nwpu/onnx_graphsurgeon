import torch
import torch.nn as nn 

model = nn.Mish()

torch.onnx.export(model, torch.rand(1,3,224,224), 'mish.onnx')