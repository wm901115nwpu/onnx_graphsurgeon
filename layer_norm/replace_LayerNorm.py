import numpy as np
from copy import deepcopy
import onnx_graphsurgeon as gs
import onnx

# fmt: off
'''
             x
         /      \
         |  ReduceMean
         \      /
            Sub
         /      \
         |     Pow
         |      |
         |  ReduceMean
         |      |
         |     Add
         |      |
         |     Sqrt
         \      /
            Div
             |
            Mul
             |
            Add
'''
# Now we'll do the actual replacement
graph = gs.import_onnx(onnx.load("./layer_norm.onnx"))

tmap = graph.tensors()
nLayerNormPlugin = 0
for node in graph.nodes:
    if node.op == 'ReduceMean' and \
        node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
        node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
        node.o().o(0).o().op == 'ReduceMean' and \
        node.o().o(0).o().o().op == 'Add' and \
        node.o().o(0).o().o().o().op == 'Sqrt' and \
        node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1) and \
        node.o().o(0).o().o().o().o().o().op == 'Mul' and \
        node.o().o(0).o().o().o().o().o().o().op == 'Add':

        inputTensor = node.inputs[0]

        lastMultipyNode = node.o().o(0).o().o().o().o().o()
        index = ['weight' in i.name for i in lastMultipyNode.inputs].index(True)
        b = np.array(deepcopy(lastMultipyNode.inputs[index].values.tolist()), dtype=np.float32)
        constantB = gs.Constant("LayerNormB-" + str(nLayerNormPlugin), np.ascontiguousarray(b.reshape(-1)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

        lastAddNode = node.o().o(0).o().o().o().o().o().o()
        index = ['bias' in i.name for i in lastAddNode.inputs].index(True)
        a = np.array(deepcopy(lastAddNode.inputs[index].values.tolist()), dtype=np.float32)
        constantA = gs.Constant("LayerNormA-" + str(nLayerNormPlugin), np.ascontiguousarray(a.reshape(-1)))

        inputList = [inputTensor, constantB, constantA]
        layerNormV = gs.Variable("LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float32), None)
        layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=inputList, outputs=[layerNormV])
        graph.nodes.append(layerNormN)

        if lastAddNode.outputs[0] in graph.outputs:  # the last LayerNorm provide one of the graph's output, and do not unsqueeze to 4 dimension
            # oldLastAdd -> graph.outputs[0] ===> LayerNorm -> Squeeze -> graph.outputs[0]
            layerNormN.outputs[0].name = 'encoder_out'
            index = graph.outputs.index(lastAddNode.outputs[0])
            graph.outputs[index] = layerNormN.outputs[0]
        else:  # other LayerNorm contain the subsequent Squeeze operation
            for n in graph.nodes:
                if lastAddNode.outputs[0] in n.inputs:
                    index = n.inputs.index(lastAddNode.outputs[0])
                    n.inputs[index] = layerNormN.outputs[0]

            lastAddNode.outputs = []

        nLayerNormPlugin += 1
        continue
    
# Remove the now-dangling subgraph.
graph.cleanup().toposort()

# That's it!
onnx.save(gs.export_onnx(graph), "layer_norm_replaced.onnx")
