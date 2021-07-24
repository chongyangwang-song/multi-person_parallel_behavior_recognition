from tool.infertools import *
import numpy as np


class TSM_ONLINE():

    def __init__(self,trt_pth):
        self.pth = trt_pth
        self.engine = get_engine(self.pth)
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers_tsm(self.engine)
        self.context = self.engine.create_execution_context()
        self.shape_dict = {0: [7], 1: [3, 100, 56], 2: [4, 50, 28],
                      3: [4, 50, 28], 4: [8, 25, 14], 5: [8, 25, 14],
                      6: [8, 25, 14], 7: [12, 25, 14], 8: [12, 25, 14],
                      9: [20, 13, 7], 10: [20, 13, 7]}
        print("tsm trt model initialize finished")
    def go(self,input,buffer):
        input_numpy = input.cpu().numpy().astype(np.float32)
        batchsize = input_numpy.shape[0]
        self.inputs[0].host = input_numpy.data
        for i,data in enumerate(buffer):
            self.inputs[i+1].host = data.cpu().numpy().data
        self.context.set_binding_shape(0, [batchsize, ] + [3, 400, 224])
        self.context.set_binding_shape(1, [batchsize, ] + [3, 100, 56])
        self.context.set_binding_shape(2, [batchsize, ] + [4, 50, 28])
        self.context.set_binding_shape(3, [batchsize, ] + [4, 50, 28])
        self.context.set_binding_shape(4, [batchsize, ] + [8, 25, 14])
        self.context.set_binding_shape(5, [batchsize, ] + [8, 25, 14])
        self.context.set_binding_shape(6, [batchsize, ] + [8, 25, 14])
        self.context.set_binding_shape(7, [batchsize, ] + [12, 25, 14])
        self.context.set_binding_shape(8, [batchsize, ] + [12, 25, 14])
        self.context.set_binding_shape(9, [batchsize, ] + [20, 13, 7])
        self.context.set_binding_shape(10, [batchsize, ] + [20, 13, 7])
        trt_outputs = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        newdata = []
        newdata.append(trt_outputs[-1])
        for item in trt_outputs[:-1]:
            newdata.append(item)
        result = []
        for id, j in enumerate(newdata):
            x = j.reshape([-1] + self.shape_dict[id])
            x = x[:batchsize, :]
            result.append(x)
        return result