from model import torch_module
import torch
import onnx
import numpy as np
import onnxruntime
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pdb
import time
TRT_LOGGER = trt.Logger()
b=3
MAX_BATCH = 8
input= torch.randn((b,3,400,224),device='cuda')
buffer = [
    torch.randn((b, 3, 100, 56), device='cuda'),
    torch.randn((b, 4, 50, 28), device='cuda'),
    torch.randn((b, 4, 50, 28), device='cuda'),
    torch.randn((b, 8, 25, 14), device='cuda'),
    torch.randn((b, 8, 25, 14), device='cuda'),
    torch.randn((b, 8, 25, 14), device='cuda'),
    torch.randn((b, 12, 25, 14), device='cuda'),
    torch.randn((b, 12, 25, 14), device='cuda'),
    torch.randn((b, 20, 13, 7), device='cuda'),
    torch.randn((b, 20, 13, 7), device='cuda')
]
input_one= torch.ones((1,3,400,224),device='cuda')
buffer_one = [
    torch.ones((1, 3, 100, 56), device='cuda'),
    torch.ones((1, 4, 50, 28), device='cuda'),
    torch.ones((1, 4, 50, 28), device='cuda'),
    torch.ones((1, 8, 25, 14), device='cuda'),
    torch.ones((1, 8, 25, 14), device='cuda'),
    torch.ones((1, 8, 25, 14), device='cuda'),
    torch.ones((1, 12, 25, 14), device='cuda'),
    torch.ones((1, 12, 25, 14), device='cuda'),
    torch.ones((1, 20, 13, 7), device='cuda'),
    torch.ones((1, 20, 13, 7), device='cuda')
]
input_test_one = []
buffer_test_one = []
for i in range(1,MAX_BATCH+1):
    input_test_one.append(torch.ones((i,3,400,224),device='cuda'))
    buffer_test_one.append([
    torch.ones((i, 3, 100, 56), device='cuda'),
    torch.ones((i, 4, 50, 28), device='cuda'),
    torch.ones((i, 4, 50, 28), device='cuda'),
    torch.ones((i, 8, 25, 14), device='cuda'),
    torch.ones((i, 8, 25, 14), device='cuda'),
    torch.ones((i, 8, 25, 14), device='cuda'),
    torch.ones((i, 12, 25, 14), device='cuda'),
    torch.ones((i, 12, 25, 14), device='cuda'),
    torch.ones((i, 20, 13, 7), device='cuda'),
    torch.ones((i, 20, 13, 7), device='cuda')
    ])
torch_result_store = []
start = time.time()
with torch.no_grad():
# torch_outputs = torch_module(input,*buffer)
    for j in range(5):
        for i in range(1,MAX_BATCH+1):
            torch_outputs_one = torch_module(input_test_one[i-1],*buffer_test_one[i-1])
            if j==4:
                torch_result_store.append(list(torch_outputs_one))
    end = time.time()
print("time consume:",end - start)
input_names = ['i0','i1','i2','i3','i4','i5','i6','i7','i8','i9','i10']
output_names=["o" + str(i) for i in range(len(input_names))]
export_onnx_file = 'tsm_dynamic.onnx'
dynamic_axes = {'i0':{0:'batchsize'},'i1':{0:'batchsize'},'i2':{0:'batchsize'},'i3':{0:'batchsize'},
                'i4':{0:'batchsize'},'i5':{0:'batchsize'},'i6':{0:'batchsize'},'i7':{0:'batchsize'},
                'i8':{0:'batchsize'},'i9':{0:'batchsize'},'i10':{0:'batchsize'},
                'o0':{0:'batchsize'},'o1':{0:'batchsize'},'o2':{0:'batchsize'},'o3':{0:'batchsize'},
                'o4':{0:'batchsize'},'o5':{0:'batchsize'},'o6':{0:'batchsize'},'o7':{0:'batchsize'},
                'o8':{0:'batchsize'},'o9':{0:'batchsize'},'o10':{0:'batchsize'}}
# torch.onnx.export(torch_module,(input,*buffer),export_onnx_file,input_names=input_names,output_names=output_names,dynamic_axes=dynamic_axes)

# onnx_model = onnx.load(export_onnx_file)
# onnx.checker.check_model(onnx_model)   #检查文件模型是否正确
# onnx.helper.printable_graph(onnx_model.graph)  #输出计算图
# ort_session = onnxruntime.InferenceSession(export_onnx_file)  #运行一个session
# def to_numpy(input,buffer):
#     input_np = input.cpu().numpy()
#     buffer_np = []
#     for item in buffer:
#         buffer_np.append(item.cpu().numpy())
#     res = []
#     res.append(input_np)
#     res.extend(buffer_np)
#     return res
# ort_inputs = {}
# temp = to_numpy(input,buffer)
# for i,name in enumerate(input_names):
#     ort_inputs[name] = temp[i]
# # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input,buffer)}
# ort_outputs = ort_session.run(None, ort_inputs)
# ort_out = ort_outputs[0]
# print(ort_out)

# torch_output = np.array(output.flatten(),dtype='float32')
# onnx_output = np.array(np.asarray(ort_outputs).flatten(),dtype='float32')
# np.testing.assert_almost_equal(torch_output,onnx_output,decimal=3)  #判断输出的float
# torch_out_numpy = [item.cpu().numpy() for item in torch_outputs]
#
# for i,j in zip(ort_outputs,torch_out_numpy):
#     print(np.linalg.norm(i-j))
# print("successful")


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", fp16_mode=False, int8_mode=False,
               save_engine=True, test_set_fname=None):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network,
                                                                                                             TRT_LOGGER) as parser:
            builder.max_workspace_size = 1000  # 1 <<  # 1GB
            builder.max_batch_size = max_batch_size
            # pdb.set_trace()
            builder.fp16_mode = fp16_mode
            builder.int8_mode = int8_mode
            config = builder.create_builder_config()
            profile = builder.create_optimization_profile()

            if int8_mode:
                exit("Not implemented")

            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parsing_succeed = parser.parse(model.read())

                # if not parsing_succeed:
                #    exit('Failed to parse the ONNX model')
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            # Static input

            profile.set_shape('i0', [1,3,400,224],[5,3,400,224],[MAX_BATCH,3,400,224])
            profile.set_shape('i1', [1,3,100,56],[5,3,100,56],[MAX_BATCH,3,100,56])
            profile.set_shape('i2', [1,4,50,28],[5,4,50,28],[MAX_BATCH,4,50,28])
            profile.set_shape('i3',[1,4,50,28],[5,4,50,28],[MAX_BATCH,4,50,28])
            profile.set_shape('i4',[1,8,25,14],[5,8,25,14],[MAX_BATCH,8,25,14])
            profile.set_shape('i5',[1,8,25,14],[5,8,25,14],[MAX_BATCH,8,25,14])
            profile.set_shape('i6',[1,8,25,14],[5,8,25,14],[MAX_BATCH,8,25,14])
            profile.set_shape('i7',[1,12,25,14],[5,12,25,14],[MAX_BATCH,12,25,14])
            profile.set_shape('i8',[1,12,25,14],[5,12,25,14],[MAX_BATCH,12,25,14])
            profile.set_shape('i9',[1,20,13,7],[5,20,13,7],[MAX_BATCH,20,13,7])
            profile.set_shape('i10',[1,20,13,7],[5,20,13,7],[MAX_BATCH,20,13,7])
            config.add_optimization_profile(profile)

            engine = builder.build_engine(network, config=config)

            if not engine:
                exit('Failed to build the engine')

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
                    print("Completed creating Engine")
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size)


onnx_file_path = 'tsm_dynamic.onnx'
fp16_mode = False
int8_mode = False
# max_batch_size = MAX_BATCH
engine_file_path = "net_fp16_{}_int8_{}_bs_{}.trt".format(fp16_mode, int8_mode, MAX_BATCH)

print("Building Engine")

calibration_stream = None

engine = get_engine(MAX_BATCH, onnx_file_path, engine_file_path, fp16_mode=fp16_mode, int8_mode=int8_mode,
                    test_set_fname=None)
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        #pdb.set_trace()
        size = trt.volume(engine.get_binding_shape(binding)[1:]) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes) # Only bytes, no need for size
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()


def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
start = time.time()
trt_out_store = []
for j in range(5):
    for batchsize in range(1, MAX_BATCH + 1):
        i0 = np.ones([batchsize, 3, 400, 224]).astype(np.float32)
        i1 = np.ones([batchsize, 3, 100, 56]).astype(np.float32)
        i2 = np.ones([batchsize, 4, 50, 28]).astype(np.float32)
        i3 = np.ones([batchsize, 4, 50, 28]).astype(np.float32)
        i4 = np.ones([batchsize, 8, 25, 14]).astype(np.float32)
        i5 = np.ones([batchsize, 8, 25, 14]).astype(np.float32)
        i6 = np.ones([batchsize, 8, 25, 14]).astype(np.float32)
        i7 = np.ones([batchsize, 12, 25, 14]).astype(np.float32)
        i8 = np.ones([batchsize, 12, 25, 14]).astype(np.float32)
        i9 = np.ones([batchsize, 20, 13, 7]).astype(np.float32)
        i10 = np.ones([batchsize, 20, 13, 7]).astype(np.float32)
        for i in range(len(inputs)):
            inputs[i].host = eval('i'+str(i)).data
        # x = np.ones([batchsize, ] + input_shape).astype(np.float32)
        #
        # inputs[0].host = x.data
        context.set_binding_shape(0, [batchsize, ] + [3,400,224])
        context.set_binding_shape(1, [batchsize, ] + [3, 100, 56])
        context.set_binding_shape(2, [batchsize, ] + [4, 50, 28])
        context.set_binding_shape(3, [batchsize, ] + [4, 50, 28])
        context.set_binding_shape(4, [batchsize, ] + [8, 25, 14])
        context.set_binding_shape(5, [batchsize, ] + [8, 25, 14])
        context.set_binding_shape(6, [batchsize, ] + [8, 25, 14])
        context.set_binding_shape(7, [batchsize, ] + [12, 25, 14])
        context.set_binding_shape(8, [batchsize, ] + [12, 25, 14])
        context.set_binding_shape(9, [batchsize, ] + [20, 13, 7])
        context.set_binding_shape(10, [batchsize, ] + [20, 13, 7])
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        if j==4:
            trt_out_store.append(trt_outputs)
        # print("ik")
        # print('\nBatchSize=' + str(batchsize) + '\n', trt_outputs[0].reshape(-1, num_clasess), )
        """
        You will see the some output like (the value may be different):
            BatchSize=1
            [[-0.4005382   0.37080884]
            [ 0.          0.        ]
            [ 0.          0.        ]
            [ 0.          0.        ]]

            BatchSize=2
            [[-0.4005382   0.37080884]
            [-0.4005382   0.37080884]
            [ 0.          0.        ]
            [ 0.          0.        ]]

            BatchSize=3
            [[-0.4005382   0.37080884]
            [-0.4005382   0.37080884]
            [-0.4005382   0.37080884]
            [ 0.          0.        ]]

            BatchSize=4
            [[-0.4005382   0.37080884]
            [-0.4005382   0.37080884]
            [-0.4005382   0.37080884]
            [-0.4005382   0.37080884]]

        ! The output_shape depends how you allocate the max batch size and memory. 
        ! Zeros will be filled to where the batch dimensions > batchsize   
    """

end = time.time()
print("time consume",end-start)
torch_result_store_numpy = []
for i in torch_result_store:
    temp = []
    for j in i:
        x = j.cpu().numpy()
        temp.append(x)
    torch_result_store_numpy.append(temp)
# pdb.set_trace()
shape_dict = {0:[7],1:[3,100,56],2:[4,50,28],
              3:[4,50,28],4:[8,25,14],5:[8,25,14],
              6:[8,25,14],7:[12,25,14],8:[12,25,14],
              9:[20,13,7],10:[20,13,7]}
trt_out_store_numpy = []
for batchsize,i in enumerate(trt_out_store):
    new_i = []
    new_i.append(i[-1])
    for item in i[:-1]:
        new_i.append(item)

    # new_i = i[-1].extend(i[1:])
    temp = []
    for id,j in enumerate(new_i):
        x = j.reshape([-1]+shape_dict[id])
        x = x[:batchsize+1,:]
        temp.append(x)
    trt_out_store_numpy.append(temp)


# 'i1', [1,3,100,56],[5,3,100,56],[MAX_BATCH,3,100,56],
# 'i2', [1,4,50,28],[5,4,50,28],[MAX_BATCH,4,50,28],
# 'i3',[1,4,50,28],[5,4,50,28],[MAX_BATCH,4,50,28],
# 'i4',[1,8,25,14],[5,8,25,14],[MAX_BATCH,8,25,14],
# 'i5',[1,8,25,14],[5,8,25,14],[MAX_BATCH,8,25,24],
# 'i6',[1,8,25,14],[5,8,25,14],[MAX_BATCH,8,25,24],
# 'i7',[1,12,25,14],[5,12,25,14],[MAX_BATCH,12,25,24],
# 'i8',[1,12,25,14],[5,12,25,14],[MAX_BATCH,12,25,24],
# 'i9',[1,20,13,7],[5,20,13,7],[MAX_BATCH,20,13,7],
# 'i10',[1,20,13,7],[5,20,13,7],[MAX_BATCH,20,13,7])
for i,j in zip(torch_result_store_numpy,trt_out_store_numpy):
    for i1,j1 in zip(i,j):
        print(np.linalg.norm(i1-j1))
print("successful")
print('debug')