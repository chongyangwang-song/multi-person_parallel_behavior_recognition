import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from tool.utils import *
from tool.genarate import *
import torch
from ops.transforms import *
TRT_LOGGER = trt.Logger()

def GiB(val):
    return val * 1 << 30

input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]
normalize = GroupNormalize(input_mean, input_std)
transform = torchvision.transforms.Compose([
    GroupScale([457, 256]),
    GroupCenterCrop((400, 224)),
    Stack(roll=False),
    ToTorchFormatTensor(div=True),
    normalize,
])

def updateBuffer(ids,buffer,s):
    for id in ids:
        if id not in s:
            for i in range(len(buffer)):
                shape = list(buffer[i].shape)
                shape[0] = 1
                buffer[i] = torch.cat((buffer[i],torch.zeros(tuple(shape),device='cuda')),dim=0)
    mask = []
    for id in s:
        if id not in ids:
            mask.append(False)
    s = s.union(set(ids))
    for i in range(len(s)-len(mask)):
        mask.append(True)
    mask = torch.Tensor(mask)
    mask = mask.bool()
    for i in range(len(buffer)):
        buffer[i] = buffer[i][mask]
    return s

def process_frame(frames):
    result = torch.Tensor()
    for frame in frames:
        frame = transform([frame])
        frame = frame.unsqueeze(0)
        result = torch.cat((result,frame),dim=0)
    result = result.cuda()
    return result

def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    # if idx_ in [7, 8, 21, 22, 3]:
    #     idx_ = history[-1]

    # use only single no action class
    # if idx_ == 0:
    #     idx_ = 2

    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]):  # and history[-2] == history[-3]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history

def find_sample_data(description="Runs a TensorRT Python sample",
                     subfolder="",
                     find_files=[]):
    '''
    Parses sample arguments.
    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.
    Returns:
        str: Path of data directory.
    Raises:
        FileNotFoundError
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d",
                        "--datadir",
                        help="Location of the TensorRT sample data directory.",
                        default=kDEFAULT_DATA_ROOT)
    args, unknown_args = parser.parse_known_args()

    # If data directory is not specified, use the default.
    data_root = args.datadir
    # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
    subfolder_path = os.path.join(data_root, subfolder)
    data_path = subfolder_path
    if not os.path.exists(subfolder_path):
        print("WARNING: " + subfolder_path + " does not exist. Trying " +
              data_root + " instead.")
        data_path = data_root

    # Make sure data directory exists.
    if not (os.path.exists(data_path)):
        raise FileNotFoundError(
            data_path +
            " does not exist. Please provide the correct data path with the -d option."
        )

    # Find all requested files.
    for index, f in enumerate(find_files):
        find_files[index] = os.path.abspath(os.path.join(data_path, f))
        if not os.path.exists(find_files[index]):
            raise FileNotFoundError(
                find_files[index] +
                " does not exist. Please provide the correct data path with the -d option."
            )

    return data_path, find_files


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def allocate_buffers_tsm(engine):
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


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
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

def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


#@cal_time
def detect(batchsize, data, context, buffers, num_classes, x, y, w_r, h_r):
    inputs, outputs, bindings, stream = buffers
    inputs[0].host = data
    trt_outputs = do_inference(context,
                               bindings=bindings,
                               inputs=inputs,
                               outputs=outputs,
                               stream=stream)
    trt_outputs[0] = trt_outputs[0].reshape(batchsize, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(batchsize, -1, num_classes)
    trt_outputs[0] = boxmap(trt_outputs[0], x, y, w_r, h_r)
    boxes = post_processing(0.25, 0.85, trt_outputs)
    boxes = nms(boxes, 0.75,min_mode=True)
    return boxes
