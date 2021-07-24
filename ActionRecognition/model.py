from ActionRecognation.mobilenet_v2_tsm import MobileNetV2
import torch
from ops.transforms import *
from collections import defaultdict
import copy
##########################################################################
test_compile_path = 'weights/ckpt.best.pth.tar'
# net.load_state_dict()
net = MobileNetV2(n_class=7)
store = torch.load(test_compile_path)
checkpoint = store['state_dict']
new_checkpoint = {k.replace('module.base_model.', ''): v for k, v in checkpoint.items()}
new_checkpoint = {k.replace('module.new_fc.', 'classifier.'): v for k, v in new_checkpoint.items()}
new_checkpoint = {k.replace('net.', ''): v for k, v in new_checkpoint.items()}
# new_checkpoint = {k.replace('base_model.',''):v for k,v in new_checkpoint.items()}
net.load_state_dict(new_checkpoint)
torch_module = net.cuda()
torch_module.eval()
##############################################################################
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
my_dict = {0: '刺杀', 1: '射击', 2: '投掷', 3: '拳打',
           4: '掐脖子', 5: '脚踢', 6: '正常'}
catigories = ["stabing",
              "shooting",
              "throwing",
              "boxing",
              "squeeze_neck",
              "kicking",
              "normal"]

class Buffer:
    def __init__(self):
        self.buffer = [
                        torch.zeros((1,3, 100, 56), device='cuda'),
                        torch.zeros((1,4, 50, 28), device='cuda'),
                        torch.zeros((1,4, 50, 28), device='cuda'),
                        torch.zeros((1,8, 25, 14), device='cuda'),
                        torch.zeros((1,8, 25, 14), device='cuda'),
                        torch.zeros((1,8, 25, 14), device='cuda'),
                        torch.zeros((1,12, 25, 14), device='cuda'),
                        torch.zeros((1,12, 25, 14), device='cuda'),
                        torch.zeros((1,20, 13, 7), device='cuda'),
                        torch.zeros((1,20, 13, 7), device='cuda')
                     ]
        self.maxnum = 7
        self.bufferDict = {}
        self.countDict = defaultdict(int)
        self.flagDict = {}
        self.history = defaultdict(list)
        self.history_logit = defaultdict(list)
        self.num = 0
    def add(self,ids_):
        # temp = list(self.bufferDict.keys())
        # for k in self.countDict.keys():
        #     self.countDict[k] += 1
        self.num += 1
        for id in ids_:
            if id not in self.bufferDict.keys():
                # self.history[id] = [6,6]
                self.bufferDict[id] = copy.deepcopy(self.buffer)
            # self.count[id] = 0
        # temp = self.ids
        # self.ids.union(ids_)
        # for id in temp:
        #     if id not in ids_:
        #         self.count[id] += 0

    def getbuffer(self,ids):
        result = [torch.Tensor().cuda()]*10
        for id in ids:
            for i in range(10):
                result[i] = torch.cat((result[i],self.bufferDict[id][i]),dim=0)
        return result

    def update(self,ids,buffer):
        for i,id in enumerate(ids):
            temp = []
            for j in range(10):
                temp.append(buffer[j][i].unsqueeze(0))
            self.bufferDict[id] = temp

    def makeLabel(self,ids,feat):
        result = []
        for i,id in enumerate(ids):
            self.history_logit[id].append(feat[i].detach().cpu().unsqueeze(0).numpy())
            self.history_logit[id] = self.history_logit[id][-12:]
            avg_logit = sum(self.history_logit[id])
            idx_ = np.argmax(avg_logit, axis=1)[0]
            result.append(idx_)
        return result

    def delete(self):
        if self.num == 4:
            self.bufferDict = {}









