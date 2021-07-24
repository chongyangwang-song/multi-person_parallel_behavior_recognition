from trt_inference import TSM_ONLINE
def cameradetect(self):
    buffer = Buffer()

    tsm = TSM_ONLINE('./engine/tsm_dynamic.trt')
    #you can use yolo to get frames,and use sort to get ids.
    while(True):
        buffer.add(ids)
        buffer_select = buffer.getbuffer(ids)
        result = process_frame(frames)
        if (len(result) != 0):
            outputs1 = tsm.go(result,buffer_select)
            temp = []
            for item in outputs1:
                temp.append(torch.from_numpy(item).cuda())
            feat,buffer_new = temp[0],temp[1:]
            buffer.update(ids, buffer_new)
            label = buffer.makeLabel(ids, feat)
            label = [catigories[i] for i in label]


