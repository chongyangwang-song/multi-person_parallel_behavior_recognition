# Multi-person parallel behavior recognition

![Demo](https://github.com/chongyangwang-song/multi-person_parallel_behavior_recognition/blob/main/image/%E8%84%9A%E8%B8%A2%E5%88%BA%E6%9D%80%E8%BE%93%E5%87%BA%5B00-00-09--00-00-11%5D.gif)

## 1. Behavior recognition model

We use TSM, which is a popular method in behavior recognition. [The paper link](https://arxiv.org/abs/1811.08383),  The original [code link](https://github.com/mit-han-lab/temporal-shift-module)

Note that TSM has two versions, One is offline version and another is online verson. I implement the online version.

## 2. My work

**1.multi-person parallel inference:**

The original code is just single scene inference,  My work implements multi-person parallel inference. For TSM online version， it works like RNN，The current frame inference relies on the current frame information and historical information.  For multi-person behavior recognition, the difficulty lies in how to find the historical information of the corresponding person. To do this, I designed a history hash table to get the history information based on the person's ID.

![picture](https://github.com/chongyangwang-song/multi-person_parallel_behavior_recognition/blob/main/image/TSMonline.png)

You can use Yolo to get the bounding box of the person，then you can get the RGB information of a person from the original scene.

You can use SORT to track preson, then you can get the ID information.

All in all, for a scene frame, you can get RGB and ID information for each person.

**2.Cache information is dynamically cleared**

For real-time tasks, if a person exits the camera, that person's historical information will always remain in the cache, which makes the cache become lager and lager. We counted each person and deleted the person's history when that person did not appear on the camera for a specified number of frames.

**3.We use TensorRT to accelerate the model**

In tests,TensorRT model is four times faster than the Torch model

# Documents Description

### ActionRecognition

1.mobilenet_v2_tsm.py: In this script, we define the TSM online model，which use MobileNetV2 as backbone.

2.model.py: Some auxiliary tools, including data processing. In this script，Buffer is the hashtable saving historical information.

3.myTransform.py: In this script, you can transfer torch model to onnx，and then you can use TensorRT to get an engine.

4.trt_inference.py:  Encapsulation of inference algorithm.

5.main.py: Online behavior recognition algorithm main program

## tool and ops

some useful packges
