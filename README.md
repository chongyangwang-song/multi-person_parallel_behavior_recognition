# Multi-person parallel behavior recognition

![](C:\Users\lenovo\Desktop\找工作相关\项目展示\脚踢刺杀输出[00-00-09--00-00-11].gif)

## 1. Behavior recognition model

We use TSM, which is a popular method in behavior recognition. [The paper link](https://arxiv.org/abs/1811.08383),  The original [code link](https://github.com/mit-han-lab/temporal-shift-module)

Note that TSM has two version, One is offline version and another is online verson. I implement the online version.

## 2. My work

**1.multi-person parallel inference:**

The original code is just single scene inference,  My work implements multi-person parallel inference. For TSM online version， it works like RNN，The current frame inference relies on the current frame information and historical information.  For multi-person behavior recognition, the difficulty lies in how to find the historical information of the corresponding person. To do this, I designed a history hash table to get the history information based on the person's ID.

<img src="C:\Users\lenovo\Desktop\找工作相关\TSMonline.png" alt="TSMonline" style="zoom:50%;" />

You can use Yolo to get the bounding box of the person，then you can get the RGB information of a person from a original scene.

You can use SORT to track preson, then you can get the ID information.

All in all, for a scene frame, you can get RGB and ID information for each person.

**2.Cache information is dynamically cleared**

For real-time tasks, if a person exits the camera, that person's historical information will always remain in the cache, which makes the cache become lager and lager. We counted each person and deleted the person's history when that person did not appear on the camera for a specified number of frames.

**3.We use TensorRT to accelerate the model**

In tests,TensorRT was four times faster than the Torch model