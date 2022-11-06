# YOLOv1目录
[1.为什么叫YOLO](./README.md/#1.为什么叫YOLO)

[3.YOLO简介](./README.md/#3.YOLO简介)

[4.YOLO的原理](./README.md/#4.YOLO的原理)

## 1.为什么叫YOLO
You Only Look Once: Unified, Real-Time Object Detection

You Only Look Once表示one stage；
Unified表示统一的架构，相对RCNN来说，其只有一个网络；
Real-Time表示算法速度快，能满足实时性要求

## 2.YOLO的灵感来源
### 2.1.分析人类的视觉系统：

人类看一张图片的时候，可以立刻知道物体是什么，在哪里，在做什么。
人类的视觉系统是快速的、准确的、不需要太多思考的。

###2.2.分析当前目标检测模型：

当前的一些目标检测系统，还是把检测当做分类任务来做，
使用滑动窗口在不同缩放尺度的图片上进行分类。
缺点很明显，计算量巨大。

像R-CNN使用提议框（region proposal）的方法，
- 第一步，先产生可能包含物体的框（potential/proposed bounding boxes，潜在/提议框）；
- 第二步，对潜在框进行图像分类，判断是否含有物体；
- 第三步，再使用分类模型确定物体的具体类别，并对bounding box进行微调；
- 第四步，根据预测分数，去除重叠程度较大的冗余框。

整个过程步骤很复杂（2阶段模型），所以速度会慢，并且难以端到端的优化。

### 2.3.因此我们提出YOLO
We reframe object detection as a single regression problem, 
straight from image pixels to bounding box coordinates and class probabilities. 
Using our system, you only look once (YOLO) at an image to 
predict what objects arepresent and where they are.

把目标检测视为一个单阶段回归问题，
直接根据图像像素预测物体框和物体类别。
使用我们的系统，你只需要看一眼，就可以知道图像中的物体和框体。

## 3.YOLO简介
![img.png](img.png)

### 3.1.YOLO的优势1：速度极快
因为它架构简单，效率肯定很高。
标准模型可以达到45fps，轻量模型可以达到150fps

### 3.2.YOLO的优势2：不仅仅使用局部图像，会考虑整张图像
不像基于滑动窗口或基于提议框的算法，
YOLO在训练和预测的时候，是基于整张图像的，
因此可以更好的结合上下文来做预测，这显然会提升算法的智能。

### 3.3.YOLO的优势3：能学到物体更好的表征
前两点比较好理解。
这一点主要是通过实验结论得出的。
作者对比了DPM、R-CNN，发现YOLO可以在艺术品图像检测上，领先它们一大截。
至于，为什么YOLO会有这种能力可能和很多因素有关，不太容易解释出来，但事实如此。

### 3.4.YOLO的劣势1：检测小物体精度低
在检测小物体的时候，YOLO的效果是比不上当时最先进的目标检测算法的，
YOLO主打的是速度快。


## 4.YOLO的原理




