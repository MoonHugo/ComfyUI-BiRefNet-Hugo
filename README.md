# ComfyUI-BiRefNet-Hugo

## 介绍 | Introduction

本仓库将BiRefNet最新模型封装为ComfyUI节点来使用，相较于旧模型，最新模型的抠图精度更高更好。<br>
This repository wraps the latest BiRefNet model as ComfyUI nodes. Compared to the previous model, the latest model offers higher and better matting accuracy.

## 安装 | Installation 

1. 进入节点目录, `ComfyUI/custom_nodes/`
2. `git clone https://github.com/MoonHugo/ComfyUI-BiRefNet-Hugo.git`
3. `cd ComfyUI-BiRefNet-Hugo`
4. `pip install -r requirements.txt`
___

1. Go to comfyUI custom_nodes folder, `ComfyUI/custom_nodes/`
2. `git clone https://github.com/MoonHugo/ComfyUI-BiRefNet-Hugo.git`
3. `cd ComfyUI-BiRefNet-Hugo`
4. `pip install -r requirements.txt`

## 使用 | Usage

示例工作流放置在`ComfyUI-BiRefNet-Hugo/workflow`中<br/>
The demo workflow placed in `ComfyUI-BiRefNet-Hugo/workflow`
___
工作流workflow.json的使用<br/>
The use of workflow.json

![plot](./assets/d0a22b2a-ceb3-4205-9b4e-f6a68e4337c7.png)

工作流video_workflow.json的使用<br/>
The use of video_workflow.json
___
![plot](./assets/2de5b085-1125-46f9-8ef3-06706743f182.png)

## 效果演示 | Sample Result

![](./assets/demo1.gif)

![](./assets/demo2.gif)

![](./assets/demo3.gif)

## 社交账号 | Social Account Homepage
- Bilibili：[我的B站主页](https://space.bilibili.com/1303099255)

## 感谢 | Acknowledgments

感谢BiRefNet仓库的所有作者 [ZhengPeng7/BiRefNet](https://github.com/zhengpeng7/birefnet)

Thanks to BiRefNet repo owner [ZhengPeng7/BiRefNet](https://github.com/zhengpeng7/birefnet)
```
library_name: birefnet
tags:
  - background-removal
  - mask-generation
  - Dichotomous Image Segmentation:二分图像分割(DIS)，指分割成前景与背景，二个集合，分割出高精度效果。
  - Camouflaged Object Detection:伪装物体检测(COD)，偏工程向，旨在识别“无缝”嵌入到周围环境中的物体，例如野生动物保护、军事侦察或者工业自动化。
  - Salient Object Detection:显著性目标检测(SOD)，自动检测图像中最具视觉吸引力的部分。
  - pytorch_model_hub_mixin
  - model_hub_mixin
repo_url: https://github.com/ZhengPeng7/BiRefNet
pipeline_tag: image-segmentation
license: mit
```
部分代码参考了 [ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO) 感谢！

Some of the code references [ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO) Thanks!

## 关注历史 | star history

[![Star History Chart](https://api.star-history.com/svg?repos=MoonHugo/ComfyUI-BiRefNet-Hugo&type=Date)](https://star-history.com/#MoonHugo/ComfyUI-BiRefNet-Hugo&Date)