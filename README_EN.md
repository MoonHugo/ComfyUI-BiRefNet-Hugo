<div style="text-align: center; font-size: 24px; font-weight: bold;">ComfyUI-BiRefNet-Hugo</div>
<p align="center">
    <br> <font size=5>English | <a href="README.md">中文</a></font>
</p>

## Introduction

This repository wraps the latest BiRefNet model as ComfyUI nodes. Compared to the previous model, the latest model offers higher and better matting accuracy.

## Installation 

#### Method  1:

1. Go to comfyUI custom_nodes folder, `ComfyUI/custom_nodes/`
2. `git clone https://github.com/MoonHugo/ComfyUI-BiRefNet-Hugo.git`
3. `cd ComfyUI-BiRefNet-Hugo`
4. `pip install -r requirements.txt`
5. restart ComfyUI

#### Method 2:
Directly download the node source package, then extract it into the custom_nodes directory, and finally restart ComfyUI.

#### Method 3：
Install through ComfyUI-Manager by searching for 'ComfyUI-BiRefNet-Hugo' and installing it.

## Usage

The demo workflow placed in `ComfyUI-BiRefNet-Hugo/workflow`

Loading the model supports two methods: one is to automatically download and load a remote model, and the other is to load a local model. When loading a local model, you need to set 'load_local_model' to true and 'local_model_path' to the path where the local model is located, for example: H:\ZhengPeng7\BiRefNet.

![](./assets/9e6bf0f9-67a7-41ea-bc4b-d8352e4fac4a.png)

___

![](./assets/e21c32bf-ab98-444a-8055-54975ac47da3.png)


Model download address: https://huggingface.co/ZhengPeng7/BiRefNet/tree/main

___
The use of workflow.json

![plot](./assets/d0a22b2a-ceb3-4205-9b4e-f6a68e4337c7.png)

___
The use of video_workflow.json

![plot](./assets/2de5b085-1125-46f9-8ef3-06706743f182.png)

## Sample Result

![](./assets/demo1.gif)

![](./assets/demo2.gif)

![](./assets/demo3.gif)

![](./assets/demo4.gif)

## Social Account Homepage
- Bilibili：[My BILIBILI Homepage](https://space.bilibili.com/1303099255)

## Acknowledgments

Thanks to BiRefNet repo owner [ZhengPeng7/BiRefNet](https://github.com/zhengpeng7/birefnet)

Some of the code references [ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO) Thanks!

## star history

[![Star History Chart](https://api.star-history.com/svg?repos=MoonHugo/ComfyUI-BiRefNet-Hugo&type=Date)](https://star-history.com/#MoonHugo/ComfyUI-BiRefNet-Hugo&Date)