<strong><p align="center"><font size=5>ComfyUI-BiRefNet-Hugo</font></p>
</strong>
<p align="center">
    <br> <font size=4>中文 | <a href="README_EN.md">English</a></font>
</p>

## 介绍

本仓库将BiRefNet最新模型封装为ComfyUI节点来使用，相较于旧模型，最新模型的抠图精度更高更好。<br>

## 安装 

#### 方法1:

1. 进入节点目录, `ComfyUI/custom_nodes/`
2. `git clone https://github.com/MoonHugo/ComfyUI-BiRefNet-Hugo.git`
3. `cd ComfyUI-BiRefNet-Hugo`
4. `pip install -r requirements.txt`
5. 重启ComfyUI

#### 方法2:
直接下载节点源码包，然后解压到custom_nodes目录下，最后重启ComfyUI

#### 方法3：
通过ComfyUI-Manager安装，搜索“ComfyUI-BiRefNet-Hugo”进行安装

## 使用

示例工作流放置在`ComfyUI-BiRefNet-Hugo/workflow`中<br/>

加载模型支持两种方式，一种是自动下载远程模型并加载模型，另外一种是加载本地模型。加载本地模型的时候需要把load_local_model设置为true，并把local_model_path设置为本地模型所在路径，例如：H:\ZhengPeng7\BiRefNet<br/>

![](./assets/9e6bf0f9-67a7-41ea-bc4b-d8352e4fac4a.png)
___

![](./assets/e21c32bf-ab98-444a-8055-54975ac47da3.png)


模型下载地址：https://huggingface.co/ZhengPeng7/BiRefNet/tree/main<br/>

___
工作流workflow.json的使用<br/>

![plot](./assets/d0a22b2a-ceb3-4205-9b4e-f6a68e4337c7.png)

___
工作流video_workflow.json的使用<br/>

![plot](./assets/2de5b085-1125-46f9-8ef3-06706743f182.png)

## 效果演示

![](./assets/demo1.gif)

![](./assets/demo2.gif)

![](./assets/demo3.gif)

![](./assets/demo4.gif)

## 社交账号
- Bilibili：[我的B站主页](https://space.bilibili.com/1303099255)

## 感谢

感谢BiRefNet仓库的所有作者 [ZhengPeng7/BiRefNet](https://github.com/zhengpeng7/birefnet)

部分代码参考了 [ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO) 感谢！

## 关注历史

[![Star History Chart](https://api.star-history.com/svg?repos=MoonHugo/ComfyUI-BiRefNet-Hugo&type=Date)](https://star-history.com/#MoonHugo/ComfyUI-BiRefNet-Hugo&Date)