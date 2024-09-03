from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
from .BiRefNet_node_config import Config
import folder_paths
import os
import comfy.model_management as mm
from huggingface_hub import snapshot_download

comfyui_models_dir = folder_paths.models_dir

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image

def get_device_by_name(device):
    """
    "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto"}), 
    """
    if device == 'auto':
        try:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
                # device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = "mps"
                # device = torch.device("mps")
            elif torch.xpu.is_available():
                device = "xpu"
                # device = torch.device("xpu")
            # device = mm.get_torch_device()
        except:
                raise AttributeError("What's your device(到底用什么设备跑的)？")
    # elif device == 'cuda':
    #     device = torch.device("cuda")
    # elif device == "mps":
    #     device = torch.device("mps")
    # elif device == "xpu":
    #     device = torch.device("xpu")
    print("\033[93mUse Device(使用设备):", device, "\033[0m")
    return device

def get_dtype_by_name(dtype):
    """
    "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto"}),
    """
    if dtype == 'auto':
        try:
            if mm.should_use_fp16():
                dtype = torch.float16
            elif mm.should_use_bf16():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtypes manually.")
    elif dtype== "fp16":
         dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype = torch.float32
    elif dtype == "fp8_e4m3fn":
        dtype = torch.float8_e4m3fn
    elif dtype == "fp8_e4m3fnuz":
        dtype = torch.float8_e4m3fnuz
    elif dtype == "fp8_e5m2":
        dtype = torch.float8_e5m2
    elif dtype == "fp8_e5m2fnuz":
        dtype = torch.float8_e5m2fnuz
    print("\033[93mModel Precision(模型精度):", dtype, "\033[0m")
    return dtype


class BiRefNet_Hugo:
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        rembg_list = os.listdir(os.path.join(comfyui_models_dir, "rembg"))
        rembg_list.insert(0, "Auto_DownLoad-ZhengPeng7/BiRefNet")
        rembg_list.insert(1, "Auto_DownLoad-ZhengPeng7/BiRefNet-DIS5K-TR_TEs")
        rembg_list.insert(2, "Auto_DownLoad-ZhengPeng7/BiRefNet-COD")
        return {
            "required": {
                "model": (rembg_list, ),
                # "model": (["Auto_Download"] + os.listdir(os.path.join(comfyui_models_dir, "rembg"))), 
                "image": ("IMAGE",),
                "background_color_name": (["transparency", "green", "white", "red", "yellow", "blue", "black"],{"default": "transparency"}), 
                "background_color_code": ("STRING",{"default": "#00FF00"}), 
                "background_color_mode": ("BOOLEAN", {"default": True, "label_on": "color_name", "label_off": "color_code"}), 
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto"}), 
                "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"fp32"}),
                "cpu_offload": ("BOOLEAN", {"default": False, "label_on": "model_to_cpu", "label_off": "unload_model"}),
                "Auto_Download_Path": ("BOOLEAN", {"default": False, "label_on": "rembg_local本地", "label_off": ".cache缓存"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "background_remove"
    CATEGORY = "🔥BiRefNet"
  
    def background_remove(self, 
                          image, 
                          model, 
                          device, 
                          dtype, 
                          cpu_offload,
                          background_color_name, 
                          background_color_code,
                          background_color_mode,
                          Auto_Download_Path, 
                          ):
        Config()
        torch.set_float32_matmul_precision(["high", "highest"][0])
        processed_images = []
        processed_masks = []
        model_name = model.replace("Auto_DownLoad-", "")
        if 'Auto_DownLoad-' not in model:
            model_path = os.path.join(comfyui_models_dir, "rembg", model)
        elif ('Auto_DownLoad-' in model) and (Auto_Download_Path == True):
            model_path = os.path.join(comfyui_models_dir, "rembg", ("models--" + str(model_name).replace("/", "--")))
            if not os.path.exists(os.path.join(model_path, "model.safetensors")):
                snapshot_download(model_name, 
                                local_dir=model_path, 
                                local_dir_use_symlinks=False
                                )
        elif ('Auto_DownLoad-' in model) and (Auto_Download_Path == False):
            model_path = model_name
            
        device = get_device_by_name(device)
        dtype = get_dtype_by_name(dtype)
            
        if self.model == None:
            self. model = AutoModelForImageSegmentation.from_pretrained(
                model_path, 
                trust_remote_code=True,
            ).to(device, dtype)
        else:
            self.model.to(device)
        self.model.eval()
        for image in image:
            orig_image = tensor2pil(image)
            w,h = orig_image.size
            image = resize_image(orig_image)
            im_tensor = transform_image(image).unsqueeze(0)
            im_tensor=im_tensor.to(device, dtype)
            with torch.no_grad():
                result = self.model(im_tensor)[-1].sigmoid().cpu()
            result = torch.squeeze(F.interpolate(result, size=(h,w)))
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)    
            im_array = (result*255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            if background_color_name == 'transparency' and background_color_mode == True:
                color = (0,0,0,0)
                mode = "RGBA"
            else:
                color = background_color_name
                mode = "RGB"
                if not background_color_mode:
                    color = "#" + background_color_code # 颜色码-绿色：#00FF00
            # new_im = Image.new("RGBA", pil_im.size, (0,0,0,0))
            new_im = Image.new(mode, pil_im.size, color)
            new_im.paste(orig_image, mask=pil_im)
            new_im_tensor = pil2tensor(new_im)
            pil_im_tensor = pil2tensor(pil_im)
            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)
        if cpu_offload == True:
            self.model.to("cpu")
        else:
            del self.model
            self.model = None

        return new_ims, new_masks


NODE_CLASS_MAPPINGS = {
    "BiRefNet_Hugo": BiRefNet_Hugo
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNet_Hugo": "🔥BiRefNet"
}
