from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import comfy.model_management as mm
import os

torch.set_float32_matmul_precision(["high", "highest"][0])

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

current_path  = os.getcwd()

## ComfyUI portable standalone build for Windows 
model_path = os.path.join(current_path, "ComfyUI"+os.sep+"models"+os.sep+"BiRefNet")

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image

colors = ["transparency", "green", "white", "red", "yellow", "blue", "black", "pink", "purple", "brown", "violet", "wheat", "whitesmoke", "yellowgreen", "turquoise", "tomato", "thistle", "teal", "tan", "steelblue", "springgreen", "snow", "slategrey", "slateblue", "skyblue", "orange"]

def get_device_by_name(device):
    """
    "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto"}), 
    """
    if device == 'auto':
        try:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            elif torch.xpu.is_available():
                device = "xpu"
        except:
                raise AttributeError("What's your device(到底用什么设备跑的)？")
    print("\033[93mUse Device(使用设备):", device, "\033[0m")
    return device


class BiRefNet_Hugo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "required": {
                "image": ("IMAGE",),
                "load_local_model": ("BOOLEAN", {"default": False}),
                "background_color_name": (colors,{"default": "transparency"}), 
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto"})
            },
            "optional": {
                "local_model_path": ("STRING", {"default":model_path}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "background_remove"
    CATEGORY = "🔥BiRefNet"
  
    def background_remove(self, 
                          image, 
                          load_local_model,
                          device, 
                          background_color_name, 
                          *args, **kwargs
                          ):
        processed_images = []
        processed_masks = []
       
        device = get_device_by_name(device)
        
        if load_local_model:
            local_model_path = kwargs.get("local_model_path", model_path)
            birefnet = AutoModelForImageSegmentation.from_pretrained(local_model_path,trust_remote_code=True)
        else:
            birefnet = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True
            )
        
        birefnet.to(device)
        for image in image:
            orig_image = tensor2pil(image)
            w,h = orig_image.size
            image = resize_image(orig_image)
            im_tensor = transform_image(image).unsqueeze(0)
            im_tensor=im_tensor.to(device)
            with torch.no_grad():
                result = birefnet(im_tensor)[-1].sigmoid().cpu()
            result = torch.squeeze(F.interpolate(result, size=(h,w)))
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)    
            im_array = (result*255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            if background_color_name == 'transparency':
                color = (0,0,0,0)
                mode = "RGBA"
            else:
                color = background_color_name
                mode = "RGB"
            new_im = Image.new(mode, pil_im.size, color)
            new_im.paste(orig_image, mask=pil_im)
            new_im_tensor = pil2tensor(new_im)
            pil_im_tensor = pil2tensor(pil_im)
            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        return new_ims, new_masks


NODE_CLASS_MAPPINGS = {
    "BiRefNet_Hugo": BiRefNet_Hugo
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNet_Hugo": "🔥BiRefNet"
}
