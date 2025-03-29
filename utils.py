import numpy as np
import torch
from PIL import Image
from typing import List, Union

def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """
    Convert PIL image(s) to tensor, matching ComfyUI's implementation.
    
    Args:
        image: Single PIL Image or list of PIL Images
        
    Returns:
        torch.Tensor: Image tensor with values normalized to [0, 1]
    """
    if isinstance(image, list):
        if len(image) == 0:
            return torch.empty(0)
        
        tensors = []
        for img in image:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_array)[None,]
            tensors.append(tensor)
        
        if len(tensors) == 1:
            return tensors[0]
        else:
            shapes = [t.shape[1:3] for t in tensors]
            if all(shape == shapes[0] for shape in shapes):
                return torch.cat(tensors, dim=0)
            else:
                max_h = max(t.shape[1] for t in tensors)
                max_w = max(t.shape[2] for t in tensors)
                
                padded_tensors = []
                for t in tensors:
                    h, w = t.shape[1:3]
                    if h == max_h and w == max_w:
                        padded_tensors.append(t)
                    else:
                        padded = torch.zeros((1, max_h, max_w, 3), dtype=t.dtype)
                        padded[0, :h, :w, :] = t[0, :h, :w, :]
                        padded_tensors.append(padded)
                
                return torch.cat(padded_tensors, dim=0)

    # Convert PIL image to RGB if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Return tensor with shape [1, H, W, 3]
    return torch.from_numpy(img_array)[None,]


def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    """
    Convert tensor to PIL image(s), matching ComfyUI's implementation.
    
    Args:
        image: Tensor with shape [B, H, W, 3] or [H, W, 3], values in range [0, 1]
        
    Returns:
        List[Image.Image]: List of PIL Images
    """
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    # Convert tensor to numpy array, scale to [0, 255], and clip values
    numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    
    # Convert numpy array to PIL Image
    return [Image.fromarray(numpy_image)]