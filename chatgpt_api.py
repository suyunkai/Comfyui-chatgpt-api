import mimetypes
import os
import io
import math
import random
import torch
import requests
import time
from PIL import Image
from io import BytesIO
import json
import comfy.utils
import re
import aiohttp
import asyncio
import base64
import uuid
import numpy as np
import folder_paths
from .utils import pil2tensor, tensor2pil
from comfy.utils import common_upscale


def get_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


# reference: OpenAIGPTImage1 node from comfyui node
def downscale_input(image):
    samples = image.movedim(-1,1)

    total = int(1536 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1,-1)
    return s

class Comfyui_gpt_image_1_edit:

    _last_edited_image = None
    _conversation_history = []
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "mask": ("MASK",),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gpt-image-1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "clear_chats": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("edited_image", "response", "chats")
    FUNCTION = "edit_image"
    CATEGORY = "ainewsto/Chatgpt"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def format_conversation_history(self):
        """Format the conversation history for display"""
        if not Comfyui_gpt_image_1_edit._conversation_history:
            return ""
        formatted_history = ""
        for entry in Comfyui_gpt_image_1_edit._conversation_history:
            formatted_history += f"**User**: {entry['user']}\n\n"
            formatted_history += f"**AI**: {entry['ai']}\n\n"
            formatted_history += "---\n\n"
        return formatted_history.strip()
    
    def edit_image(self, image, prompt, model="gpt-image-1", n=1, quality="auto", 
              seed=0, mask=None, api_key="", size="auto", clear_chats=True):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
 
        original_image = image
        original_batch_size = image.shape[0]
        use_saved_image = False

        if not clear_chats and Comfyui_gpt_image_1_edit._last_edited_image is not None:
            if original_batch_size > 1:
                last_batch_size = Comfyui_gpt_image_1_edit._last_edited_image.shape[0]
                last_image_first = Comfyui_gpt_image_1_edit._last_edited_image[0:1]
                if last_image_first.shape[1:] == original_image.shape[1:]:
                    image = torch.cat([last_image_first, original_image[1:]], dim=0)
                    use_saved_image = True
            else:

                image = Comfyui_gpt_image_1_edit._last_edited_image
                use_saved_image = True

        if clear_chats:
            Comfyui_gpt_image_1_edit._conversation_history = []

            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())
          
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            files = {}
 
            if image is not None:
                batch_size = image.shape[0]
                for i in range(batch_size):
                    single_image = image[i:i+1]
                    scaled_image = downscale_input(single_image).squeeze()
                    
                    image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(image_np)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    if batch_size == 1:
                        files['image'] = ('image.png', img_byte_arr, 'image/png')
                    else:
                        if 'image[]' not in files:
                            files['image[]'] = []
                        files['image[]'].append(('image_{}.png'.format(i), img_byte_arr, 'image/png'))
            
            if mask is not None:
                if image.shape[0] != 1:
                    raise Exception("Cannot use a mask with multiple images")
                if image is None:
                    raise Exception("Cannot use a mask without an input image")
                if mask.shape[1:] != image.shape[1:-1]:
                    raise Exception("Mask and Image must be the same size")
                
                batch, height, width = mask.shape
                rgba_mask = torch.zeros(height, width, 4, device="cpu")
                rgba_mask[:,:,3] = (1-mask.squeeze().cpu())
                scaled_mask = downscale_input(rgba_mask.unsqueeze(0)).squeeze()
                mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)
                mask_byte_arr = io.BytesIO()
                mask_img.save(mask_byte_arr, format='PNG')
                mask_byte_arr.seek(0)
                files['mask'] = ('mask.png', mask_byte_arr, 'image/png')

            if 'image[]' in files:

                data = {
                    'prompt': prompt,
                    'model': model,
                    'n': str(n),
                    'quality': quality
                }
                
                if size != "auto":
                    data['size'] = size

                image_files = []
                for file_tuple in files['image[]']:
                    image_files.append(('image', file_tuple))

                if 'mask' in files:
                    image_files.append(('mask', files['mask']))

                response = requests.post(
                    "https://ai.comfly.chat/v1/images/edits",
                    headers=self.get_headers(),
                    data=data,
                    files=image_files,
                    timeout=self.timeout
                )
            else:
                data = {
                    'prompt': prompt,
                    'model': model,
                    'n': str(n),
                    'quality': quality
                }
                
                if size != "auto":
                    data['size'] = size

                request_files = []

                if 'image' in files:
                    request_files.append(('image', files['image']))

                if 'mask' in files:
                    request_files.append(('mask', files['mask']))

                response = requests.post(
                    "https://ai.comfly.chat/v1/images/edits",
                    headers=self.get_headers(),
                    data=data,
                    files=request_files,
                    timeout=self.timeout
                )

            pbar.update_absolute(50)

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())
            result = response.json()
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())

            edited_images = []
            image_urls = []

            for item in result["data"]:
                if "b64_json" in item:
                    image_data = base64.b64decode(item["b64_json"])
                    edited_image = Image.open(BytesIO(image_data))
                    edited_tensor = pil2tensor(edited_image)
                    edited_images.append(edited_tensor)
                elif "url" in item:
                    image_urls.append(item["url"])
                    try:
                        img_response = requests.get(item["url"])
                        if img_response.status_code == 200:
                            edited_image = Image.open(BytesIO(img_response.content))
                            edited_tensor = pil2tensor(edited_image)
                            edited_images.append(edited_tensor)
                    except Exception as e:
                        print(f"Error downloading image from URL: {str(e)}")

            pbar.update_absolute(90)

            if edited_images:
                combined_tensor = torch.cat(edited_images, dim=0)
                response_info = f"Successfully edited {len(edited_images)} image(s)\n"
                response_info += f"Prompt: {prompt}\n"
                response_info += f"Model: {model}\n"
                response_info += f"Quality: {quality}\n"
                
                if use_saved_image:
                    response_info += "[Using previous edited image as input]\n"
                    
                if size != "auto":
                    response_info += f"Size: {size}\n"

                Comfyui_gpt_image_1_edit._conversation_history.append({
                    "user": f"Edit image with prompt: {prompt}",
                    "ai": f"Generated edited image with {model}"
                })
 
                Comfyui_gpt_image_1_edit._last_edited_image = combined_tensor
                
                pbar.update_absolute(100)
                return (combined_tensor, response_info, self.format_conversation_history())
            else:
                error_message = "No edited images in response"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())
            
        except Exception as e:
            error_message = f"Error in image editing: {str(e)}"
            import traceback
            print(traceback.format_exc())  
            print(error_message)
            return (original_image, error_message, self.format_conversation_history())

        

class Comfyui_gpt_image_1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gpt-image-1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "ainewsto/Chatgpt"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_image(self, prompt, model="gpt-image-1", n=1, quality="auto", 
                size="auto", background="auto", output_format="png", 
                moderation="auto", seed=0, api_key=""):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message)
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            payload = {
                "prompt": prompt,
                "model": model,
                "n": n,
                "quality": quality,
                "background": background,
                "output_format": output_format,
                "moderation": moderation,
            }
            
            # Only include size if it's not "auto"
            if size != "auto":
                payload["size"] = size
            
            response = requests.post(
                "https://ai.comfly.chat/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(50)
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message)
                
            # Parse the response
            result = response.json()
            
            # Format the response information
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response_info = f"**GPT-image-1 Generation ({timestamp})**\n\n"
            response_info += f"Prompt: {prompt}\n"
            response_info += f"Model: {model}\n"
            response_info += f"Quality: {quality}\n"
            if size != "auto":
                response_info += f"Size: {size}\n"
            response_info += f"Background: {background}\n"
            response_info += f"Seed: {seed} (Note: Seed not used by API)\n\n"
            
            # Process the generated images
            generated_images = []
            image_urls = []
            
            if "data" in result and result["data"]:
                for i, item in enumerate(result["data"]):
                    pbar.update_absolute(50 + (i+1) * 50 // len(result["data"]))
                    
                    if "b64_json" in item:
                        # Decode base64 image
                        image_data = base64.b64decode(item["b64_json"])
                        generated_image = Image.open(BytesIO(image_data))
                        generated_tensor = pil2tensor(generated_image)
                        generated_images.append(generated_tensor)
                    elif "url" in item:
                        image_urls.append(item["url"])
                        # Download and process the image from URL
                        try:
                            img_response = requests.get(item["url"])
                            if img_response.status_code == 200:
                                generated_image = Image.open(BytesIO(img_response.content))
                                generated_tensor = pil2tensor(generated_image)
                                generated_images.append(generated_tensor)
                        except Exception as e:
                            print(f"Error downloading image from URL: {str(e)}")
            else:
                error_message = "No generated images in response"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, response_info)
                
            # Add usage information to the response if available
            if "usage" in result:
                response_info += "Usage Information:\n"
                if "total_tokens" in result["usage"]:
                    response_info += f"Total Tokens: {result['usage']['total_tokens']}\n"
                if "input_tokens" in result["usage"]:
                    response_info += f"Input Tokens: {result['usage']['input_tokens']}\n"
                if "output_tokens" in result["usage"]:
                    response_info += f"Output Tokens: {result['usage']['output_tokens']}\n"
                
                # Add detailed token usage if available
                if "input_tokens_details" in result["usage"]:
                    response_info += "Input Token Details:\n"
                    details = result["usage"]["input_tokens_details"]
                    if "text_tokens" in details:
                        response_info += f"  Text Tokens: {details['text_tokens']}\n"
                    if "image_tokens" in details:
                        response_info += f"  Image Tokens: {details['image_tokens']}\n"
            
            if generated_images:
                # Combine all generated images into a single tensor
                combined_tensor = torch.cat(generated_images, dim=0)
                
                pbar.update_absolute(100)
                return (combined_tensor, response_info)
            else:
                error_message = "No images were successfully processed"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, response_info)
                
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message)


class ComfyuiChatGPTApi:

    _last_generated_image_urls = ""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gpt-image-1", "multiline": False}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "files": ("FILES",), 
                "image_url": ("STRING", {"multiline": False, "default": ""}),
                "images": ("IMAGE", {"default": None}),  
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 16384, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": -2.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "image_download_timeout": ("INT", {"default": 600, "min": 300, "max": 1200, "step": 10}),
                "clear_chats": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "response", "image_urls", "chats")
    FUNCTION = "process"
    CATEGORY = "ainewsto/Chatgpt"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 800
        self.image_download_timeout = 600
        self.api_endpoint = "https://ai.comfly.chat/v1/chat/completions"
        self.conversation_history = []

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def file_to_base64(self, file_path):
        """Convert file to base64 string and return appropriate MIME type"""
        try:
            with open(file_path, "rb") as file:
                file_content = file.read()
                encoded_content = base64.b64encode(file_content).decode('utf-8')
                # Get MIME type
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    # Default to binary if MIME type can't be determined
                    mime_type = "application/octet-stream"
                return encoded_content, mime_type
        except Exception as e:
            print(f"Error encoding file: {str(e)}")
            return None, None

    def extract_image_urls(self, response_text):
        """Extract image URLs from markdown format in response"""
      
        image_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(image_pattern, response_text)
      
        if not matches:
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
            matches = re.findall(url_pattern, response_text)
        
        if not matches:
            all_urls_pattern = r'https?://\S+'
            matches = re.findall(all_urls_pattern, response_text)
        return matches if matches else []

    def download_image(self, url, timeout=30):
        """Download image from URL and convert to tensor with improved error handling"""
        try:
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://comfyui.com/'
            }
           
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
          
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                print(f"Warning: URL doesn't point to an image. Content-Type: {content_type}")
               
            image = Image.open(BytesIO(response.content))
            return pil2tensor(image)
        except requests.exceptions.Timeout:
            print(f"Timeout error downloading image from {url} (timeout: {timeout}s)")
            return None
        except requests.exceptions.SSLError as e:
            print(f"SSL Error downloading image from {url}: {str(e)}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"Connection error downloading image from {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error downloading image from {url}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error downloading image from {url}: {str(e)}")
            return None

    def format_conversation_history(self):
        """Format the conversation history for display"""
        if not self.conversation_history:
            return ""
        formatted_history = ""
        for entry in self.conversation_history:
            formatted_history += f"**User**: {entry['user']}\n\n"
            formatted_history += f"**AI**: {entry['ai']}\n\n"
            formatted_history += "---\n\n"
        return formatted_history.strip()

    def process(self, prompt, model, clear_chats=True, files=None, image_url="", images=None, temperature=0.7, 
               max_tokens=4096, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, seed=-1,
               image_download_timeout=100, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
        try:
          
            self.image_download_timeout = image_download_timeout
          
            if clear_chats:
                self.conversation_history = []
                
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
               
                blank_img = Image.new('RGB', (512, 512), color='white')
                return (pil2tensor(blank_img), error_message, "", self.format_conversation_history()) 
            
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
           
            if seed < 0:
                seed = random.randint(0, 2147483647)
                print(f"Using random seed: {seed}")
           
            content = []
            
            content.append({"type": "text", "text": prompt})
            
           
            if not clear_chats and ComfyuiChatGPTApi._last_generated_image_urls:
                prev_image_url = ComfyuiChatGPTApi._last_generated_image_urls.split('\n')[0].strip()
                if prev_image_url:
                    print(f"Using previous image URL: {prev_image_url}")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": prev_image_url}
                    })
            
            elif clear_chats:
                if images is not None:
                    batch_size = images.shape[0]
                    max_images = min(batch_size, 4)  
                    for i in range(max_images):
                        pil_image = tensor2pil(images)[i]
                        image_base64 = self.image_to_base64(pil_image)
                        content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        })
                    if batch_size > max_images:
                        content.append({
                            "type": "text",
                            "text": f"\n(Note: {batch_size-max_images} additional images were omitted due to API limitations)"
                        })
                
                elif image_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
        
            elif image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            if files:
                file_paths = files if isinstance(files, list) else [files]
                for file_path in file_paths:
                    encoded_content, mime_type = self.file_to_base64(file_path)
                    if encoded_content and mime_type:
                       
                        if mime_type.startswith('image/'):
                           
                            content.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:{mime_type};base64,{encoded_content}"}
                            })
                        else:
                            
                            content.append({
                                "type": "text", 
                                "text": f"\n\nI've attached a file ({os.path.basename(file_path)}) for analysis."
                            })
                            content.append({
                                "type": "file_url",
                                "file_url": {
                                    "url": f"data:{mime_type};base64,{encoded_content}",
                                    "name": os.path.basename(file_path)
                                }
                            })
        
            messages = []
        
            messages.append({
                "role": "user",
                "content": content
            })
        
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "seed": seed,
                "stream": True  
            }
        
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_text = loop.run_until_complete(self.stream_response(payload, pbar))
            loop.close()
        
            self.conversation_history.append({
                "user": prompt,
                "ai": response_text
            })
        
            technical_response = f"**Model**: {model}\n**Temperature**: {temperature}\n**Seed**: {seed}\n**Time**: {timestamp}"
        
            image_urls = self.extract_image_urls(response_text)
            image_urls_string = "\n".join(image_urls) if image_urls else ""
            
            if image_urls:
                ComfyuiChatGPTApi._last_generated_image_urls = image_urls_string
     
            chat_history = self.format_conversation_history()
            if image_urls:
                try:
                    
                    img_tensors = []
                    successful_downloads = 0
                    for i, url in enumerate(image_urls):
                        print(f"Attempting to download image {i+1}/{len(image_urls)} from: {url}")
                
                        pbar.update_absolute(min(80, 40 + (i+1) * 40 // len(image_urls)))
                        img_tensor = self.download_image(url, self.image_download_timeout)
                        if img_tensor is not None:
                            img_tensors.append(img_tensor)
                            successful_downloads += 1
                    print(f"Successfully downloaded {successful_downloads} out of {len(image_urls)} images")
                    if img_tensors:
                
                        combined_tensor = torch.cat(img_tensors, dim=0)
                        pbar.update_absolute(100)
                        return (combined_tensor, technical_response, image_urls_string, chat_history)
                except Exception as e:
                    print(f"Error processing image URLs: {str(e)}")
        
            if images is not None:
                pbar.update_absolute(100)
                return (images, technical_response, image_urls_string, chat_history)  
            else:
                blank_img = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor(blank_img)
                pbar.update_absolute(100)
                return (blank_tensor, technical_response, image_urls_string, chat_history)  
                
        except Exception as e:
            error_message = f"Error calling ChatGPT API: {str(e)}"
            print(error_message)
        
            if images is not None:
                return (images, error_message, "", self.format_conversation_history())  
            else:
                blank_img = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor(blank_img)
                return (blank_tensor, error_message, "", self.format_conversation_history()) 

    async def stream_response(self, payload, pbar):
        """Stream response from API"""
        full_response = ""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_endpoint, 
                    headers=self.get_headers(), 
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API Error {response.status}: {error_text}")
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]  
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_response += content
                                       
                                        pbar.update_absolute(min(40, 20 + len(full_response) // 100))
                            except json.JSONDecodeError:
                                continue
            return full_response
        
        except asyncio.TimeoutError:
            raise TimeoutError(f"API request timed out after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Error in streaming response: {str(e)}")

        
NODE_CLASS_MAPPINGS = {
    "ComfyuiChatGPTApi": ComfyuiChatGPTApi,
    "Comfyui_gpt_image_1_edit": Comfyui_gpt_image_1_edit,
    "Comfyui_gpt_image_1": Comfyui_gpt_image_1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiChatGPTApi": "ComfyuiChatGPTApi",
    "Comfyui_gpt_image_1_edit": "Comfyui_gpt_image_1_edit",
    "Comfyui_gpt_image_1": "Comfyui_gpt_image_1",
}
