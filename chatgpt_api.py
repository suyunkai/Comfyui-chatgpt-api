import mimetypes
import os
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
import folder_paths
from .utils import pil2tensor, tensor2pil


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


class ComfyuiChatGPTApi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gpt-4o-image-vip", "multiline": False}),             
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "files": ("FILES",), 
                "image_url": ("STRING", {"multiline": False, "default": ""}),
                "images": ("IMAGE", {"default": None}),  
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 16384, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "image_download_timeout": ("INT", {"default": 100, "min": 5, "max": 300, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "response", "image_urls", "prompt")
    FUNCTION = "process"
    CATEGORY = "ainewsto/Chatgpt"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        self.image_download_timeout = 100
        self.api_endpoint = "https://ai.comfly.chat/v1/chat/completions"

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
        import re
        # Extract URLs from markdown image format: ![description](url)
        image_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(image_pattern, response_text)
        
        # If no markdown format found, extract raw URLs that look like image links
        if not matches:
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
            matches = re.findall(url_pattern, response_text)
            
        # Extract all URLs that look like links
        if not matches:
            all_urls_pattern = r'https?://\S+'
            matches = re.findall(all_urls_pattern, response_text)
            
        return matches if matches else []

    def download_image(self, url, timeout=30):
        """Download image from URL and convert to tensor with improved error handling"""
        try:
            # Custom headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://comfyui.com/'
            }
            
            # Try to download with custom headers
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            # Check content type to verify it's an image
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                print(f"Warning: URL doesn't point to an image. Content-Type: {content_type}")
                # Still try to open it anyway
            
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

    def process(self, prompt, model, files=None, image_url="", images=None, temperature=0.7, 
               max_tokens=4096, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, seed=-1,
               image_download_timeout=100, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
        try:
            # Set image download timeout from parameter
            self.image_download_timeout = image_download_timeout
            
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                # Return a blank image if no input image
                blank_img = Image.new('RGB', (512, 512), color='white')
                return (pil2tensor(blank_img), error_message, "", "") 
            
            # Initialize progress bar
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            # Get current timestamp for formatting
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Handle seed
            if seed < 0:
                seed = random.randint(0, 2147483647)
                print(f"Using random seed: {seed}")
            
            # Prepare message content
            content = []
            
            # Add prompt text
            content.append({"type": "text", "text": prompt})
            
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
            
            # Add image URL if provided and no images
            elif image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            # Add files if provided
            if files:
                file_paths = files if isinstance(files, list) else [files]
                for file_path in file_paths:
                    encoded_content, mime_type = self.file_to_base64(file_path)
                    if encoded_content and mime_type:
                        # Add file as attachment
                        if mime_type.startswith('image/'):
                            # If it's an image, use image_url type
                            content.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:{mime_type};base64,{encoded_content}"}
                            })
                        else:
                            # For non-image files
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
            
            # Create messages
            messages = [{
                "role": "user",
                "content": content
            }]
            
            # Create API payload
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "seed": seed,
                "stream": True  # Use streaming for better user experience
            }
            
            # Use asyncio for streaming response
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_text = loop.run_until_complete(self.stream_response(payload, pbar))
            loop.close()
            
            # Format the response
            formatted_response = f"**User prompt**: {prompt}\n\n**Response** ({timestamp}):\n{response_text}"
            
            # Check if response contains image URLs
            image_urls = self.extract_image_urls(response_text)
            image_urls_string = "\n".join(image_urls) if image_urls else ""
            
            if image_urls:
                try:
                    # Download and convert all found images
                    img_tensors = []
                    successful_downloads = 0
                    
                    for i, url in enumerate(image_urls):
                        print(f"Attempting to download image {i+1}/{len(image_urls)} from: {url}")
                        
                        # Update progress bar for each image
                        pbar.update_absolute(min(80, 40 + (i+1) * 40 // len(image_urls)))
                        
                        img_tensor = self.download_image(url, self.image_download_timeout)
                        if img_tensor is not None:
                            img_tensors.append(img_tensor)
                            successful_downloads += 1
                    
                    print(f"Successfully downloaded {successful_downloads} out of {len(image_urls)} images")
                    
                    if img_tensors:
                        # Combine all images into a single tensor batch
                        combined_tensor = torch.cat(img_tensors, dim=0)
                        pbar.update_absolute(100)
                        return (combined_tensor, formatted_response, image_urls_string)
                except Exception as e:
                    print(f"Error processing image URLs: {str(e)}")
            
            # Return appropriate response if no image was found or downloads failed
            if images is not None:
                # Return input images with text response
                pbar.update_absolute(100)
                return (images, formatted_response, image_urls_string, prompt)  
            else:
                # Create a default blank image
                blank_img = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor(blank_img)
                pbar.update_absolute(100)
                return (blank_tensor, formatted_response, image_urls_string, prompt)  
            
        except Exception as e:
            error_message = f"Error calling ChatGPT API: {str(e)}"
            print(error_message)
            
            # Return error with original image or blank image
            if images is not None:
                return (images, error_message, "", prompt)  
            else:
                blank_img = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor(blank_img)
                return (blank_tensor, error_message, "", prompt) 

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
                    
                    # Process streaming response
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
                                        
                                        # Update progress (approximate progress)
                                        pbar.update_absolute(min(40, 20 + len(full_response) // 100))
                            except json.JSONDecodeError:
                                continue
            
            return full_response
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"API request timed out after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Error in streaming response: {str(e)}")


WEB_DIRECTORY = "./web"    
        
NODE_CLASS_MAPPINGS = {
    "ComfyuiChatGPTApi": ComfyuiChatGPTApi,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiChatGPTApi": "ComfyuiChatGPTApi",
}
