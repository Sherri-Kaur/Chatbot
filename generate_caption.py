import os
import tempfile
import base64
from PIL import Image
import io
import json
from dotenv import load_dotenv

try:
    from openai import AzureOpenAI
    HAS_AZURE_OPENAI = True
except ImportError:
    HAS_AZURE_OPENAI = False
    print("Azure OpenAI package not available. Install with: pip install openai")

load_dotenv()

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_VISION_DEPLOYMENT = os.getenv("AZURE_VISION_DEPLOYMENT", "gpt-4o-mini")

# Image storage directory
IMAGE_STORAGE_PATH = os.path.join(tempfile.gettempdir(), "rag_images")

class ImageCaptionGenerator:
    def __init__(self):
        self.client = None
        # Create image storage directory
        os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)
        
        if HAS_AZURE_OPENAI and AZURE_OPENAI_API_KEY:
            try:
                self.client = AzureOpenAI(
                    api_key=AZURE_OPENAI_API_KEY,
                    api_version=AZURE_OPENAI_API_VERSION,
                    azure_endpoint=AZURE_OPENAI_ENDPOINT
                )
                print("Azure OpenAI client initialized for image analysis")
            except Exception as e:
                print(f"Failed to initialize Azure OpenAI client: {e}")
                self.client = None
    
    def save_image_file(self, image_file, file_hash):
        try:
            if hasattr(image_file, 'name'):
                original_name = image_file.name
                extension = os.path.splitext(original_name)[1] or '.png'
            else:
                original_name = os.path.basename(str(image_file))
                extension = os.path.splitext(original_name)[1] or '.png'
            
            # Create unique filename
            filename = f"{file_hash}{extension}"
            filepath = os.path.join(IMAGE_STORAGE_PATH, filename)
            
            # Save the image
            if hasattr(image_file, 'read'):
                with open(filepath, 'wb') as f:
                    f.write(image_file.getvalue())
            else:
                with open(image_file, 'rb') as src, open(filepath, 'wb') as dst:
                    dst.write(src.read())
            
            return filepath
        except Exception as e:
            print(f"Error saving image file: {e}")
            return None
    
    def image_to_base64(self, image_file):
        try:
            if hasattr(image_file, 'read'):
                # Streamlit UploadedFile object
                image_data = image_file.getvalue()
            else:
                # File path
                with open(image_file, 'rb') as f:
                    image_data = f.read()
            
            # Convert to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return base64_image
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None
    
    def analyze_image(self, image_file, max_tokens=300):
        if not self.client:
            return "Image analysis not available - Azure OpenAI client not configured"
        
        try:
            base64_image = self.image_to_base64(image_file)
            if not base64_image:
                return "Failed to process image"
            
            response = self.client.chat.completions.with_raw_response.create(
                model=AZURE_VISION_DEPLOYMENT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this image in detail and provide:
                                1. A comprehensive description of what's visible
                                2. Key objects, people, text, or elements
                                3. Context or potential meaning
                                4. Any notable colors, composition, or style
                                
                                Be thorough and descriptive as this will be used for document search."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            # Extract the response content
            completion = response.parse()
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error analyzing image with Azure OpenAI: {e}")
            return f"Image analysis failed: {str(e)}"
    
    def generate_simple_caption(self, image_file):
        try:
            if hasattr(image_file, 'read'):
                image = Image.open(io.BytesIO(image_file.getvalue()))
            else:
                image = Image.open(image_file)
            
            width, height = image.size
            format_info = image.format
            mode = image.mode
            
            caption = f"Image: {width}x{height} pixels, format: {format_info}, mode: {mode}"
            return caption
            
        except Exception as e:
            print(f"Error generating simple caption: {e}")
            return "Unable to process image"
    
    def process_image_files(self, image_files):
        from langchain_core.documents import Document
        import hashlib
        
        image_documents = []
        
        for image_file in image_files:
            try:
                # Generate filename or use existing
                if hasattr(image_file, 'name'):
                    filename = image_file.name
                else:
                    filename = os.path.basename(str(image_file))
                
                # Generate file hash for unique identification
                if hasattr(image_file, 'getvalue'):
                    file_content = image_file.getvalue()
                else:
                    with open(image_file, 'rb') as f:
                        file_content = f.read()
                
                file_hash = hashlib.md5(file_content).hexdigest()[:16]
                
                # Save image file
                image_path = self.save_image_file(image_file, file_hash)
                
                # Generate caption
                if self.client:
                    caption = self.analyze_image(image_file)
                    source_type = "Azure OpenAI Vision Analysis"
                else:
                    caption = self.generate_simple_caption(image_file)
                    source_type = "Basic Image Analysis"
                
                # Create Document object with image metadata
                doc = Document(
                    page_content=f"IMAGE CAPTION: {caption}\nFILENAME: {filename}",
                    metadata={
                        "source": filename,
                        "type": "image",
                        "analysis_method": source_type,
                        "file_hash": file_hash,
                        "content_type": "image_caption",
                        "image_path": image_path,
                        "original_filename": filename,
                        "stored_at": image_path  # Store the path where image is saved
                    }
                )
                
                image_documents.append(doc)
                print(f"Processed image: {filename} → {source_type}")
                
            except Exception as e:
                print(f"Failed to process image {image_file}: {str(e)}")
                # Create a basic document even if analysis fails
                basic_doc = Document(
                    page_content=f"IMAGE: {filename} - Processing failed: {str(e)}",
                    metadata={
                        "source": str(image_file),
                        "type": "image",
                        "analysis_method": "failed",
                        "content_type": "image_error"
                    }
                )
                image_documents.append(basic_doc)
        
        return image_documents

    def get_image_by_hash(self, file_hash):
        for filename in os.listdir(IMAGE_STORAGE_PATH):
            if filename.startswith(file_hash):
                return os.path.join(IMAGE_STORAGE_PATH, filename)
        return None

    def get_all_images(self):
        images = {}
        for filename in os.listdir(IMAGE_STORAGE_PATH):
            filepath = os.path.join(IMAGE_STORAGE_PATH, filename)
            images[filename] = filepath
        return images

# Global instance
caption_generator = ImageCaptionGenerator()

def generate_image_captions(image_files):
    if not image_files:
        return []
    
    print(f"Processing {len(image_files)} image files...")
    return caption_generator.process_image_files(image_files)

def is_vision_available():
    return caption_generator.client is not None

def get_image_path(file_hash):
    return caption_generator.get_image_by_hash(file_hash)

def get_stored_images():
    return caption_generator.get_all_images()