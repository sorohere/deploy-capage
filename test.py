import requests
import json
from PIL import Image
import io
import base64
import os

def ensure_response_directory():
    """Ensure response directories exist"""
    os.makedirs('response', exist_ok=True)
    os.makedirs('response/attention', exist_ok=True)

def test_caption_generation(image_path):
    """Test the caption generation endpoint"""
    url = "http://127.0.0.1:5000/generate-caption"
    
    files = {
        'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')
    }
    
    try:
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            # Save caption response
            with open('response/caption.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            return result
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

def test_attention_maps(image_path):
    """Test the attention maps generation endpoint"""
    caption_url = "http://127.0.0.1:5000/generate-caption"
    files = {
        'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')
    }
    
    try:
        response = requests.post(caption_url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            attention_count = result.get('attention_count', 0)
            
            # Store all attention responses
            attention_responses = []
            
            # Now get each attention map
            for i in range(attention_count):
                attention_url = f"http://127.0.0.1:5000/generate-attention/{i}"
                files = {
                    'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')
                }
                
                response = requests.post(attention_url, files=files)
                
                if response.status_code == 200:
                    attention_data = response.json()
                    attention_responses.append(attention_data)
                    
                    # Save the Base64 image to a PNG file
                    img_data = base64.b64decode(attention_data['image_base64'])
                    with open(f'response/attention/attention_map_{i+1}.png', 'wb') as f:
                        f.write(img_data)
                    
                    # Remove the base64 image from the data before saving to JSON
                    attention_data_for_json = attention_data.copy()
                    attention_data_for_json['image_base64'] = '[BASE64_IMAGE_DATA]'  # Replace with placeholder
                    
                else:
                    print("Error Response:", response.text)
            
            # Save all attention responses to a single JSON file
            with open('response/attention_responses.json', 'w', encoding='utf-8') as f:
                json.dump(attention_responses, f, indent=4, ensure_ascii=False)
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    image_path = "test.png"
    
    # Ensure response directories exist
    ensure_response_directory()
    
    # Test caption generation
    caption_result = test_caption_generation(image_path)
    
    # Test attention maps generation
    test_attention_maps(image_path)
