from flask import Flask, request, render_template, jsonify, send_file
import torch
import numpy as np
from PIL import Image
import cv2
import io
import base64
from construct.utils import set_cuda, image_transformation, plot_attention
from construct.architecture import Encoder_Decoder_Model, Vocabulary, Image_encoder, Attention_Based_Decoder, AttentionLayer

# Set up Flask app
app = Flask(__name__)

# Set the device for computation
device = set_cuda()

# Load the pre-trained model and vocabulary
model = torch.load("deploy/model/model.pt", map_location=device, weights_only=False)
vocab = torch.load("deploy/model/vocab.pth", map_location=device, weights_only=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    uploaded_image = request.files['image']
    try:
        # Preprocess the uploaded image
        image = Image.open(uploaded_image).convert("RGB")
        image = np.array(image)
        aspect_ratio = image.shape[0] / image.shape[1]
        new_height = int(480 * aspect_ratio)
        image = cv2.resize(image, (480, new_height))

        # Generate caption and attention maps
        attentions, caption = model.predict(image, vocab)
        caption_text = ' '.join(caption[1:-1])

        return jsonify({
            "caption": caption_text,
            "attention_count": len(caption) - 1
        })
    except Exception as e:
        return jsonify({"error": f"Error generating caption: {str(e)}"}), 500

@app.route('/generate-attention/<int:index>', methods=['POST'])
def generate_attention(index):
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    uploaded_image = request.files['image']
    
    try:
        # Process image same as generate_caption
        image = Image.open(io.BytesIO(uploaded_image.read()))
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = np.array(image)
        
        # Generate attention maps
        attentions, caption = model.predict(image, vocab)
        
        if index >= len(attentions):
            return jsonify({"error": "Invalid attention map index"}), 400
            
        # Create attention map visualization
        temp_att = attentions[index].reshape(7, 7)
        att_resized = cv2.resize(temp_att, (image.shape[1], image.shape[0]))
        att_normalized = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min())
        heatmap = (att_normalized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
        
        # Add the word as text
        cv2.putText(
            overlay, f"Word: {caption[index]}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Convert to Base64
        _, buffer = cv2.imencode('.png', overlay)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Return JSON with Base64 image and metadata
        return jsonify({
            "word": caption[index],
            "attention_index": index,
            "image_base64": img_base64
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
