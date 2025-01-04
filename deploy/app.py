from flask import Flask, request, render_template, jsonify, send_file
import torch
import numpy as np
from PIL import Image
import cv2
import io
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
        # Preprocess the uploaded image
        image = Image.open(uploaded_image).convert("RGB")
        image = np.array(image)
        aspect_ratio = image.shape[0] / image.shape[1]
        new_height = int(480 * aspect_ratio)
        image = cv2.resize(image, (480, new_height))

        # Generate attention maps
        attentions, caption = model.predict(image, vocab)
        if index >= len(caption) - 1:
            return jsonify({"error": "Invalid attention index."}), 400

        # Create an attention map image
        temp_att = attentions[index].reshape(7, 7)
        att_resized = cv2.resize(temp_att, (image.shape[1], image.shape[0]))
        att_normalized = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min())
        heatmap = (att_normalized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

        # Convert to PNG and return
        _, buffer = cv2.imencode('.png', overlay)
        return send_file(
            io.BytesIO(buffer),
            mimetype='image/png',
            as_attachment=False,
            download_name=f"attention_{index}.png"
        )
    except Exception as e:
        return jsonify({"error": f"Error generating attention map: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
