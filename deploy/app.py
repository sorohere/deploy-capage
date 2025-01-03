from flask import Flask, request, render_template, jsonify
import torch
import numpy as np
from PIL import Image
import cv2
from construct.utils import set_cuda, image_transformation, plot_attention
from construct.architecture import Encoder_Decoder_Model, Vocabulary, Image_encoder, Attention_Based_Decoder, AttentionLayer

# Set up Flask app
app = Flask(__name__)

# Set the device for computation
device = set_cuda()

# Load the pre-trained model and vocabulary
model = torch.load("model/model.pt", map_location=device, weights_only=False)
vocab = torch.load("model/vocab.pth", map_location=device, weights_only=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    uploaded_image = request.files['image']
    image = Image.open(uploaded_image)
    image = np.array(image)

    # Preprocess the image for resizing
    aspect_ratio = image.shape[0] / image.shape[1]
    new_height = int(480 * aspect_ratio)
    image = cv2.resize(image, (480, new_height))

    try:
        # Generate caption and attention maps
        attentions, caption = model.predict(image, vocab)
        caption_text = ' '.join(caption[1:-1])

        return jsonify({
            "caption": caption_text
        })
    except Exception as e:
        return jsonify({"error": f"Error generating caption: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
