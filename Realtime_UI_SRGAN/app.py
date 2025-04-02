from flask import Flask, jsonify, send_file
from flask_cors import CORS
import os
import random
from model_inference import upscale_image
from PIL import Image

app = Flask(__name__)
CORS(app)

IMAGE_FOLDER = "../images_6x6"
STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/predict", methods=["GET"])
def predict():
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".png")]
    if not image_files:
        return jsonify({"error": "No images found."}), 404

    # Select a random 6x6 image
    selected_image = random.choice(image_files)
    input_path = os.path.join(IMAGE_FOLDER, selected_image)

    # Save resized version of the original 6x6 image
    input_img = Image.open(input_path).convert("RGB")
    input_display_path = os.path.join(STATIC_FOLDER, "input_6x6.png")
    input_img.resize((300,300), Image.NEAREST).save(input_display_path)

    # Upscale with SR-GAN and save output
    output_img = upscale_image(input_path)
    output_path = os.path.join(STATIC_FOLDER, "output_srgan_32x32.png")
    output_img.resize((300, 300), Image.BICUBIC).save(output_path)


    # Return both image URLs
    return jsonify({
        "input_image": "http://127.0.0.1:5000/static/input_6x6.png",
        "output_image": "http://127.0.0.1:5000/static/output_srgan_32x32.png"
    })

# Serve static files
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_file(os.path.join(STATIC_FOLDER, filename), mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
