import os
import datetime
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

app = Flask(__name__, static_folder='static', template_folder='templates')

# Directories for uploaded 6x6 images and the pre-existing 32x32 images
UPLOAD_FOLDER = os.path.join("static", "images_6x6")
OUTPUT_FOLDER = os.path.join("static", "upscaled_250_32x32")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------
# Device Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# (Optional) Generator Model Definition (6x6 → 32x32)
# (Currently unused in this approach, but kept for reference.)
# ---------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4/3, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

generator = Generator().to(device)

model_path = "models/generator.pth"
if os.path.exists(model_path):
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    print("✅ Generator loaded successfully (though not used here).")
else:
    print("⚠️ Generator model not found. (Not used in this approach.)")

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """
    1) Receive the uploaded 6×6 file, e.g. 'pressure_image_6x6_200.png'
    2) Save it to static/images_6x6 (optional).
    3) Build the corresponding 32×32 filename by replacing '_6x6_' with '_32x32_'.
    4) Check if it exists in static/upscaled_250_32x32.
    5) Return that file's URL or an error if not found.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded 6x6 image (for record keeping)
    upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_path)

    # Validate that filename has "_6x6_" in it
    if "_6x6_" not in file.filename:
        return jsonify({'error': "Filename must contain '_6x6_' to map to a 32x32 image."}), 400

    # Build the corresponding 32x32 filename
    corresponding_32x32 = file.filename.replace("_6x6_", "_32x32_")
    upscaled_path = os.path.join(OUTPUT_FOLDER, corresponding_32x32)

    # Check if the corresponding 32x32 file exists
    if not os.path.exists(upscaled_path):
        return jsonify({'error': f"32x32 file '{corresponding_32x32}' not found."}), 404

    # Return the existing 32x32 image URL
    return jsonify({
        'generated_image_url': f'/static/upscaled_250_32x32/{corresponding_32x32}'
    })

if __name__ == '__main__':
    app.run(debug=True)
