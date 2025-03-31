import os
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

app = FastAPI()

IMAGE_6X6_DIR = "images_6x6"
GENERATED_32X32_DIR = "generated_32x32"

# ✅ Endpoint to list available 6x6 images
@app.get("/list_images")
def list_images():
    images = [f for f in os.listdir(IMAGE_6X6_DIR) if f.endswith(".png")]
    return JSONResponse(content={"images": images})

# ✅ Endpoint to get corresponding 32x32 image
@app.get("/get_32x32/{image_name}")
def get_32x32(image_name: str):
    image_path = os.path.join(GENERATED_32X32_DIR, image_name)
    
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        return JSONResponse(content={"error": "32×32 image not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
