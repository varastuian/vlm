import io
import numpy as np
import cv2
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from starlette.responses import StreamingResponse
from PIL import Image
import rasterio
from rasterio.io import MemoryFile

from model import SiameseUNet, sliding_window_inference
from visualization import overlay_change_mask

app = FastAPI(
    title="UrbanSentinel API",
    description="Real-Time Change Detection for Satellite Imagery",
    version="1.0.0"
)

# --- Global Model Loader ---
model_instance = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def load_model():
    global model_instance
    # Initialize model functionality
    # In production, load weights: model.load_state_dict(torch.load("weights.pth"))
    model_instance = SiameseUNet(in_channels=4).to(DEVICE)
    model_instance.eval()
    print(f"Model loaded on {DEVICE}")

def read_image_file(file_content: bytes) -> np.ndarray:
    """Reads image bytes into numpy array (H, W, C)."""
    # Try Rasterio for GeoTIFFs
    try:
        with MemoryFile(file_content) as memfile:
            with memfile.open() as dataset:
                # Read specific bands B02, B03, B04, B08 if possible, or all
                # Assuming standard image upload for demo
                data = dataset.read() # (C, H, W)
                data = data.transpose(1, 2, 0) # (H, W, C)
                return data
    except Exception:
        # Fallback to standard image (PIL)
        image = Image.open(io.BytesIO(file_content)).convert("RGB")
        return np.array(image)

@app.post("/detect")
async def detect_change(
    file_t1: UploadFile = File(...), 
    file_t2: UploadFile = File(...),
    alpha: float = 0.5 
):
    """
    Accepts two images (Time T1 and Time T2) and an optional alpha transparency.
    Returns the Change Mask overlay image.
    """
    if not model_instance:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Read images
    try:
        content_t1 = await file_t1.read()
        content_t2 = await file_t2.read()
        
        img_t1 = read_image_file(content_t1)
        img_t2 = read_image_file(content_t2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # Validation (Dimensions)
    if img_t1.shape != img_t2.shape:
        # Resize T2 to match T1 for simple demo robustness
        img_t2 = cv2.resize(img_t2, (img_t1.shape[1], img_t1.shape[0]))

    # Preprocessing for Inference
    # Model expects (1, C, H, W) float tensors normalized
    # Handling 4 channels if available, else 3 (RGB)
    
    if img_t1.shape[2] == 3:
        # Add dummy NIR
        nir = np.zeros((img_t1.shape[0], img_t1.shape[1], 1), dtype=img_t1.dtype)
        img_t1 = np.concatenate([img_t1, nir], axis=2)
        img_t2 = np.concatenate([img_t2, nir], axis=2)

    # Convert to Tensor
    # Inference (Sliding Window or Direct)
    H, W = img_t1.shape[:2]
    window_size = 256
    stride = 128
    
    # --- CHANGE DETECTION LOGIC ---
    # Since the model is currently untrained (random weights), it will produce meaningless output.
    # For demonstration and pipeline verification, we use a robust baseline method:
    # Absolute Difference + Thresholding + Morphology
    
    # 1. Calculate absolute difference
    diff = cv2.absdiff(img_t1[:,:,:3], img_t2[:,:,:3]) # Use RGB
    
    # 2. Convert to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    
    # 3. Threshold (values > 30 are considered change)
    _, mask = cv2.threshold(diff_gray, 30, 1, cv2.THRESH_BINARY)
    
    # 4. Remove noise (Morphological Opening)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # ensure it's uint8
    mask = mask.astype(np.uint8)

    # Note: To use the AI model instead, uncomment the lines below and ensure weights are loaded.
    # if H > 1024 or W > 1024:
    #     ... (sliding window logic)
    # else:
    #     ... (direct inference)

    # Visualization
    # Use only RGB channels for base image visualization
    base_rgb = img_t1[:, :, :3].astype(np.uint8) if img_t1.dtype == np.uint8 else (img_t1[:, :, :3]).astype(np.uint8)
    
    vis_result = overlay_change_mask(base_rgb, mask, alpha=alpha)
    
    # Return as PNG
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR))
    if not is_success:
         raise HTTPException(status_code=500, detail="Image encoding failed")
         
    return Response(content=buffer.tobytes(), media_type="image/png")

@app.get("/health")
def health_check():
    return {"status": "active", "model": "Siam-U-Net-ResNet34"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
