import cv2
import numpy as np

def overlay_change_mask(base_image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a binary Change Mask on the base image.
    Change pixels (1) are highlighted in Red (RGB: 255, 0, 0).
    
    Args:
        base_image: (H, W, 3) or (H, W, 4) uint8 image (RGB or RGBA).
        mask: (H, W) uint8 binary mask (0=No Change, 1=Change).
        alpha: Transparency factor for the overlay (0.0 - 1.0).
        
    Returns:
        (H, W, 3) uint8 image with overlay.
    """
    # Ensure base image is RGB (drop alpha if exists, or expand gray)
    if len(base_image.shape) == 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
    elif base_image.shape[2] == 4:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_RGBA2RGB)
        
    # Create a red overlay layer
    overlay = base_image.copy()
    
    # Define Red color (R, G, B)
    # Note: OpenCV standard is BGR, but we usually work in RGB in Python unless using cv2.imread/imshow directly.
    # Assuming input is RGB (from Rasterio/PIL). If BGR, swap to (0, 0, 255).
    # Let's assume RGB for consistency with pipeline.
    red_color = (255, 0, 0) 
    
    # Apply red color where mask is 1
    overlay[mask == 1] = red_color
    
    # Blend with original image
    # result = alpha * overlay + (1 - alpha) * base_image
    # But strictly, we only want to blend where mask == 1.
    # Where mask == 0, we want original image.
    
    output = base_image.copy()
    
    # Boolean indexing for masking
    mask_indices = (mask == 1)
    
    if np.any(mask_indices):
        # Simpler approach:
        # cv2.addWeighted applies to whole image. (Robust to flatten/shape issues)
        blended = cv2.addWeighted(base_image, 1 - alpha, overlay, alpha, 0)
        output[mask_indices] = blended[mask_indices]

    return output.astype(np.uint8)
