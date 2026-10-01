import os
import cv2
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image, reshape_as_raster
from sentinelhub import (
    SHConfig, 
    SentinelHubRequest, 
    DataCollection, 
    MimeType, 
    Box, 
    CRS
)
from datetime import datetime
from typing import Tuple, List, Optional
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentinelIngestor:
    """
    Handles fetching of Sentinel-2 L2A imagery.
    Requires SH_CLIENT_ID and SH_CLIENT_SECRET environment variables.
    """
    def __init__(self, client_id: str = None, client_secret: str = None):
        self.config = SHConfig()
        self.config.sh_client_id = client_id or os.getenv("SH_CLIENT_ID")
        self.config.sh_client_secret = client_secret or os.getenv("SH_CLIENT_SECRET")
        
        if not self.config.sh_client_id or not self.config.sh_client_secret:
            logger.warning("SentinelHub credentials not provided. API calls will fail.")

    def fetch_data(self, roi_bbox:  Tuple[float, float, float, float], time_interval: Tuple[str, str], max_cloud_cover=0.1) -> np.ndarray:
        """
        Fetches Sentinel-2 L2A data for a given ROI and time interval.
        Returns the image with B04(Red), B03(Green), B02(Blue), B08(NIR).
        """
        bbox = Box(bbox=roi_bbox, crs=CRS.WGS84)
        
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B04", "B03", "B02", "B08"],
            output: { bands: 4 }
          };
        }
        function evaluatePixel(sample) {
          return [sample.B04, sample.B03, sample.B02, sample.B08];
        }
        """
        
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A, 
                    time_interval=time_interval,
                    maxcc=max_cloud_cover
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            bbox=bbox,
            resolution=(10, 10),
            config=self.config
        )
        # Note: Actual API call requires valid credentials. 
        # This is the architectural implementation.
        logger.info(f"Requesting data for interval: {time_interval}")
        # data = request.get_data()[0] 
        # For simulation without credits, return dummy
        logger.warning("Returning dummy data for demonstration (No API Credits)")
        return np.random.rand(1024, 1024, 4).astype(np.float32)

class Preprocessor:
    """
    Handles Co-registration and Radiometric Normalization.
    """
    
    @staticmethod
    def coregister_ecc(ref_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
        """
        Aligns target_img to ref_img using OpenCV's ECC algorithm.
        Input images should be (H, W, C) or (H, W).
        Uses the first channel (usually Red) for alignment calculation to save time, 
        then applies the warp to all channels.
        """
        logger.info("Starting ECC Co-registration...")
        
        # Convert to grayscale (or use one band) for registration matrix calculation
        # Assuming H, W, C format
        if len(ref_img.shape) == 3:
            ref_gray = ref_img[:, :, 0].astype(np.float32)
            target_gray = target_img[:, :, 0].astype(np.float32)
        else:
            ref_gray = ref_img.astype(np.float32)
            target_gray = target_img.astype(np.float32)

        # Initialize warp matrix (Identity)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        
        try:
            # Find the transform (Euclidean: translation + rotation)
            # MOTION_HOMOGRAPHY could be used for more complex distortions, 
            # but MOTION_EUCLIDEAN is usually sufficient for satellite if orthorectified.
            # Using MOTION_AFFINE to be safe for slight scale/shear diffs.
            mode = cv2.MOTION_AFFINE 
            _, warp_matrix = cv2.findTransformECC(ref_gray, target_gray, warp_matrix, mode, criteria)
            logger.info(f"ECC Converged. Warp Matrix:\n{warp_matrix}")
            
            # Warp the target image
            h, w = ref_img.shape[:2]
            aligned_target = cv2.warpAffine(
                target_img, 
                warp_matrix, 
                (w, h), 
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            return aligned_target
            
        except cv2.error as e:
            logger.error(f"ECC failed to converge: {e}")
            return target_img # Return original if fails

    @staticmethod
    def normalize_histogram_matching(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Performs Histogram Matching to normalize lighting conditions.
        Matches 'source' histogram to 'reference' histogram.
        """
        from skimage.exposure import match_histograms
        
        logger.info("Performing Histogram Matching...")
        matched = match_histograms(source, reference, channel_axis=-1)
        return matched.astype(source.dtype)

    @staticmethod
    def save_geotiff(data: np.ndarray, output_path: str, profile: dict):
        """
        Saves numpy array as GeoTIFF.
        Expects (H, W, C) input, converts to (C, H, W) for Rasterio.
        """
        # Ensure data is (C, H, W)
        if len(data.shape) == 3 and data.shape[2] <= 13: # Heuristic for HWC
            data = reshape_as_raster(data)
            
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
        logger.info(f"Saved GeoTIFF to {output_path}")
