import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from tqdm import tqdm # For progress bar

# --- CONFIGURATION MATCHING CHECKPOINT ---
NUM_CLASSES = 4 

# Define colors for visualization (BGR format for OpenCV)
COLOR_MAP = {
    0: (0, 0, 0),  # Background (Black)
    1: (0, 255, 0),  # Solid White Lane (Green)
    2: (0, 165, 255), # Dashed White Lane (Orange/Amber)
    3: (0, 0, 255), # Yellow Lane (Red)
}


# --- PATHS AND SETUP ---
VIDEO_PATH = r'D:\Download\PXL_20251005_010326946.mp4'
OUTPUT_VIDEO_PATH = 'PXL_20251005_010326946.mp4' 
MODEL_PATH = r'D:\AI_BATCH3\Project samples\best_unet_lane_detector.pth'

# --- MODEL CONFIGURATION ---
TARGET_SIZE = (256, 512) # (Height, Width) used during training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from UnetModel import UNET
except ImportError:
    class UNET(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.down = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1) 
        
        def forward(self, x):
            x = F.relu(self.down(x))
            return self.final_conv(x)


# --- 2. Core Processing Functions ---

def load_model():
    """Initializes the model and loads the trained weights."""
    model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    
    print(f"LOG: Attempting to load weights from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at: {MODEL_PATH}")
        return None

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("LOG: Weights loaded successfully.")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load state dict: {e}")
        return None

def preprocess_frame(frame):
    """Resizes, normalizes, and converts a single OpenCV frame (BGR) for UNET input."""
    # Convert BGR to RGB for PyTorch model consistency
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    input_img = cv2.resize(input_img, (TARGET_SIZE[1], TARGET_SIZE[0]))
    
    # Normalize and change layout to (C, H, W)
    input_tensor = input_img.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1)) 
    
    # Convert to PyTorch tensor and add batch dimension (B, C, H, W)
    tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)
    
    return tensor

def postprocess_and_overlay(original_img, output_tensor, color_map=COLOR_MAP):
    H_orig, W_orig, _ = original_img.shape
    
    # 1. Post-process prediction (Multiclass to Index Mask)
    pred_probs = F.softmax(output_tensor, dim=1)
    # Find the class index with the highest probability
    pred_mask_indices_model_size = torch.argmax(pred_probs, dim=1).squeeze().cpu().numpy()
    
    # 2. Resize mask to original frame dimensions
    pred_mask_resized_indices = cv2.resize(
        pred_mask_indices_model_size, 
        (W_orig, H_orig), 
        interpolation=cv2.INTER_NEAREST
    )

    # 3. Create Colored Overlay (BGR)
    overlay_color_bgr = np.zeros_like(original_img, dtype=np.uint8)
    
    # Identify all pixels that are NOT background (class 0)
    lane_pixels = (pred_mask_resized_indices != 0)
    
    # Map all non-background pixels to the representative lane color (Green)
    lane_color = color_map.get(1, (0, 255, 0)) 
    overlay_color_bgr[lane_pixels] = lane_color
    
    # 4. Blend the original image and the color mask
    blended_img = cv2.addWeighted(
        original_img, 
        1.0 - 0.5, # Original weight
        overlay_color_bgr, 
        0.5, # Overlay weight (transparency)
        0 # Gamma
    )
    return blended_img


def process_video():
    """Main function to run the video prediction pipeline."""
    
    model = load_model()
    if model is None:
        return

    # Initialize Video Capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"CRITICAL ERROR: Could not open video file at {VIDEO_PATH}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"LOG: Processing video: {frame_width}x{frame_height} @ {fps:.2f} FPS ({frame_count} frames)")

    # Initialize Video Processor
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print(f"CRITICAL ERROR: Could not initialize VideoWriter for {OUTPUT_VIDEO_PATH}.")
        cap.release()
        return

    # Process frames
    print("LOG: Starting frame processing...")
    
    # Use tqdm for a progress bar over the total frame count
    for _ in tqdm(range(frame_count), desc="Processing Frames"):
        ret, frame = cap.read()
        
        if not ret:
            break

        # 1. Preprocess Frame
        input_tensor = preprocess_frame(frame)

        # 2. Inference
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
                output = model(input_tensor) 
        
        # 3. Post-process and Overlay
        blended_frame = postprocess_and_overlay(frame, output)

        # 4. Write Frame to Output Video
        out.write(blended_frame)

    # Cleanup
    cap.release()
    out.release()
    print(f"\n✨ Video processing complete. Output saved to: {OUTPUT_VIDEO_PATH}")
    print("Remember to check the frame-writing codec ('mp4v') if the output file is unplayable.")


if __name__ == '__main__':
    process_video()
