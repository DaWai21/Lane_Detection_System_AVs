import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm # For a nice progress bar

# --- CONFIGURATION MATCHING CHECKPOINT ---
# FIX: NUM_CLASSES is 4 to match the shape of your saved checkpoint (4, 64, 1, 1)
NUM_CLASSES = 4 

# Define colors for visualization, collapsing all lane classes into a single green color
COLOR_MAP = {
    0: (0, 0, 0),       # Black for Background
    1: (0, 255, 0),     # Green for Lane 1 (BGR)
    2: (0, 255, 0),     # Green for Lane 2 (BGR)
    3: (0, 255, 0),     # Green for Lane 3 (BGR)
}


# --- 1. UNET Model Definition (Placeholder) ---
# NOTE: Ensure your 'UnetModel.py' is available for the real model.
try:
    from UnetModel import UNET
except ImportError:
    class UNET(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.down = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1) 
            print("LOG: Using minimal UNET placeholder. Ensure your 'UNetModel.py' is available for the real model.")
        
        def forward(self, x):
            x = F.relu(self.down(x))
            return self.final_conv(x)


# --- 2. Helper Functions ---
def colorize_mask(mask_indices, color_map=COLOR_MAP):
    """Converts a single-channel index mask (H, W) into a 3-channel BGR colored image."""
    height, width = mask_indices.shape
    color_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Identify all pixels that are NOT background (class 0)
    lane_pixels = (mask_indices != 0)
    
    # Map all non-background pixels to the representative lane color (Green)
    # Use the color for class 1 as the representative color
    lane_color = color_map.get(1, (0, 255, 0)) 
    color_img[lane_pixels] = lane_color
        
    return color_img

def find_random_image(root_dir):
    """Recursively finds all JPG files and returns the path to a random one."""
    print(f"LOG: Searching for random image in: {root_dir}")
    all_image_paths = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                all_image_paths.append(os.path.join(root, file))
    
    if not all_image_paths:
        return None
    
    # Select a random path and verify it exists
    selected_path = random.choice(all_image_paths)
    if os.path.exists(selected_path):
        return selected_path
    return None

def find_random_video(root_dir):
    """Recursively finds all video files (mp4, avi) and returns the path to a random one."""
    print(f"LOG: Searching for random video in: {root_dir}")
    all_video_paths = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                all_video_paths.append(os.path.join(root, file))
    
    if not all_video_paths:
        return None
    
    # Select a random path and verify it exists
    selected_path = random.choice(all_video_paths)
    if os.path.exists(selected_path):
        return selected_path
    return None


# --- 3. MODEL AND CONFIGURATION ---
TARGET_SIZE = (256, 512) # H x W for model input
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verify these paths are absolutely correct on your system
VIDEO_PATH = r'D:\Download\PXL_20251005_010326946.mp4'
OUTPUT_VIDEO_PATH = 'test6.mp4' 
MODEL_PATH = r'D:\AI_BATCH3\Project samples\.venv\Scripts\best_unet_lane_detector.pth'
TEST_SET_FOLDER = r'D:\Download\archive\TUSimple\test_set\clips' 
# --- 4. Core Prediction Logic (Reusable for Image and Video) ---

# We define the model outside the prediction functions to load it only once
def load_model(model_path):
    """Loads the UNET model and weights."""
    model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    
    print(f"LOG: Attempting to load weights from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at: {model_path}")
        return None

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("LOG: Weights loaded successfully. (NUM_CLASSES = 4)")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load state dict even with NUM_CLASSES=4: {e}")
        return None
        
    model.eval()
    return model

# Global variable for model
MODEL = None

def process_frame(model, frame_bgr):
    """
    Performs lane detection on a single BGR frame.
    Returns the blended BGR image.
    """
    if model is None:
        raise ValueError("Model is not loaded.")
        
    H_orig, W_orig, _ = frame_bgr.shape

    # Preprocess image for model input
    input_img = cv2.resize(frame_bgr, (TARGET_SIZE[1], TARGET_SIZE[0]))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_tensor = input_img.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1)) 
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE) 
    
    # Run Inference
    with torch.no_grad():
        # Use autocast for potential speed up on GPU
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
            output = model(input_tensor) 
            
    # Post-process Prediction and Resize
    pred_probs = F.softmax(output, dim=1)
    pred_mask_indices_model_size = torch.argmax(pred_probs, dim=1).squeeze().cpu().numpy()
    
    pred_mask_resized_indices = cv2.resize(
        pred_mask_indices_model_size, 
        (W_orig, H_orig), 
        interpolation=cv2.INTER_NEAREST
    )

    # Create Visualization Overlay (in BGR format)
    overlay_color_bgr_raw_mask = colorize_mask(pred_mask_resized_indices)
    
    # Blending
    blended_img_bgr = frame_bgr.copy()
    lane_mask_binary = (pred_mask_resized_indices != 0)
    
    if np.any(lane_mask_binary):
        # cv2.addWeighted works with BGR directly
        blended_img_bgr[lane_mask_binary] = cv2.addWeighted(
            frame_bgr[lane_mask_binary], 
            0.5, # Original weight
            overlay_color_bgr_raw_mask[lane_mask_binary], 
            0.5, # Overlay weight
            0
        )
    
    return blended_img_bgr


# --- 5. Video Processing Function ---

def process_video(model, VIDEO_PATH, output_dir):
    """Processes an input video and saves the output with lane overlays."""
    print("-" * 50)
    print(f"LOG: Starting video processing for: {VIDEO_PATH}")
    
    # 1. Setup Video Capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {VIDEO_PATH}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 2. Setup Video Writer
    os.makedirs(output_dir, exist_ok=True)
    video_filename = os.path.basename(VIDEO_PATH)
    output_path = os.path.join(output_dir, f"detected_{video_filename}")
    
    # Use MP4V codec which is often compatible, or 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"CRITICAL ERROR: Cannot open VideoWriter for {output_path}. Check codec (e.g., install relevant ffmpeg/vfw codecs).")
        cap.release()
        return

    print(f"LOG: Input video resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}")
    print(f"LOG: Output video path: {output_path}")

    # 3. Process Frames
    for _ in tqdm(range(frame_count), desc="Processing Video Frames"):
        ret, frame_bgr = cap.read()
        
        if not ret:
            break
        
        # Process the frame
        try:
            blended_frame = process_frame(model, frame_bgr)
            out.write(blended_frame)
        except Exception as e:
            print(f"\nError processing frame: {e}")
            # Write the original frame to avoid video corruption
            out.write(frame_bgr) 
            

    # 4. Cleanup
    cap.release()
    out.release()
    print("-" * 50)
    print(f"SUCCESS: Video processing complete. Saved to: {output_path}")
    print("-" * 50)


# --- 6. Original Image Visualization Function (Simplified) ---

def predict_and_visualize_image(model, image_path):
    """Loads image, runs prediction, and displays results using Matplotlib."""
    
    print(f"LOG: Reading image: {image_path}")
    original_img_bgr = cv2.imread(image_path)
    
    if original_img_bgr is None or original_img_bgr.size == 0:
        print(f"ERROR: Could not load or read valid image data from {image_path}. Check file permissions or corruption.")
        return

    # Use the reusable frame processing logic to get the blended image
    blended_img_bgr = process_frame(model, original_img_bgr)
    
    # Prepare images for Matplotlib (BGR -> RGB)
    display_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
    blended_img_rgb = cv2.cvtColor(blended_img_bgr, cv2.COLOR_BGR2RGB)
    
    # To display the raw mask, we re-run the core logic to get indices
    # This is slightly inefficient but keeps the display function self-contained
    H_orig, W_orig, _ = original_img_bgr.shape
    input_img = cv2.resize(original_img_bgr, (TARGET_SIZE[1], TARGET_SIZE[0]))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_tensor = input_img.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1)) 
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE) 
    
    with torch.no_grad():
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
            output = model(input_tensor) 
            
    pred_probs = F.softmax(output, dim=1)
    pred_mask_indices_model_size = torch.argmax(pred_probs, dim=1).squeeze().cpu().numpy()
    pred_mask_resized_indices = cv2.resize(
        pred_mask_indices_model_size, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST
    )
    overlay_color_bgr_raw_mask = colorize_mask(pred_mask_resized_indices)
    overlay_color_rgb_raw_mask = cv2.cvtColor(overlay_color_bgr_raw_mask, cv2.COLOR_BGR2RGB)
    
    # Display results using Matplotlib
    print("LOG: Preparing plot for display...")
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].imshow(display_img_rgb)
    ax[0].set_title('Original Image') 
    ax[0].axis('off')
    
    ax[1].imshow(overlay_color_rgb_raw_mask)
    ax[1].set_title('Raw Segmentation Mask (All Lanes in Green)')
    ax[1].axis('off')
    
    ax[2].imshow(blended_img_rgb)
    ax[2].set_title('Blended Simple Lane Overlay') 
    ax[2].axis('off')
    
    plt.subplots_adjust(wspace=0.05, hspace=0) 
    plt.tight_layout()
    print("LOG: Calling plt.show() now...")
    plt.show()


if __name__ == '__main__':
    
    # Load Model once
    MODEL = load_model(MODEL_PATH)
    if MODEL is None:
        print("Script terminated due to critical model loading error.")
        exit()

    # --- MODE SELECTION ---
    # Change to 'video' to test video processing
    MODE = 'video' # Options: 'image', 'video'
    
    if MODE == 'image':
        # --- IMAGE MODE ---
        TEST_IMAGE_PATH = find_random_image(TEST_SET_FOLDER)
        
        if TEST_IMAGE_PATH is None:
            print(f"ERROR: No JPG files found or accessible in: {TEST_SET_FOLDER}")
            print("Please verify the folder path and ensure there are .jpg files inside.")
        else:
            print(f"LOG: Test image selected: {TEST_IMAGE_PATH}")
            print("-" * 50)
            
            # Run the main prediction function
            try:
                predict_and_visualize_image(MODEL, TEST_IMAGE_PATH)
            except Exception as e:
                print(f"\nFATAL ERROR DURING IMAGE PREDICTION: {e}")
                print("An unexpected error occurred during model inference or visualization.")

    elif MODE == 'video':
        # --- VIDEO MODE ---
        # NOTE: This assumes there are videos in the TEST_SET_FOLDER (e.g., .mp4 or .avi)
        TEST_VIDEO_PATH = find_random_video(VIDEO_PATH)
        
        if TEST_VIDEO_PATH is None:
            print(f"ERROR: No video files (.mp4, .avi) found or accessible in: {VIDEO_PATH}")
            print("Please verify the folder path and ensure there are video files inside.")
        else:
            # Run the video processing function
            try:
                process_video(MODEL, TEST_VIDEO_PATH, VIDEO_PATH)
            except Exception as e:
                print(f"\nFATAL ERROR DURING VIDEO PROCESSING: {e}")
                print("An unexpected error occurred during video processing.")
    
    else:
        print(f"ERROR: Invalid MODE '{MODE}'. Please choose 'image' or 'video'.")