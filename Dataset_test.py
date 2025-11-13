import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random

# FIX: NUM_CLASSES is 4 to match the shape of your saved checkpoint (4, 64, 1, 1)
NUM_CLASSES = 4 

# Define colors for visualization, collapsing all lane classes into a single green color
COLOR_MAP = {
    0: (0, 0, 0),       # Black for Background
    1: (0, 255, 0),     # Green for Lane 1 (BGR)
    2: (0, 255, 0),     # Green for Lane 2 (BGR)
    3: (0, 255, 0),     # Green for Lane 3 (BGR)
}


try:
    from UnetModel import UNET
except ImportError:
    class UNET(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.down = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1) 
            print("LOG: Using minimal UNET placeholder. Ensure your 'UNetMD.py' is available for the real model.")
        
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


# --- 3. MODEL AND CONFIGURATION ---
TARGET_SIZE = (256, 512)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verify these paths are absolutely correct on your system
MODEL_PATH = r'D:\AI_BATCH3\Project samples\best_unet_lane_detector.pth'
TEST_SET_FOLDER = r'D:\Download\archive\TUSimple\test_set\clips' 


def predict_and_visualize(image_path):
    # 1. Load Model and Weights
    model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    
    print(f"LOG: Attempting to load weights from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at: {MODEL_PATH}")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("LOG: Weights loaded successfully. (NUM_CLASSES = 4)")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load state dict even with NUM_CLASSES=4: {e}")
        return
        
    model.eval()
    
    # 2. Load and Preprocess Image
    print(f"LOG: Reading image: {image_path}")
    original_img = cv2.imread(image_path)
    
    if original_img is None or original_img.size == 0:
        print(f"ERROR: Could not load or read valid image data from {image_path}. Check file permissions or corruption.")
        return

    # Prepare image for display (CV2 BGR -> Matplotlib RGB)
    display_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    H_orig, W_orig, _ = display_img_rgb.shape

    # Preprocess image for model input
    input_img = cv2.resize(original_img, (TARGET_SIZE[1], TARGET_SIZE[0]))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_tensor = input_img.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1)) 
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE) 
    
    print("LOG: Running inference...")
    # 3. Run Inference
    with torch.no_grad():
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
            output = model(input_tensor) 
            
    # 4. Post-process Prediction and Resize
    pred_probs = F.softmax(output, dim=1)
    pred_mask_indices_model_size = torch.argmax(pred_probs, dim=1).squeeze().cpu().numpy()
    
    pred_mask_resized_indices = cv2.resize(
        pred_mask_indices_model_size, 
        (W_orig, H_orig), 
        interpolation=cv2.INTER_NEAREST
    )

    # 5. Create Visualization Overlays
    overlay_color_bgr_raw_mask = colorize_mask(pred_mask_resized_indices)
    overlay_color_rgb_raw_mask = cv2.cvtColor(overlay_color_bgr_raw_mask, cv2.COLOR_BGR2RGB)
    
    blended_img_raw = display_img_rgb.copy()
    lane_mask_binary = (pred_mask_resized_indices != 0)
    
    if np.any(lane_mask_binary):
        blended_img_raw[lane_mask_binary] = cv2.addWeighted(
            display_img_rgb[lane_mask_binary], 
            1.0 - 0.5,
            overlay_color_rgb_raw_mask[lane_mask_binary], 
            0.5,
            0
        )
    else:
        print("LOG: No lane pixels detected by the model (empty mask). Blending skipped.")

    
    # 6. Display results using Matplotlib
    print("LOG: Preparing plot for display...")
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].imshow(display_img_rgb)
    ax[0].set_title('Original Image') 
    ax[0].axis('off')
    
    ax[1].imshow(overlay_color_rgb_raw_mask)
    ax[1].set_title('Raw Segmentation Mask (All Lanes in Green)')
    ax[1].axis('off')
    
    ax[2].imshow(blended_img_raw)
    ax[2].set_title('Blended Simple Lane Overlay') 
    ax[2].axis('off')
    
    plt.subplots_adjust(wspace=0.05, hspace=0) 
    plt.tight_layout()
    print("LOG: Calling plt.show() now...")
    plt.show() # This call opens the window


if __name__ == '__main__':
    TEST_IMAGE_PATH = find_random_image(TEST_SET_FOLDER)
    
    if TEST_IMAGE_PATH is None:
        print(f"ERROR: No JPG files found or accessible in: {TEST_SET_FOLDER}")
        print("Please verify the folder path and ensure there are .jpg files inside.")
    else:
        print(f"LOG: Test image selected: {TEST_IMAGE_PATH}")
        print("-" * 50)
        
        # Run the main prediction function
        try:
            predict_and_visualize(TEST_IMAGE_PATH)
        except Exception as e:
            print(f"\nFATAL ERROR DURING PREDICTION: {e}")
            print("An unexpected error occurred during model inference or visualization.")
