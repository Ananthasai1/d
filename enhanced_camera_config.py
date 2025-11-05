#!/usr/bin/env python3
"""
Enhanced Camera Configuration for CyberCrawl
Improved quality, colors, and image processing
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import config

class EnhancedImageProcessor:
    """Image enhancement for better quality and colors"""
    
    @staticmethod
    def enhance_frame(frame):
        """
        Apply multiple enhancements to improve image quality
        """
        # 1. Auto White Balance correction
        frame = EnhancedImageProcessor.auto_white_balance(frame)
        
        # 2. Brightness and Contrast adjustment
        frame = EnhancedImageProcessor.adjust_brightness_contrast(frame, 
                                                                   brightness=15, 
                                                                   contrast=20)
        
        # 3. Color saturation boost
        frame = EnhancedImageProcessor.boost_saturation(frame, factor=1.3)
        
        # 4. Sharpening
        frame = EnhancedImageProcessor.sharpen(frame)
        
        # 5. Noise reduction (optional, may reduce FPS)
        # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        return frame
    
    @staticmethod
    def auto_white_balance(frame):
        """Improve white balance for better color accuracy"""
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    @staticmethod
    def adjust_brightness_contrast(frame, brightness=0, contrast=0):
        """
        Adjust brightness and contrast
        brightness: -100 to 100
        contrast: -100 to 100
        """
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow
            frame = cv2.addWeighted(frame, alpha_b, frame, 0, gamma_b)
        
        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            frame = cv2.addWeighted(frame, alpha_c, frame, 0, gamma_c)
        
        return frame
    
    @staticmethod
    def boost_saturation(frame, factor=1.5):
        """Increase color saturation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def sharpen(frame):
        """Apply sharpening filter"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(frame, -1, kernel)
    
    @staticmethod
    def denoise_light(frame):
        """Light denoising for cleaner image"""
        return cv2.bilateralFilter(frame, 9, 75, 75)


def get_enhanced_camera_config():
    """
    Get optimized camera configuration for better quality
    """
    camera_config = {
        "main": {
            "size": config.CAMERA_RESOLUTION,
            "format": "RGB888"
        },
        "controls": {
            # Frame rate
            "FrameRate": config.CAMERA_FPS,
            
            # Auto White Balance
            "AwbEnable": True,
            "AwbMode": 0,  # Auto
            
            # Auto Exposure
            "AeEnable": True,
            "AeExposureMode": 0,  # Normal
            "AeConstraintMode": 0,  # Normal
            
            # Brightness (range: -1.0 to 1.0)
            "Brightness": 0.1,
            
            # Contrast (range: 0.0 to 2.0)
            "Contrast": 1.2,
            
            # Saturation (range: 0.0 to 2.0)
            "Saturation": 1.3,
            
            # Sharpness (range: 0.0 to 16.0)
            "Sharpness": 2.0,
            
            # Reduce noise
            "NoiseReductionMode": 1,  # Fast
        }
    }
    return camera_config


# Enhanced config.py settings
ENHANCED_SETTINGS = """
# ===== Enhanced Camera Settings =====
CAMERA_RESOLUTION = (640, 480)  # or (800, 600) for better quality
CAMERA_FPS = 15

# Image Enhancement
ENABLE_AUTO_ENHANCEMENT = True
BRIGHTNESS_ADJUST = 15      # -100 to 100
CONTRAST_ADJUST = 20        # -100 to 100
SATURATION_FACTOR = 1.3     # 0.5 to 2.0
ENABLE_SHARPENING = True
ENABLE_DENOISING = False    # Set True for cleaner image (reduces FPS)

# Video Quality
VIDEO_QUALITY = 95          # JPEG quality: 85-95 recommended

# Color Grading Presets
COLOR_PRESET = 'vibrant'    # Options: 'natural', 'vibrant', 'warm', 'cool', 'cinematic'

COLOR_PRESETS = {
    'natural': {
        'brightness': 0,
        'contrast': 10,
        'saturation': 1.0
    },
    'vibrant': {
        'brightness': 15,
        'contrast': 20,
        'saturation': 1.3
    },
    'warm': {
        'brightness': 10,
        'contrast': 15,
        'saturation': 1.2,
        'temperature': 'warm'  # Shift towards orange/red
    },
    'cool': {
        'brightness': 5,
        'contrast': 15,
        'saturation': 1.1,
        'temperature': 'cool'   # Shift towards blue
    },
    'cinematic': {
        'brightness': -5,
        'contrast': 30,
        'saturation': 1.4
    }
}
"""


class ColorGrading:
    """Apply color grading presets"""
    
    @staticmethod
    def apply_preset(frame, preset='vibrant'):
        """Apply color preset to frame"""
        
        if preset == 'warm':
            # Increase red/orange tones
            frame[:, :, 2] = np.clip(frame[:, :, 2] * 1.1, 0, 255)  # Red
            frame[:, :, 0] = np.clip(frame[:, :, 0] * 0.95, 0, 255)  # Blue
            
        elif preset == 'cool':
            # Increase blue tones
            frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.1, 0, 255)  # Blue
            frame[:, :, 2] = np.clip(frame[:, :, 2] * 0.95, 0, 255)  # Red
            
        elif preset == 'cinematic':
            # Crushed blacks, lifted shadows
            frame = ColorGrading.cinematic_look(frame)
        
        return frame
    
    @staticmethod
    def cinematic_look(frame):
        """Apply cinematic color grading"""
        # Convert to float
        frame_float = frame.astype(np.float32) / 255.0
        
        # Lift shadows (S-curve)
        frame_float = np.power(frame_float, 0.9)
        
        # Crush blacks
        frame_float = np.where(frame_float < 0.1, frame_float * 0.8, frame_float)
        
        # Slight teal in shadows, orange in highlights
        hsv = cv2.cvtColor((frame_float * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = np.where(hsv[:, :, 2] < 100, 
                                np.clip(hsv[:, :, 0] + 10, 0, 180),  # Teal in shadows
                                np.clip(hsv[:, :, 0] - 5, 0, 180))   # Orange in highlights
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Example usage in camera_yolo.py
def process_captured_frame(frame, enable_enhancement=True, preset='vibrant'):
    """
    Process frame with enhancements
    Add this to your _capture_frames() method
    """
    if enable_enhancement:
        # Apply color preset
        frame = ColorGrading.apply_preset(frame, preset)
        
        # Apply image enhancements
        frame = EnhancedImageProcessor.enhance_frame(frame)
    
    return frame