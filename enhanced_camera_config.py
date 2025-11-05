#!/usr/bin/env python3
"""
Enhanced Camera Configuration for CyberCrawl
Optimized for better quality, higher FPS, and natural-looking images
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import config

class OptimizedImageProcessor:
    """Optimized image processing for natural appearance and performance"""
    
    @staticmethod
    def process_frame(frame, preset='natural_hq'):
        """
        Process frame with optimized settings for quality and speed
        
        Presets:
        - 'natural_hq': High quality, natural colors (recommended)
        - 'performance': Maximum FPS, minimal processing
        - 'balanced': Balance between quality and speed
        """
        
        if preset == 'natural_hq':
            return OptimizedImageProcessor._natural_hq_pipeline(frame)
        elif preset == 'performance':
            return OptimizedImageProcessor._performance_pipeline(frame)
        elif preset == 'balanced':
            return OptimizedImageProcessor._balanced_pipeline(frame)
        else:
            return frame
    
    @staticmethod
    def _natural_hq_pipeline(frame):
        """High quality natural image processing"""
        
        # 1. Gentle denoising (preserves detail)
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
        
        # 2. Subtle contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame = cv2.merge([l, a, b])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        # 3. Subtle sharpening
        kernel = np.array([[-0.5, -0.5, -0.5],
                          [-0.5,  5.0, -0.5],
                          [-0.5, -0.5, -0.5]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        return frame
    
    @staticmethod
    def _performance_pipeline(frame):
        """Minimal processing for maximum FPS"""
        # Only basic white balance correction
        frame = OptimizedImageProcessor._simple_white_balance(frame)
        return frame
    
    @staticmethod
    def _balanced_pipeline(frame):
        """Balanced quality and performance"""
        
        # 1. Light denoising
        frame = cv2.bilateralFilter(frame, 5, 50, 50)
        
        # 2. Moderate contrast
        alpha = 1.1  # Contrast
        beta = 5     # Brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        return frame
    
    @staticmethod
    def _simple_white_balance(frame):
        """Fast white balance correction"""
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * 0.5)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * 0.5)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def get_optimized_camera_config(quality_mode='high'):
    """
    Get optimized camera configuration
    
    Modes:
    - 'high': Best quality, ~15-20 FPS (800x600)
    - 'balanced': Good quality, ~20-25 FPS (640x480)
    - 'performance': Maximum FPS, ~25-30 FPS (320x240)
    """
    
    configs = {
        'high': {
            'resolution': (800, 600),
            'fps': 20,
            'processing': 'natural_hq'
        },
        'balanced': {
            'resolution': (640, 480),
            'fps': 25,
            'processing': 'balanced'
        },
        'performance': {
            'resolution': (320, 240),
            'fps': 30,
            'processing': 'performance'
        }
    }
    
    selected = configs.get(quality_mode, configs['balanced'])
    
    camera_config = {
        "main": {
            "size": selected['resolution'],
            "format": "RGB888"
        },
        "controls": {
            # Frame rate - higher for smoother video
            "FrameRate": selected['fps'],
            
            # Auto White Balance - natural colors
            "AwbEnable": True,
            "AwbMode": 0,  # Auto (most natural)
            
            # Auto Exposure - well-lit images
            "AeEnable": True,
            "AeExposureMode": 0,  # Normal exposure
            "AeConstraintMode": 0,  # Normal constraint
            "AeMeteringMode": 0,    # Centre-weighted
            
            # Brightness (0.0 = neutral, natural look)
            "Brightness": 0.0,
            
            # Contrast (1.0 = neutral, natural)
            "Contrast": 1.0,
            
            # Saturation (1.0 = neutral, realistic colors)
            "Saturation": 1.0,
            
            # Sharpness (1.0 = moderate, not oversharpened)
            "Sharpness": 1.0,
            
            # Noise reduction - minimal for natural look
            "NoiseReductionMode": 1,  # Fast (light denoising)
            
            # Color gains - neutral for natural colors
            "ColourGains": (1.0, 1.0),
        }
    }
    
    return camera_config, selected['processing']


# ===== Updated config.py Settings =====
OPTIMIZED_CONFIG = """
# ===== Optimized Camera Settings =====

# Choose quality mode: 'high', 'balanced', or 'performance'
CAMERA_QUALITY_MODE = 'balanced'

# Resolution (auto-selected based on quality mode)
# high: 800x600, balanced: 640x480, performance: 320x240
CAMERA_RESOLUTION = (640, 480)

# FPS (auto-selected based on quality mode)
# high: 20, balanced: 25, performance: 30
CAMERA_FPS = 25

# Image processing preset
# 'natural_hq': Best quality, natural colors
# 'balanced': Good quality, faster
# 'performance': Maximum speed, minimal processing
IMAGE_PROCESSING_PRESET = 'balanced'

# Camera warmup time (seconds)
CAMERA_WARMUP_TIME = 2

# YOLO detection interval (seconds)
# Lower = more frequent detection but lower FPS
# Higher = less frequent detection but higher FPS
DETECTION_INTERVAL = 0.15  # Optimized for balance

# Video quality (JPEG compression)
VIDEO_QUALITY = 90  # 85-95 recommended (higher = better quality, larger size)

# Advanced: Frame buffer size
FRAME_BUFFER_SIZE = 3  # Reduce lag

# Advanced: Enable hardware acceleration
ENABLE_HARDWARE_ACCEL = True
"""


# ===== Usage in camera_yolo.py =====
class OptimizedCameraYOLO:
    """
    Drop-in replacement for EnhancedCameraYOLO with optimization
    Add these methods to your existing EnhancedCameraYOLO class
    """
    
    def _init_camera_optimized(self):
        """Initialize camera with optimized settings"""
        print("  🎥 Initializing optimized camera...")
        
        try:
            self.camera = Picamera2()
            
            # Get optimized config
            quality_mode = getattr(config, 'CAMERA_QUALITY_MODE', 'balanced')
            camera_config, self.processing_preset = get_optimized_camera_config(quality_mode)
            
            self.camera.configure(camera_config)
            
            # Set buffer configuration for lower latency
            self.camera.set_controls({
                "FrameDurationLimits": (33333, 33333)  # ~30 FPS max
            })
            
            # Start camera
            print(f"  ⏳ Starting camera in '{quality_mode}' mode...")
            self.camera.start()
            
            # Shorter warmup for faster startup
            print(f"  ⏳ Warming up camera (1s)...")
            time.sleep(1)
            
            # Test capture
            test_frame = self.camera.capture_array()
            if test_frame is not None and test_frame.size > 0:
                print(f"  ✅ Camera ready! Resolution: {test_frame.shape}, Mode: {quality_mode}")
                self.camera_ready = True
                self.camera_type = 'picamera2'
            else:
                raise Exception("Failed to capture test frame")
            
        except Exception as e:
            print(f"  ❌ Camera initialization failed: {e}")
            self.camera = None
            self.camera_type = 'none'
            self.camera_ready = False
    
    def _capture_frames_optimized(self):
        """Optimized frame capture with processing"""
        print("  🎬 Optimized capture thread started")
        
        consecutive_errors = 0
        max_errors = 10
        frame_skip = 0
        
        while self.capture_running:
            try:
                if self.camera is None or not self.camera_ready:
                    frame = self._generate_placeholder("Camera not available")
                    with self.frame_lock:
                        self.frame = frame
                    time.sleep(1)
                    continue
                
                # Capture frame
                frame = self.camera.capture_array()
                
                if frame is None or frame.size == 0:
                    consecutive_errors += 1
                    if consecutive_errors > max_errors:
                        print(f"  ⚠️  Too many capture errors, reinitializing...")
                        self._reinit_camera()
                        consecutive_errors = 0
                    time.sleep(0.1)
                    continue
                
                consecutive_errors = 0
                
                # Convert RGB to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Apply rotation if needed
                if config.CAMERA_ROTATION != 0:
                    if config.CAMERA_ROTATION == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif config.CAMERA_ROTATION == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif config.CAMERA_ROTATION == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Apply optimized processing (every frame for quality)
                preset = getattr(config, 'IMAGE_PROCESSING_PRESET', 'balanced')
                frame = OptimizedImageProcessor.process_frame(frame, preset)
                
                # Store frame
                with self.frame_lock:
                    self.frame = frame.copy()
                
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time + 0.001)
                self.fps_counter.append(fps)
                self.last_time = current_time
                
                # Dynamic frame rate control
                target_delay = 1.0 / config.CAMERA_FPS
                time.sleep(max(0.001, target_delay))
                
            except Exception as e:
                consecutive_errors += 1
                print(f"  ❌ Capture error: {e}")
                if consecutive_errors > max_errors:
                    print(f"  ⚠️  Too many errors, stopping capture")
                    self.capture_running = False
                time.sleep(0.1)
        
        print("  🛑 Optimized capture thread stopped")


# ===== Quick Apply Instructions =====
"""
TO APPLY THESE OPTIMIZATIONS:

1. Update config.py - Add these settings:
   
   CAMERA_QUALITY_MODE = 'balanced'  # or 'high' or 'performance'
   CAMERA_RESOLUTION = (640, 480)
   CAMERA_FPS = 25
   IMAGE_PROCESSING_PRESET = 'balanced'
   DETECTION_INTERVAL = 0.15
   VIDEO_QUALITY = 90

2. Update camera/camera_yolo.py - Replace these methods:
   
   - Replace _init_camera() with _init_camera_optimized()
   - Replace _capture_frames() with _capture_frames_optimized()
   - Add: from enhanced_camera_config import OptimizedImageProcessor, get_optimized_camera_config

3. Test different modes:
   
   For best quality (natural image, ~15-20 FPS):
   CAMERA_QUALITY_MODE = 'high'
   
   For balanced (good quality, ~20-25 FPS):
   CAMERA_QUALITY_MODE = 'balanced'
   
   For maximum FPS (~25-30 FPS):
   CAMERA_QUALITY_MODE = 'performance'

4. Restart the application:
   
   python app.py

EXPECTED IMPROVEMENTS:
- FPS: 3.2 → 20-25 fps (balanced mode)
- Quality: More natural colors and lighting
- Reduced noise and artifacts
- Lower latency
- Smoother video stream
"""