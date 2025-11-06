#!/usr/bin/env python3
"""
Enhanced Camera and YOLOv8 Object Detection Module
Using Picamera2 (rpicam-still backend) for better quality
With night vision support and full color imaging
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from picamera2 import Picamera2
    from libcamera import Transform, controls
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️  Picamera2 not available - falling back to OpenCV")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not available")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️  RPi.GPIO not available - night vision disabled")


class EnhancedCameraYOLO:
    def __init__(self):
        """Initialize camera with Picamera2 and YOLO"""
        print("  🔷 Initializing enhanced camera system (Picamera2)...")
        
        self.camera = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.detections = []
        self.detection_lock = threading.Lock()
        
        self.is_running = False
        self.capture_running = False
        self.detection_running = False
        self.camera_ready = False
        
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        self.frame_count = 0
        
        # Night vision
        self.night_vision_enabled = False
        self.brightness_threshold = config.NIGHT_VISION_THRESHOLD
        
        # Initialize night vision GPIO
        if GPIO_AVAILABLE:
            self._init_night_vision()
        
        # Initialize camera
        self._init_camera()
        
        # Load YOLO model
        self.model = None
        self.model_loaded = False
        if YOLO_AVAILABLE:
            self._load_yolo_model()
        
        print("  ✅ Enhanced camera initialized")
    
    def _init_night_vision(self):
        """Initialize IR LED control for night vision"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(config.NIGHT_VISION_GPIO, GPIO.OUT)
            GPIO.output(config.NIGHT_VISION_GPIO, GPIO.LOW)
            print("  🌙 Night vision GPIO initialized")
        except Exception as e:
            print(f"  ⚠️  Night vision GPIO failed: {e}")
    
    def _enable_night_vision(self, enable=True):
        """Toggle IR LEDs for night vision"""
        if GPIO_AVAILABLE:
            try:
                GPIO.output(config.NIGHT_VISION_GPIO, GPIO.HIGH if enable else GPIO.LOW)
                self.night_vision_enabled = enable
                status = "ON" if enable else "OFF"
                print(f"  🌙 Night vision: {status}")
            except Exception as e:
                print(f"  ⚠️  Night vision toggle failed: {e}")
    
    def _load_yolo_model(self):
        """Load YOLOv8 model"""
        try:
            print("  🧠 Loading YOLOv8 model...")
            model_path = config.YOLO_MODEL_PATH
            
            if not os.path.exists(model_path):
                print(f"     📥 Downloading YOLOv8n...")
                model_path = 'yolov8n.pt'
            
            self.model = YOLO(model_path)
            self.model.to('cpu')
            
            # Test inference
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy_image, verbose=False)
            
            self.model_loaded = True
            print("  ✅ YOLOv8 model loaded successfully")
            
        except Exception as e:
            print(f"  ❌ YOLO loading error: {e}")
            self.model_loaded = False
    
    def _init_camera(self):
        """Initialize camera with Picamera2 for best quality"""
        print("  📹 Initializing Picamera2...")
        
        try:
            if not PICAMERA2_AVAILABLE:
                raise Exception("Picamera2 not available")
            
            # Create Picamera2 instance
            self.camera = Picamera2()
            
            # Get camera properties
            print("  📷 Camera detected:")
            print(f"     Model: {self.camera.camera_properties.get('Model', 'Unknown')}")
            
            # Configure for high-quality video capture
            # Use full color processing for realistic images
            config_dict = self.camera.create_video_configuration(
                main={
                    "size": config.CAMERA_RESOLUTION,
                    "format": "RGB888"  # Full RGB for true colors
                },
                controls={
                    # Auto exposure and white balance for adaptive lighting
                    "AeEnable": True,
                    "AwbEnable": True,
                    "AwbMode": controls.AwbModeEnum.Auto,
                    
                    # Brightness and contrast
                    "Brightness": 0.0,  # -1.0 to 1.0
                    "Contrast": 1.2,    # Boost contrast slightly
                    
                    # Saturation for vivid colors
                    "Saturation": 1.3,  # 1.0 = normal, >1.0 = more vivid
                    
                    # Sharpness
                    "Sharpness": 1.5,   # Slightly sharper
                    
                    # Frame rate
                    "FrameRate": config.CAMERA_FPS,
                    
                    # Exposure time (auto-adjusted, but set max)
                    "ExposureTime": None,  # Let it auto-adjust
                    
                    # Analog gain (sensitivity)
                    "AnalogueGain": None,  # Auto
                    
                    # Noise reduction
                    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
                }
            )
            
            self.camera.configure(config_dict)
            
            # Start camera
            print("  ⏳ Starting camera...")
            self.camera.start()
            
            # Warm up camera - give time for auto exposure/white balance
            print("  ⏳ Warming up camera (auto-exposure)...")
            warmup_frames = 30
            for i in range(warmup_frames):
                frame = self.camera.capture_array()
                if i % 10 == 0:
                    brightness = frame.mean()
                    print(f"     Frame {i+1}/{warmup_frames} - Brightness: {brightness:.1f}")
                time.sleep(0.1)
            
            self.camera_ready = True
            print("  ✅ Picamera2 initialized successfully!")
            print("     Full RGB color mode enabled")
            print("     Auto white balance enabled")
            print("     High quality noise reduction enabled")
            self.camera_type = 'picamera2'
            
        except Exception as e:
            print(f"  ❌ Picamera2 initialization failed: {e}")
            print("  Falling back to OpenCV...")
            self._init_camera_opencv_fallback()
    
    def _init_camera_opencv_fallback(self):
        """Fallback to OpenCV if Picamera2 fails"""
        try:
            print("  📹 Initializing OpenCV camera...")
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera")
            
            # Set properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
            self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Enable auto exposure
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 55)
            self.camera.set(cv2.CAP_PROP_CONTRAST, 50)
            self.camera.set(cv2.CAP_PROP_SATURATION, 65)  # More vivid colors
            
            # Warmup
            print("  ⏳ Warming up camera...")
            for _ in range(30):
                self.camera.read()
                time.sleep(0.1)
            
            self.camera_ready = True
            print("  ✅ OpenCV camera initialized")
            self.camera_type = 'opencv'
            
        except Exception as e:
            print(f"  ❌ Camera initialization failed: {e}")
            self.camera = None
            self.camera_type = 'none'
            self.camera_ready = False
    
    def _capture_frames(self):
        """Continuous frame capture thread with night vision"""
        print("  🎥 Capture thread started")
        frame_errors = 0
        success_count = 0
        last_night_check = time.time()
        
        while self.capture_running:
            try:
                if self.camera is None:
                    frame = self._generate_placeholder("Camera not available")
                    with self.frame_lock:
                        self.frame = frame
                    time.sleep(1/config.CAMERA_FPS)
                    continue
                
                # Capture frame based on camera type
                if self.camera_type == 'picamera2':
                    # Picamera2: Direct array capture
                    frame = self.camera.capture_array()
                    
                    # Frame is already in RGB888 format
                    if frame is None or frame.size == 0:
                        frame_errors += 1
                        time.sleep(0.05)
                        continue
                    
                else:
                    # OpenCV fallback
                    ret, frame = self.camera.read()
                    
                    if not ret or frame is None or frame.size == 0:
                        frame_errors += 1
                        time.sleep(0.05)
                        continue
                    
                    # Convert BGR to RGB for consistency
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Reset error counter on success
                frame_errors = 0
                
                # Ensure correct resolution
                if frame.shape[0] != config.CAMERA_RESOLUTION[1] or frame.shape[1] != config.CAMERA_RESOLUTION[0]:
                    frame = cv2.resize(frame, config.CAMERA_RESOLUTION)
                
                # Check brightness for night vision
                current_time = time.time()
                if current_time - last_night_check > 2.0:  # Check every 2 seconds
                    brightness = frame.mean()
                    
                    if brightness < self.brightness_threshold and not self.night_vision_enabled:
                        print(f"  🌙 Low light detected ({brightness:.1f}) - enabling night vision")
                        self._enable_night_vision(True)
                    elif brightness > self.brightness_threshold + 20 and self.night_vision_enabled:
                        print(f"  ☀️ Good light detected ({brightness:.1f}) - disabling night vision")
                        self._enable_night_vision(False)
                    
                    last_night_check = current_time
                
                # Store frame
                with self.frame_lock:
                    self.frame = frame.copy()
                
                self.frame_count += 1
                success_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time + 0.001)
                self.fps_counter.append(fps)
                self.last_time = current_time
                
                # Log progress
                if success_count == 1:
                    brightness = frame.mean()
                    print(f"  ✅ First frame captured! (brightness: {brightness:.1f})")
                elif success_count % 60 == 0:  # Every 2 seconds at 30fps
                    avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
                    brightness = frame.mean()
                    nv_status = "🌙 ON" if self.night_vision_enabled else "☀️ OFF"
                    print(f"  ✅ {self.frame_count} frames | {avg_fps:.1f} FPS | brightness: {brightness:.1f} | NV: {nv_status}")
                
                # Small delay to target FPS
                time.sleep(1.0 / config.CAMERA_FPS)
                
            except Exception as e:
                print(f"  ❌ Capture error: {e}")
                frame_errors += 1
                if frame_errors > 10:
                    print("  ⚠️  Too many errors - reinitializing camera...")
                    self._reinit_camera()
                    frame_errors = 0
                time.sleep(0.1)
        
        print("  🛑 Capture thread stopped")
    
    def _reinit_camera(self):
        """Reinitialize camera on errors"""
        try:
            if self.camera_type == 'picamera2' and self.camera:
                self.camera.stop()
                time.sleep(0.5)
                self.camera.start()
            elif self.camera_type == 'opencv' and self.camera:
                self.camera.release()
                time.sleep(0.5)
                self._init_camera_opencv_fallback()
        except Exception as e:
            print(f"  ❌ Reinitialization failed: {e}")
    
    def _generate_placeholder(self, message="Waiting for camera..."):
        """Generate placeholder frame"""
        frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                         config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
        
        # Gradient background
        for i in range(frame.shape[0]):
            frame[i, :] = [20 + i//8, 15 + i//10, 35 + i//12]
        
        cv2.putText(frame, message, (100, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
        
        return frame
    
    def _yolo_detection_thread(self):
        """YOLOv8 detection thread"""
        print("  🔍 Detection thread started")
        
        if not self.model_loaded or self.model is None:
            print("  ⚠️  Detection unavailable - model not loaded")
            self.detection_running = False
            return
        
        while self.detection_running:
            try:
                if self.frame is None:
                    time.sleep(0.05)
                    continue
                
                with self.frame_lock:
                    frame = self.frame.copy()
                
                # Run YOLO inference
                results = self.model(
                    frame,
                    conf=config.YOLO_CONFIDENCE_THRESHOLD,
                    iou=config.YOLO_IOU_THRESHOLD,
                    verbose=False,
                    device='cpu'
                )
                
                detections = []
                
                if len(results) > 0:
                    for result in results:
                        boxes = result.boxes
                        
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            class_name = self.model.names[cls]
                            
                            detection = {
                                'class': class_name,
                                'confidence': round(conf, 3),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'center_x': int((x1 + x2) / 2),
                                'center_y': int((y1 + y2) / 2)
                            }
                            detections.append(detection)
                
                with self.detection_lock:
                    self.detections = detections
                
                time.sleep(0.05)  # ~20 detections per second
                
            except Exception as e:
                print(f"  ❌ Detection error: {e}")
                time.sleep(0.1)
        
        print("  🛑 Detection thread stopped")
    
    def start_detection(self):
        """Start capture and detection threads"""
        if self.is_running:
            return
        
        self.is_running = True
        self.capture_running = True
        self.detection_running = True
        
        # Start capture thread
        capture_thread = threading.Thread(
            target=self._capture_frames,
            daemon=True,
            name="CameraCapture"
        )
        capture_thread.start()
        
        # Start detection thread
        if self.model_loaded and self.model is not None:
            detection_thread = threading.Thread(
                target=self._yolo_detection_thread,
                daemon=True,
                name="YOLODetection"
            )
            detection_thread.start()
        
        print("  ✅ Detection started")
    
    def stop_detection(self):
        """Stop all threads"""
        self.detection_running = False
        self.capture_running = False
        time.sleep(0.5)
        self.is_running = False
    
    def get_frame_with_detections(self):
        """Get frame with bounding boxes"""
        with self.frame_lock:
            if self.frame is None:
                return self._generate_placeholder("Initializing camera...")
            frame = self.frame.copy()
        
        # Convert RGB to BGR for OpenCV drawing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        with self.detection_lock:
            detections = self.detections.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Color by confidence
            if conf > 0.8:
                color = (0, 255, 0)
            elif conf > 0.6:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(frame_bgr, (x1, y1 - text_size[1] - 8), 
                         (x1 + text_size[0] + 8, y1), color, -1)
            cv2.putText(frame_bgr, label, (x1 + 4, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add overlay
        frame_bgr = self._add_overlay(frame_bgr, len(detections))
        
        return frame_bgr
    
    def _add_overlay(self, frame, det_count):
        """Add FPS and info overlay"""
        h, w = frame.shape[:2]
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        # FPS
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Objects
        cv2.putText(frame, f"Objects: {det_count}", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        
        # Night vision indicator
        if self.night_vision_enabled:
            cv2.putText(frame, "🌙 NV", (w - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Model
        cv2.putText(frame, "YOLOv8", (w - 150, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return frame
    
    def get_frame(self):
        """Get current frame without detections"""
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def get_detections(self):
        """Get current detections"""
        with self.detection_lock:
            return self.detections.copy()
    
    def get_performance_stats(self):
        """Get performance statistics"""
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        return {
            'fps': round(avg_fps, 1),
            'detections_count': len(self.detections),
            'model_loaded': self.model_loaded,
            'camera_ready': self.camera_ready,
            'night_vision': self.night_vision_enabled
        }
    
    def cleanup(self):
        """Cleanup resources"""
        print("  Cleaning up camera...")
        self.stop_detection()
        
        if self.camera:
            try:
                if self.camera_type == 'picamera2':
                    self.camera.stop()
                else:
                    self.camera.release()
            except:
                pass
        
        if GPIO_AVAILABLE:
            try:
                GPIO.output(config.NIGHT_VISION_GPIO, GPIO.LOW)
                GPIO.cleanup([config.NIGHT_VISION_GPIO])
            except:
                pass