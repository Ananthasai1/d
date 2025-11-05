#!/usr/bin/env python3
"""
Enhanced Camera Configuration with YOLOv8 for CyberCrawl
Optimized for better FPS, quality, and natural colors
Fixed for PyTorch 2.6+ compatibility
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️  picamera2 not available")

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not available")


class OptimizedImageProcessor:
    """Enhanced image processing for natural appearance and performance"""
    
    @staticmethod
    def process_frame(frame, preset='balanced'):
        """Process frame with optimized settings"""
        if preset == 'natural_hq':
            return OptimizedImageProcessor._natural_hq_pipeline(frame)
        elif preset == 'performance':
            return frame  # No processing for max speed
        elif preset == 'balanced':
            return OptimizedImageProcessor._balanced_pipeline(frame)
        else:
            return frame
    
    @staticmethod
    def _natural_hq_pipeline(frame):
        """High quality natural image processing"""
        # Gentle denoising
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
        
        # Subtle contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame = cv2.merge([l, a, b])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        return frame
    
    @staticmethod
    def _balanced_pipeline(frame):
        """Balanced quality and performance"""
        # Light denoising
        frame = cv2.bilateralFilter(frame, 5, 50, 50)
        
        # Moderate contrast
        alpha = 1.1  # Contrast
        beta = 5     # Brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        return frame


class EnhancedCameraYOLO:
    """Enhanced camera with YOLOv8 object detection and optimized image processing"""
    
    def __init__(self):
        """Initialize camera and YOLO"""
        print("📷 Initializing enhanced camera system...")
        
        # Camera and frame variables
        self.camera = None
        self.frame = None
        self.frame_lock = threading.Lock()
        
        # Detection variables
        self.detections = []
        self.detection_lock = threading.Lock()
        
        # Threading control
        self.is_running = False
        self.capture_running = False
        self.detection_running = False
        self.camera_ready = False
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        self.frame_count = 0
        self.detection_count = 0
        
        # Processing preset
        self.processing_preset = getattr(config, 'IMAGE_PROCESSING_PRESET', 'balanced')
        
        # YOLO model
        self.model = None
        self.model_loaded = False
        
        # Initialize camera with enhanced settings
        self._init_camera_optimized()
        
        # Load YOLO model if available
        if YOLO_AVAILABLE:
            self._load_yolo_model()
        
        print("✅ Enhanced camera system initialized")
    
    def _init_camera_optimized(self):
        """Initialize camera with optimized settings"""
        print("  🎥 Initializing optimized camera...")
        
        if not PICAMERA2_AVAILABLE:
            print("  ❌ picamera2 not available")
            self.camera_type = 'none'
            return
        
        try:
            # Create Picamera2 instance
            self.camera = Picamera2()
            
            # Get quality mode from config
            quality_mode = getattr(config, 'CAMERA_QUALITY_MODE', 'balanced')
            
            # Optimized configurations per mode
            configs = {
                'high': {
                    'resolution': (800, 600),
                    'fps': 20
                },
                'balanced': {
                    'resolution': (640, 480),
                    'fps': 25
                },
                'performance': {
                    'resolution': (320, 240),
                    'fps': 30
                }
            }
            
            selected = configs.get(quality_mode, configs['balanced'])
            
            # Configure camera with optimized settings
            camera_config = self.camera.create_still_configuration(
                main={"size": selected['resolution'], "format": "RGB888"},
                controls={
                    # Frame rate - optimized for smoothness
                    "FrameRate": selected['fps'],
                    
                    # Auto White Balance - natural colors
                    "AwbEnable": True,
                    "AwbMode": 0,  # Auto mode
                    
                    # Auto Exposure - well-lit images
                    "AeEnable": True,
                    "AeExposureMode": 0,  # Normal exposure
                    "AeConstraintMode": 0,  # Normal constraint
                    "AeMeteringMode": 0,    # Centre-weighted
                    
                    # Image adjustments - neutral for natural look
                    "Brightness": 0.0,    # Neutral
                    "Contrast": 1.0,      # Neutral
                    "Saturation": 1.0,    # Neutral
                    "Sharpness": 1.0,     # Moderate
                    
                    # Noise reduction - light for natural look
                    "NoiseReductionMode": 1,  # Fast (minimal)
                }
            )
            
            self.camera.configure(camera_config)
            
            # Set frame duration for consistent FPS
            frame_duration = int(1000000 / selected['fps'])  # Microseconds
            self.camera.set_controls({
                "FrameDurationLimits": (frame_duration, frame_duration)
            })
            
            # Start camera
            print(f"  ⏳ Starting camera in '{quality_mode}' mode...")
            self.camera.start()
            
            # Reduced warmup time
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
    
    def _load_yolo_model(self):
        """Load YOLOv8 model with PyTorch 2.6+ compatibility"""
        try:
            print("  🧠 Loading YOLOv8 model...")
            
            model_path = config.YOLO_MODEL_PATH
            
            if not os.path.exists(model_path):
                print(f"  📥 Downloading {model_path}...")
            
            # Add safe globals for PyTorch 2.6+ compatibility
            # This allows loading YOLO model weights safely
            try:
                torch.serialization.add_safe_globals([DetectionModel])
                print("  ✅ PyTorch security globals configured")
            except AttributeError:
                # Fallback for older PyTorch versions
                print("  ℹ️  Using legacy PyTorch loading mode")
            
            # Load model
            self.model = YOLO(model_path)
            self.model.to('cpu')
            
            # Test inference
            print("  ⏳ Testing model inference...")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, verbose=False)
            
            self.model_loaded = True
            print("  ✅ YOLOv8 model loaded successfully")
            
        except Exception as e:
            print(f"  ❌ YOLO loading failed: {e}")
            print(f"  💡 Trying alternative loading method...")
            
            # Alternative loading method for PyTorch 2.6+
            try:
                with torch.serialization.safe_globals([DetectionModel]):
                    self.model = YOLO(model_path)
                    self.model.to('cpu')
                    
                    # Test inference
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    _ = self.model(dummy, verbose=False)
                    
                    self.model_loaded = True
                    print("  ✅ YOLOv8 model loaded with alternative method")
            except Exception as e2:
                print(f"  ❌ Alternative method failed: {e2}")
                print(f"  💡 Camera will work without object detection")
                self.model_loaded = False
    
    def _capture_frames(self):
        """Optimized frame capture with enhanced processing"""
        print("  🎬 Optimized capture thread started")
        
        consecutive_errors = 0
        max_errors = 10
        
        while self.capture_running:
            try:
                if self.camera is None or not self.camera_ready:
                    frame = self._generate_placeholder("Camera not available")
                    with self.frame_lock:
                        self.frame = frame
                    time.sleep(1)
                    continue
                
                # Capture frame from picamera2
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
                
                # Convert from RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Apply rotation if needed
                rotation = getattr(config, 'CAMERA_ROTATION', 0)
                if rotation != 0:
                    if rotation == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif rotation == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Apply optimized processing
                frame = OptimizedImageProcessor.process_frame(frame, self.processing_preset)
                
                # Store frame
                with self.frame_lock:
                    self.frame = frame.copy()
                
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time + 0.001)
                self.fps_counter.append(fps)
                self.last_time = current_time
                
                # Log progress every 5 seconds
                if self.frame_count % 150 == 0:
                    avg_fps = np.mean(list(self.fps_counter))
                    print(f"  📊 Captured {self.frame_count} frames | {avg_fps:.1f} FPS")
                
                # Control frame rate
                target_fps = getattr(config, 'CAMERA_FPS', 25)
                target_delay = 1.0 / target_fps
                time.sleep(max(0.001, target_delay))
                
            except Exception as e:
                consecutive_errors += 1
                print(f"  ❌ Capture error: {e}")
                if consecutive_errors > max_errors:
                    print(f"  ⚠️  Too many errors, stopping capture")
                    self.capture_running = False
                time.sleep(0.1)
        
        print("  🛑 Optimized capture thread stopped")
    
    def _reinit_camera(self):
        """Reinitialize camera after errors"""
        try:
            if self.camera:
                self.camera.stop()
                time.sleep(0.5)
            self._init_camera_optimized()
        except Exception as e:
            print(f"  ❌ Camera reinit failed: {e}")
    
    def _yolo_detection_thread(self):
        """YOLOv8 detection thread"""
        print("  🔍 Detection thread started")
        
        if not self.model_loaded or self.model is None:
            print("  ⚠️  Detection unavailable - model not loaded")
            return
        
        last_detection = time.time()
        detection_interval = getattr(config, 'DETECTION_INTERVAL', 0.12)
        
        while self.detection_running:
            try:
                # Throttle detection rate
                current_time = time.time()
                if current_time - last_detection < detection_interval:
                    time.sleep(0.05)
                    continue
                
                last_detection = current_time
                
                # Get current frame
                with self.frame_lock:
                    if self.frame is None:
                        time.sleep(0.1)
                        continue
                    frame = self.frame.copy()
                
                # Run YOLO inference
                results = self.model(
                    frame,
                    conf=getattr(config, 'YOLO_CONFIDENCE_THRESHOLD', 0.5),
                    iou=getattr(config, 'YOLO_IOU_THRESHOLD', 0.45),
                    verbose=False,
                    device='cpu',
                    max_det=getattr(config, 'YOLO_MAX_DETECTIONS', 10)
                )
                
                # Parse results
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
                
                # Update detections
                with self.detection_lock:
                    self.detections = detections
                    self.detection_count = len(detections)
                
            except Exception as e:
                print(f"  ❌ Detection error: {e}")
                time.sleep(0.5)
        
        print("  🛑 Detection thread stopped")
    
    def _generate_placeholder(self, message="Waiting..."):
        """Generate placeholder image"""
        resolution = getattr(config, 'CAMERA_RESOLUTION', (640, 480))
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # Gradient background
        for i in range(frame.shape[0]):
            frame[i, :] = [20 + i//8, 15 + i//10, 35 + i//12]
        
        # Add text
        cv2.putText(frame, message, (100, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        cv2.putText(frame, "Camera: OV5647", (100, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 1)
        
        return frame
    
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
        if self.model_loaded:
            detection_thread = threading.Thread(
                target=self._yolo_detection_thread,
                daemon=True,
                name="YOLODetection"
            )
            detection_thread.start()
        
        print("  ✅ Enhanced detection system started")
    
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
        
        with self.detection_lock:
            detections = self.detections.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Color by confidence
            if conf > 0.8:
                color = (0, 255, 0)  # Green
            elif conf > 0.6:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(frame, (x1, y1 - label_h - 10), 
                         (x1 + label_w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add overlay
        frame = self._add_overlay(frame, len(detections))
        
        return frame
    
    def _add_overlay(self, frame, det_count):
        """Add FPS and detection overlay"""
        h, w = frame.shape[:2]
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        # FPS
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detection count
        cv2.putText(frame, f"Objects: {det_count}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        
        # Model indicator
        if self.model_loaded:
            cv2.putText(frame, "YOLOv8", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return frame
    
    def get_frame(self):
        """Get current frame without detections"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_detections(self):
        """Get current detections"""
        with self.detection_lock:
            return self.detections.copy()
    
    def get_performance_stats(self):
        """Get performance statistics"""
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        return {
            'fps': round(avg_fps, 1),
            'detections_count': self.detection_count,
            'model_loaded': self.model_loaded,
            'camera_ready': self.camera_ready,
            'frame_count': self.frame_count
        }
    
    def cleanup(self):
        """Cleanup resources"""
        print("  🧹 Cleaning up camera...")
        self.stop_detection()
        if self.camera and hasattr(self.camera, 'stop'):
            try:
                self.camera.stop()
            except:
                pass


# Create global instance
camera_instance = None

def get_camera():
    """Get or create camera instance"""
    global camera_instance
    if camera_instance is None:
        camera_instance = EnhancedCameraYOLO()
    return camera_instance