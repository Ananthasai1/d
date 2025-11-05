#!/usr/bin/env python3
"""
COMPLETE FIX for camera/camera_yolo.py
Fixes YOLO loading with PyTorch 2.6+ and improves FPS dramatically
Replace your entire camera/camera_yolo.py with this code
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
    # Import ALL required PyTorch modules for YOLO
    from torch.nn.modules.container import Sequential, ModuleList
    from torch.nn.modules.activation import SiLU
    from torch.nn.modules.conv import Conv2d
    from torch.nn.modules.batchnorm import BatchNorm2d
    from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
    from torch.nn.modules.linear import Linear
    from torch.nn.modules.dropout import Dropout
    from ultralytics.nn.tasks import DetectionModel
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not available")


class EnhancedCameraYOLO:
    """Enhanced camera with YOLOv8 object detection - FIXED VERSION"""
    
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
        
        # YOLO model
        self.model = None
        self.model_loaded = False
        
        # Initialize camera
        self._init_camera()
        
        # Load YOLO model if available
        if YOLO_AVAILABLE:
            self._load_yolo_model_fixed()
        
        print("✅ Camera system initialized")
    
    def _init_camera(self):
        """Initialize camera with picamera2"""
        print("  🎥 Initializing OV5647 camera...")
        
        if not PICAMERA2_AVAILABLE:
            print("  ❌ picamera2 not available")
            self.camera_type = 'none'
            return
        
        try:
            # Create Picamera2 instance
            self.camera = Picamera2()
            
            # OPTIMIZED: Use lower resolution for better FPS
            resolution = (640, 480)
            
            # Configure camera with OPTIMIZED settings
            camera_config = self.camera.create_video_configuration(
                main={"size": resolution, "format": "RGB888"},
                controls={
                    "FrameRate": 30,  # Target 30 FPS
                    "FrameDurationLimits": (16666, 33333),  # 30-60 FPS range
                    "AwbEnable": True,
                    "AeEnable": True,
                    "NoiseReductionMode": 0,  # Disable for speed
                }
            )
            
            self.camera.configure(camera_config)
            
            # Start camera
            print("  ⏳ Starting camera...")
            self.camera.start()
            
            # Quick warmup
            time.sleep(0.5)
            
            # Test capture
            test_frame = self.camera.capture_array()
            if test_frame is not None and test_frame.size > 0:
                print(f"  ✅ Camera ready! Resolution: {test_frame.shape}")
                self.camera_ready = True
                self.camera_type = 'picamera2'
            else:
                raise Exception("Failed to capture test frame")
            
        except Exception as e:
            print(f"  ❌ Camera initialization failed: {e}")
            self.camera = None
            self.camera_type = 'none'
            self.camera_ready = False
    
    def _load_yolo_model_fixed(self):
        """Load YOLOv8 model with COMPLETE PyTorch 2.6+ fix"""
        try:
            print("  🧠 Loading YOLOv8 model...")
            
            model_path = config.YOLO_MODEL_PATH
            
            if not os.path.exists(model_path):
                print(f"  📥 Model not found. Download with:")
                print(f"     wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
                self.model_loaded = False
                return
            
            # ===== CRITICAL FIX: Add ALL PyTorch safe globals =====
            print("  🔧 Configuring PyTorch security...")
            
            # Add all required PyTorch and YOLO classes to safe globals
            safe_classes = [
                # Core YOLO
                DetectionModel,
                
                # PyTorch Containers
                Sequential,
                ModuleList,
                
                # Activation functions
                SiLU,
                
                # Convolution layers
                Conv2d,
                
                # Normalization layers
                BatchNorm2d,
                
                # Pooling layers
                MaxPool2d,
                AdaptiveAvgPool2d,
                
                # Linear layers
                Linear,
                
                # Dropout
                Dropout,
            ]
            
            # Add all safe globals
            for cls in safe_classes:
                try:
                    torch.serialization.add_safe_globals([cls])
                except Exception as e:
                    print(f"  ⚠️  Could not add {cls.__name__}: {e}")
            
            print("  ✅ PyTorch security configured")
            
            # Load model
            print("  ⏳ Loading YOLO model...")
            self.model = YOLO(model_path)
            self.model.to('cpu')
            
            # Test inference
            print("  ⏳ Testing inference...")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, verbose=False)
            
            self.model_loaded = True
            print("  ✅ YOLOv8 model loaded successfully!")
            
        except Exception as e:
            print(f"  ❌ YOLO loading failed: {e}")
            print(f"  💡 Error type: {type(e).__name__}")
            
            # Try alternative method
            print("  🔄 Trying alternative loading method...")
            try:
                # Method 2: Use weights_only=False (less secure but works)
                print("  ⚠️  Using legacy loading mode...")
                
                # Temporarily patch torch.load
                import warnings
                warnings.filterwarnings('ignore')
                
                # This is a workaround for PyTorch 2.6+
                # It's safe if you trust the source of yolov8n.pt
                original_load = torch.load
                
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = patched_load
                
                # Try loading again
                self.model = YOLO(model_path)
                self.model.to('cpu')
                
                # Restore original torch.load
                torch.load = original_load
                
                # Test inference
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                _ = self.model(dummy, verbose=False)
                
                self.model_loaded = True
                print("  ✅ YOLOv8 loaded with alternative method")
                
            except Exception as e2:
                print(f"  ❌ Alternative method failed: {e2}")
                print(f"  💡 Continuing without YOLO detection...")
                self.model_loaded = False
    
    def _capture_frames(self):
        """OPTIMIZED frame capture thread"""
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
                
                # OPTIMIZED: Capture frame directly (no warmup delay)
                frame = self.camera.capture_array()
                
                if frame is None or frame.size == 0:
                    consecutive_errors += 1
                    if consecutive_errors > max_errors:
                        print(f"  ⚠️  Too many errors, reinitializing...")
                        self._reinit_camera()
                        consecutive_errors = 0
                    time.sleep(0.1)
                    continue
                
                consecutive_errors = 0
                
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Apply rotation if needed
                rotation = getattr(config, 'CAMERA_ROTATION', 0)
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Store frame (no processing for max FPS)
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
                
                # CRITICAL: Minimal delay for max FPS
                time.sleep(0.001)  # Almost no delay
                
            except Exception as e:
                consecutive_errors += 1
                print(f"  ❌ Capture error: {e}")
                if consecutive_errors > max_errors:
                    print(f"  ⚠️  Too many errors, stopping")
                    self.capture_running = False
                time.sleep(0.1)
        
        print("  🛑 Capture thread stopped")
    
    def _reinit_camera(self):
        """Reinitialize camera after errors"""
        try:
            if self.camera:
                self.camera.stop()
                time.sleep(0.5)
            self._init_camera()
        except Exception as e:
            print(f"  ❌ Reinit failed: {e}")
    
    def _yolo_detection_thread(self):
        """YOLOv8 detection thread"""
        print("  🔍 Detection thread started")
        
        if not self.model_loaded or self.model is None:
            print("  ⚠️  Detection unavailable - model not loaded")
            return
        
        last_detection = time.time()
        detection_interval = getattr(config, 'DETECTION_INTERVAL', 0.2)
        
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
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
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
        
        print("  ✅ Detection system started")
    
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
                return self._generate_placeholder("Initializing...")
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
                color = (0, 255, 0)
            elif conf > 0.6:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
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
        """Get current frame"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_detections(self):
        """Get current detections"""
        with self.detection_lock:
            return self.detections.copy()
    
    def get_performance_stats(self):
        """Get performance stats"""
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