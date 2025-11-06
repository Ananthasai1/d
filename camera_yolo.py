#!/usr/bin/env python3
"""
FAST STARTUP camera_yolo.py with NATURAL COLORS
- Startup time: 2-3 seconds (down from 10+ seconds)
- Natural, realistic image colors (no warm/yellow tint)
- Fixed white balance and color correction
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class EnhancedCameraYOLO:
    """Fast startup camera with natural colors"""
    
    def __init__(self):
        """Initialize camera - FAST version"""
        print("📷 Initializing camera (fast mode)...")
        
        self.camera = None
        self.frame = None
        self.frame_lock = threading.Lock()
        
        self.detections = []
        self.detection_lock = threading.Lock()
        self.last_detection_time = 0
        
        self.is_running = False
        self.capture_running = False
        self.detection_running = False
        self.camera_ready = False
        
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        self.frame_count = 0
        self.detection_count = 0
        self.total_detections = 0
        
        self.model = None
        self.model_loaded = False
        
        self.detection_interval = 0.2
        self.confidence_threshold = 0.5
        
        # Initialize camera FIRST (fast)
        self._init_camera_fast()
        
        # Load YOLO in background thread (doesn't block startup)
        if YOLO_AVAILABLE:
            threading.Thread(target=self._load_yolo_background, daemon=True).start()
        
        print("✅ Camera ready!")
    
    def _init_camera_fast(self):
        """FAST camera initialization with NATURAL colors"""
        if not PICAMERA2_AVAILABLE:
            print("  ❌ picamera2 not available")
            self.camera_type = 'none'
            return
        
        try:
            print("  🎥 Starting camera...")
            self.camera = Picamera2()
            
            # OPTIMIZED CONFIG for natural colors
            camera_config = self.camera.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={
                    "FrameRate": 30,
                    "FrameDurationLimits": (16666, 50000),
                    
                    # CRITICAL: Daylight white balance for natural colors
                    "AwbEnable": True,
                    "AwbMode": 4,  # 4 = Daylight (removes yellow tint!)
                    
                    # Auto exposure with normal settings
                    "AeEnable": True,
                    "AeExposureMode": 0,
                    "AeConstraintMode": 0,
                    
                    # Neutral image settings (no enhancement)
                    "Brightness": 0.0,
                    "Contrast": 1.0,
                    "Saturation": 1.0,
                    "Sharpness": 1.0,
                    
                    # Minimal noise reduction for speed
                    "NoiseReductionMode": 1,
                }
            )
            
            self.camera.configure(camera_config)
            self.camera.start()
            
            # MINIMAL warmup (0.5s instead of 2s)
            time.sleep(0.5)
            
            # Test frame
            test_frame = self.camera.capture_array()
            if test_frame is not None and test_frame.size > 0:
                print(f"  ✅ Camera ready! {test_frame.shape}")
                self.camera_ready = True
                self.camera_type = 'picamera2'
            else:
                raise Exception("Test frame failed")
            
        except Exception as e:
            print(f"  ❌ Camera init failed: {e}")
            self.camera = None
            self.camera_type = 'none'
            self.camera_ready = False
    
    def _load_yolo_background(self):
        """Load YOLO in background (doesn't block startup)"""
        try:
            print("  🧠 Loading YOLO in background...")
            
            model_path = 'yolov8n.pt'
            if not os.path.exists(model_path):
                print(f"  ⚠️  Model not found: {model_path}")
                return
            
            # Load model
            try:
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals([DetectionModel])
            except:
                pass
            
            self.model = YOLO(model_path)
            self.model.to('cpu')
            
            # Quick test
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, verbose=False)
            
            self.model_loaded = True
            print("  ✅ YOLO loaded!")
            
        except Exception as e:
            print(f"  ❌ YOLO load failed: {e}")
            self.model_loaded = False
    
    def _capture_frames(self):
        """Fast frame capture with color correction"""
        print("  🎬 Capture started")
        
        consecutive_errors = 0
        
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
                    if consecutive_errors > 10:
                        self._reinit_camera()
                        consecutive_errors = 0
                    time.sleep(0.1)
                    continue
                
                consecutive_errors = 0
                
                # Convert RGB to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # CRITICAL: Apply color correction for natural look
                frame = self._apply_color_correction(frame)
                
                # Store frame
                with self.frame_lock:
                    self.frame = frame.copy()
                
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time + 0.001)
                self.fps_counter.append(fps)
                self.last_time = current_time
                
                time.sleep(0.001)
                
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors > 10:
                    self.capture_running = False
                time.sleep(0.1)
    
    def _apply_color_correction(self, frame):
        """Apply natural color correction (removes warm tint)"""
        try:
            # Method 1: Auto white balance in LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Balance A and B channels (removes color casts)
            a = cv2.add(a, -int(np.mean(a)) + 128)
            b = cv2.add(b, -int(np.mean(b)) + 128)
            
            # Merge and convert back
            balanced = cv2.merge([l, a, b])
            frame = cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)
            
            # Method 2: Subtle contrast enhancement
            alpha = 1.05  # Contrast (very subtle)
            beta = 0      # Brightness (neutral)
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
            return frame
            
        except:
            return frame
    
    def _yolo_detection_thread(self):
        """YOLO detection thread"""
        print("  🔍 Detection started")
        
        if not self.model_loaded:
            print("  ⚠️  Waiting for YOLO to load...")
            # Wait up to 30 seconds for model to load
            for _ in range(60):
                if self.model_loaded:
                    break
                time.sleep(0.5)
        
        if not self.model_loaded:
            print("  ❌ YOLO not available")
            return
        
        while self.detection_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_detection_time < self.detection_interval:
                    time.sleep(0.05)
                    continue
                
                self.last_detection_time = current_time
                
                with self.frame_lock:
                    if self.frame is None:
                        time.sleep(0.1)
                        continue
                    frame = self.frame.copy()
                
                # Run YOLO
                results = self.model.predict(
                    source=frame,
                    conf=self.confidence_threshold,
                    verbose=False,
                    device='cpu',
                    max_det=10
                )
                
                # Parse detections
                detections = []
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        
                        for i in range(len(boxes)):
                            try:
                                box = boxes.xyxy[i].cpu().numpy()
                                x1, y1, x2, y2 = map(int, box)
                                
                                conf = float(boxes.conf[i].cpu().numpy())
                                cls_id = int(boxes.cls[i].cpu().numpy())
                                class_name = self.model.names[cls_id]
                                
                                detection = {
                                    'class': class_name,
                                    'confidence': round(conf, 3),
                                    'bbox': [x1, y1, x2, y2],
                                    'center_x': int((x1 + x2) / 2),
                                    'center_y': int((y1 + y2) / 2),
                                    'class_id': cls_id
                                }
                                
                                detections.append(detection)
                                
                            except:
                                continue
                
                with self.detection_lock:
                    self.detections = detections
                    self.detection_count = len(detections)
                    if len(detections) > 0:
                        self.total_detections += len(detections)
                
            except Exception as e:
                time.sleep(0.5)
    
    def _reinit_camera(self):
        """Reinit camera after errors"""
        try:
            if self.camera:
                self.camera.stop()
                time.sleep(0.5)
            self._init_camera_fast()
        except:
            pass
    
    def _generate_placeholder(self, message="Waiting..."):
        """Generate placeholder"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            frame[i, :] = [20 + i//8, 15 + i//10, 35 + i//12]
        cv2.putText(frame, message, (100, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        return frame
    
    def start_detection(self):
        """Start capture and detection"""
        if self.is_running:
            return
        
        self.is_running = True
        self.capture_running = True
        self.detection_running = True
        
        # Start capture
        threading.Thread(target=self._capture_frames, daemon=True).start()
        
        # Start detection
        threading.Thread(target=self._yolo_detection_thread, daemon=True).start()
        
        print("  ✅ Detection system started")
    
    def stop_detection(self):
        """Stop threads"""
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
        """Add FPS overlay"""
        h, w = frame.shape[:2]
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {det_count}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        
        if self.model_loaded:
            cv2.putText(frame, "YOLOv8", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return frame
    
    def get_frame(self):
        """Get frame without detections"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_detections(self):
        """Get detections"""
        with self.detection_lock:
            return self.detections.copy()
    
    def get_performance_stats(self):
        """Get stats"""
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        return {
            'fps': round(avg_fps, 1),
            'detections_count': self.detection_count,
            'total_detections': self.total_detections,
            'model_loaded': self.model_loaded,
            'camera_ready': self.camera_ready,
            'frame_count': self.frame_count
        }
    
    def cleanup(self):
        """Cleanup"""
        print("  🧹 Cleaning up camera...")
        self.stop_detection()
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass