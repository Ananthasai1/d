#!/usr/bin/env python3
"""
Enhanced Camera with Superior Quality and Colors
Includes image processing, color grading, and optimized camera settings
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
import os
import sys

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
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not available")


class ImageEnhancer:
    """Real-time image enhancement"""
    
    @staticmethod
    def enhance(frame):
        """Apply all enhancements"""
        # 1. Auto white balance
        frame = ImageEnhancer.auto_white_balance(frame)
        
        # 2. Brightness and contrast
        frame = ImageEnhancer.adjust_brightness_contrast(frame, 
                                                         brightness=15, 
                                                         contrast=20)
        
        # 3. Boost saturation
        frame = ImageEnhancer.boost_saturation(frame, 1.3)
        
        # 4. Sharpen
        frame = ImageEnhancer.sharpen(frame)
        
        return frame
    
    @staticmethod
    def auto_white_balance(frame):
        """Improve color accuracy"""
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def adjust_brightness_contrast(frame, brightness=0, contrast=0):
        """Adjust brightness and contrast"""
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
    def boost_saturation(frame, factor=1.3):
        """Increase color vibrancy"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def sharpen(frame):
        """Sharpen image for better clarity"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(frame, -1, kernel)


class EnhancedCameraYOLO:
    """Enhanced camera with superior quality and YOLO detection"""
    
    def __init__(self):
        """Initialize with enhanced settings"""
        print("📷 Initializing enhanced camera system...")
        
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
        self.detection_count = 0
        
        self.model = None
        self.model_loaded = False
        
        # Enhancement settings
        self.enable_enhancement = True
        
        self._init_camera()
        
        if YOLO_AVAILABLE:
            self._load_yolo_model()
        
        print("✅ Enhanced camera system initialized")
    
    def _init_camera(self):
        """Initialize camera with optimized settings"""
        print("  🎥 Initializing OV5647 camera with enhanced settings...")
        
        if not PICAMERA2_AVAILABLE:
            print("  ❌ picamera2 not available")
            self.camera_type = 'none'
            return
        
        try:
            self.camera = Picamera2()
            
            # Enhanced camera configuration
            camera_config = self.camera.create_still_configuration(
                main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"},
                controls={
                    "FrameRate": config.CAMERA_FPS,
                    
                    # Enhanced controls for better image quality
                    "AwbEnable": True,
                    "AwbMode": 0,  # Auto white balance
                    
                    "AeEnable": True,
                    "AeExposureMode": 0,  # Normal exposure
                    "AeConstraintMode": 0,
                    
                    # Improved settings
                    "Brightness": 0.1,      # Slight brightness boost
                    "Contrast": 1.2,        # Enhanced contrast
                    "Saturation": 1.3,      # More vibrant colors
                    "Sharpness": 2.0,       # Better sharpness
                    
                    # Noise reduction
                    "NoiseReductionMode": 1,  # Fast NR
                }
            )
            
            self.camera.configure(camera_config)
            
            print("  ⏳ Starting camera...")
            self.camera.start()
            
            print(f"  ⏳ Warming up camera ({config.CAMERA_WARMUP_TIME}s)...")
            time.sleep(config.CAMERA_WARMUP_TIME)
            
            test_frame = self.camera.capture_array()
            if test_frame is not None and test_frame.size > 0:
                print(f"  ✅ Enhanced camera ready! Resolution: {test_frame.shape}")
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
            
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            try:
                self.model = YOLO(model_path)
            except Exception as e1:
                print(f"  ⚠️  Standard loading failed, trying alternative...")
                try:
                    import torch
                    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
                    self.model = YOLO(model_path)
                except Exception as e2:
                    print(f"  ⚠️  Trying final workaround...")
                    import torch.serialization
                    original_load = torch.load
                    
                    def patched_load(*args, **kwargs):
                        kwargs['weights_only'] = False
                        return original_load(*args, **kwargs)
                    
                    torch.load = patched_load
                    try:
                        self.model = YOLO(model_path)
                    finally:
                        torch.load = original_load
            
            self.model.to('cpu')
            
            print("  ⏳ Testing model inference...")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, verbose=False)
            
            self.model_loaded = True
            print("  ✅ YOLOv8 model loaded successfully!")
            
        except Exception as e:
            print(f"  ❌ YOLO loading failed: {e}")
            self.model_loaded = False
    
    def _capture_frames(self):
        """Enhanced frame capture with image processing"""
        print("  🎬 Enhanced capture thread started")
        
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
                if config.CAMERA_ROTATION == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif config.CAMERA_ROTATION == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif config.CAMERA_ROTATION == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # ✨ APPLY ENHANCEMENTS ✨
                if self.enable_enhancement:
                    frame = ImageEnhancer.enhance(frame)
                
                # Store frame
                with self.frame_lock:
                    self.frame = frame.copy()
                
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time + 0.001)
                self.fps_counter.append(fps)
                self.last_time = current_time
                
                if self.frame_count % 150 == 0:
                    avg_fps = np.mean(list(self.fps_counter))
                    print(f"  📊 Captured {self.frame_count} frames | {avg_fps:.1f} FPS")
                
                time.sleep(1.0 / config.CAMERA_FPS)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"  ❌ Capture error: {e}")
                if consecutive_errors > max_errors:
                    self.capture_running = False
                time.sleep(0.1)
        
        print("  🛑 Capture thread stopped")
    
    def _reinit_camera(self):
        """Reinitialize camera"""
        try:
            if self.camera:
                self.camera.stop()
                time.sleep(0.5)
            self._init_camera()
        except Exception as e:
            print(f"  ❌ Camera reinit failed: {e}")
    
    def _yolo_detection_thread(self):
        """YOLO detection thread"""
        print("  🔍 Detection thread started")
        
        if not self.model_loaded or self.model is None:
            print("  ⚠️  Detection unavailable")
            return
        
        last_detection = time.time()
        
        while self.detection_running:
            try:
                current_time = time.time()
                if current_time - last_detection < config.DETECTION_INTERVAL:
                    time.sleep(0.05)
                    continue
                
                last_detection = current_time
                
                with self.frame_lock:
                    if self.frame is None:
                        time.sleep(0.1)
                        continue
                    frame = self.frame.copy()
                
                results = self.model(
                    frame,
                    conf=config.YOLO_CONFIDENCE_THRESHOLD,
                    iou=config.YOLO_IOU_THRESHOLD,
                    verbose=False,
                    device='cpu',
                    max_det=config.YOLO_MAX_DETECTIONS
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
                    self.detection_count = len(detections)
                
            except Exception as e:
                print(f"  ❌ Detection error: {e}")
                time.sleep(0.5)
        
        print("  🛑 Detection thread stopped")
    
    def _generate_placeholder(self, message="Waiting..."):
        """Generate placeholder image"""
        frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                         config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
        
        for i in range(frame.shape[0]):
            frame[i, :] = [20 + i//8, 15 + i//10, 35 + i//12]
        
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
        
        capture_thread = threading.Thread(
            target=self._capture_frames,
            daemon=True,
            name="CameraCapture"
        )
        capture_thread.start()
        
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
                return self._generate_placeholder("Initializing camera...")
            frame = self.frame.copy()
        
        with self.detection_lock:
            detections = self.detections.copy()
        
        # Draw detections with enhanced visuals
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Color by confidence
            if conf > 0.8:
                color = (0, 255, 100)  # Bright green
            elif conf > 0.6:
                color = (0, 200, 255)  # Orange
            else:
                color = (0, 100, 255)  # Red
            
            # Draw thicker box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with better styling
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Background for label
            cv2.rectangle(frame, (x1, y1 - label_h - 15), 
                         (x1 + label_w + 15, y1), color, -1)
            
            # Add text
            cv2.putText(frame, label, (x1 + 7, y1 - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        frame = self._add_overlay(frame, len(detections))
        
        return frame
    
    def _add_overlay(self, frame, det_count):
        """Add FPS and detection overlay"""
        h, w = frame.shape[:2]
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        # Enhanced overlay styling
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
        
        cv2.putText(frame, f"Objects: {det_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        
        if self.model_loaded:
            cv2.putText(frame, "YOLOv8", (w - 150, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)
        
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