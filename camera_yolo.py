#!/usr/bin/env python3
"""
High-Performance Camera and YOLOv8 Object Detection Module
Uses OpenCV with libcamera backend for OV5647
Optimized for high FPS with proper color adjustment
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
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not available")

class EnhancedCameraYOLO:
    def __init__(self):
        """Initialize camera and YOLO for high-performance detection"""
        print("  🔷 Initializing HIGH-FPS camera system...")
        
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
        
        # Performance: Process every Nth frame for YOLO
        self.detection_skip_frames = 2  # Process every 3rd frame
        self.frame_skip_counter = 0
        
        # Color adjustment parameters
        self.brightness_adjustment = 1.2  # Brightness multiplier
        self.contrast_adjustment = 1.1    # Contrast multiplier
        
        # Initialize camera
        self._init_camera_libcamera()
        
        # Load YOLO model
        self.model = None
        self.model_loaded = False
        if YOLO_AVAILABLE:
            self._load_yolo_model()
        
        print("  ✅ Camera initialized (HIGH FPS + COLOR ADJUSTED)")
    
    def _load_yolo_model(self):
        """Load YOLOv8 nano model optimized for speed"""
        try:
            print("  🧠 Loading YOLOv8-nano (optimized)...")
            
            model_path = config.YOLO_MODEL_PATH
            
            if os.path.exists(model_path):
                print(f"     ℹ️  Using: {model_path}")
            else:
                print(f"     📥 Downloading YOLOv8n...")
                model_path = 'yolov8n.pt'
            
            # Load model
            self.model = YOLO(model_path)
            self.model.to('cpu')
            
            # Warmup with small input size for speed
            print(f"     ⏳ Warming up...")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, verbose=False, imgsz=416)
            
            self.model_loaded = True
            print("  ✅ YOLO ready (speed optimized)")
            
        except Exception as e:
            print(f"  ❌ YOLO error: {e}")
            self.model_loaded = False
    
    def _init_camera_libcamera(self):
        """Initialize camera using OpenCV with libcamera backend"""
        print("  📹 Initializing libcamera via OpenCV...")
        
        try:
            # Open camera
            self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera")
            
            # Set optimal resolution for OV5647
            # 640x480 can achieve up to 90 FPS
            width = 640
            height = 480
            target_fps = 60
            
            # Apply settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, target_fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize lag
            
            # Color and exposure settings for OV5647
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 55)     # Brightness
            self.camera.set(cv2.CAP_PROP_CONTRAST, 55)       # Contrast
            self.camera.set(cv2.CAP_PROP_SATURATION, 60)     # Color saturation
            self.camera.set(cv2.CAP_PROP_GAIN, 0)            # Auto gain
            
            # Auto white balance for better colors
            self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
            
            print(f"     🎯 Target: {width}x{height} @ {target_fps} FPS")
            print(f"     🎨 Color adjustment enabled")
            print(f"     ⏳ Warming up camera (3 seconds)...")
            
            # Quick warmup - let auto-exposure settle
            warmup_start = time.time()
            frame_count = 0
            valid_frames = 0
            
            while (time.time() - warmup_start) < 3:
                ret, frame = self.camera.read()
                frame_count += 1
                
                if ret and frame is not None and frame.size > 0:
                    brightness = frame.mean()
                    if brightness > 10:  # Valid frame
                        valid_frames += 1
                        if valid_frames >= 5:
                            break
                
                time.sleep(0.033)  # ~30 FPS during warmup
            
            # Test frame
            ret, test_frame = self.camera.read()
            if ret and test_frame is not None and test_frame.size > 0:
                h, w = test_frame.shape[:2]
                brightness = test_frame.mean()
                print(f"     ✅ Camera ready: {w}x{h}")
                print(f"     💡 Initial brightness: {brightness:.1f}")
                
                # Auto-calibrate brightness adjustment
                if brightness < 80:
                    self.brightness_adjustment = 1.3
                    print(f"     🔆 Low light detected - brightness boost enabled")
                elif brightness > 150:
                    self.brightness_adjustment = 0.9
                    print(f"     🔅 Bright conditions - brightness reduction enabled")
                
                self.camera_ready = True
            else:
                raise Exception("Failed to capture test frame")
            
            print("  ✅ libcamera initialized via OpenCV")
            self.camera_type = 'opencv'
            
        except Exception as e:
            print(f"  ❌ Camera init failed: {e}")
            print("  💡 Troubleshooting:")
            print("     1. Disable legacy camera: sudo raspi-config")
            print("     2. Enable camera interface")
            print("     3. Test: rpicam-hello -t 3000")
            print("     4. Check: ls /dev/video*")
            print("     5. Reboot if needed")
            self.camera = None
            self.camera_ready = False
    
    def _adjust_colors(self, frame):
        """Apply color adjustments for better visibility and YOLO accuracy"""
        if frame is None or frame.size == 0:
            return frame
        
        try:
            # Convert to float for processing
            adjusted = frame.astype(np.float32)
            
            # Brightness adjustment
            adjusted = adjusted * self.brightness_adjustment
            
            # Contrast adjustment (around midpoint)
            midpoint = 128.0
            adjusted = ((adjusted - midpoint) * self.contrast_adjustment) + midpoint
            
            # Clip to valid range
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
            
            # Optional: Slight saturation boost for better colors
            hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)  # Saturation +10%
            adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return adjusted
            
        except Exception as e:
            print(f"Color adjustment error: {e}")
            return frame
    
    def _capture_frames(self):
        """High-speed frame capture with color adjustment"""
        print("  🎥 HIGH-SPEED capture thread started")
        
        frame_errors = 0
        success_count = 0
        last_fps_log = time.time()
        
        while self.capture_running:
            try:
                if self.camera is None:
                    frame = self._generate_placeholder("Camera not available")
                    with self.frame_lock:
                        self.frame = frame
                    time.sleep(0.1)
                    continue
                
                # Capture frame at maximum speed
                ret, frame = self.camera.read()
                
                if not ret or frame is None or frame.size == 0:
                    frame_errors += 1
                    if frame_errors > 20:
                        print("  ⚠️  Camera stalled - reconnecting...")
                        self._reconnect_camera()
                        frame_errors = 0
                    time.sleep(0.01)
                    continue
                
                frame_errors = 0
                
                # Apply color adjustments
                frame = self._adjust_colors(frame)
                
                # Store frame
                with self.frame_lock:
                    self.frame = frame.copy()
                
                self.frame_count += 1
                success_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time + 0.0001)
                self.fps_counter.append(fps)
                self.last_time = current_time
                
                # Log FPS periodically
                if current_time - last_fps_log > 5:
                    avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
                    print(f"  ⚡ Capture FPS: {avg_fps:.1f} | Frames: {self.frame_count}")
                    last_fps_log = current_time
                
                # Tiny sleep to prevent CPU saturation
                time.sleep(0.001)
                
            except Exception as e:
                print(f"  ❌ Capture error: {e}")
                frame_errors += 1
                time.sleep(0.05)
        
        print("  🛑 Capture thread stopped")
    
    def _reconnect_camera(self):
        """Attempt to reconnect camera"""
        try:
            if self.camera:
                self.camera.release()
            time.sleep(1)
            self._init_camera_libcamera()
        except:
            pass
    
    def _generate_placeholder(self, message="Waiting..."):
        """Generate placeholder image"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Gradient background
        for i in range(frame.shape[0]):
            frame[i, :] = [25 + i//10, 20 + i//12, 40 + i//10]
        
        cv2.putText(frame, message, (120, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)
        cv2.putText(frame, "libcamera + OpenCV", (140, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        return frame
    
    def _yolo_detection_thread(self):
        """Optimized YOLO detection with frame skipping"""
        print("  🔍 YOLO detection thread started")
        print(f"     💡 Processing every {self.detection_skip_frames + 1} frames")
        
        if not self.model_loaded or self.model is None:
            print("  ⚠️  YOLO unavailable")
            self.detection_running = False
            return
        
        detection_count = 0
        last_log = time.time()
        
        while self.detection_running:
            try:
                if self.frame is None:
                    time.sleep(0.02)
                    continue
                
                # Get frame
                with self.frame_lock:
                    frame = self.frame.copy()
                
                # Frame skipping for performance
                self.frame_skip_counter += 1
                if self.frame_skip_counter <= self.detection_skip_frames:
                    time.sleep(0.01)
                    continue
                
                self.frame_skip_counter = 0
                
                # YOLO inference with speed optimization
                results = self.model(
                    frame,
                    conf=config.YOLO_CONFIDENCE_THRESHOLD,
                    iou=config.YOLO_IOU_THRESHOLD,
                    verbose=False,
                    device='cpu',
                    imgsz=416,      # Smaller = faster (vs 640)
                    half=False,      # No FP16 on CPU
                    agnostic_nms=False,
                    max_det=50       # Limit detections
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
                
                detection_count += 1
                
                # Log detection stats
                current_time = time.time()
                if current_time - last_log > 5:
                    det_fps = detection_count / (current_time - last_log)
                    print(f"  🎯 YOLO FPS: {det_fps:.1f} | Objects: {len(detections)}")
                    detection_count = 0
                    last_log = current_time
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"  ❌ Detection error: {e}")
                time.sleep(0.1)
        
        print("  🛑 Detection thread stopped")
    
    def start_detection(self):
        """Start capture and detection"""
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
        
        print("  ✅ Detection started (HIGH FPS mode)")
    
    def stop_detection(self):
        """Stop all threads"""
        self.detection_running = False
        self.capture_running = False
        time.sleep(0.5)
        self.is_running = False
    
    def get_frame_with_detections(self):
        """Get frame with bounding boxes and labels"""
        with self.frame_lock:
            if self.frame is None:
                return self._generate_placeholder("Starting...")
            frame = self.frame.copy()
        
        if frame is None or frame.size == 0:
            return self._generate_placeholder("No frame")
        
        with self.detection_lock:
            detections = self.detections.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Color by confidence
            if conf > 0.8:
                color = (0, 255, 0)      # Green
            elif conf > 0.6:
                color = (0, 200, 255)    # Orange
            else:
                color = (0, 100, 255)    # Red
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label background
            label = f"{class_name} {int(conf*100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Draw label background
            cv2.rectangle(frame, 
                         (x1, y1 - text_size[1] - 8),
                         (x1 + text_size[0] + 8, y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 4, y1 - 4),
                       font, font_scale, (0, 0, 0), thickness + 1)
        
        # Add overlay info
        frame = self._add_overlay(frame, len(detections))
        
        return frame
    
    def _add_overlay(self, frame, det_count):
        """Add FPS and stats overlay"""
        h, w = frame.shape[:2]
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        # Semi-transparent background for text
        overlay = frame.copy()
        
        # FPS (top-left)
        cv2.rectangle(overlay, (5, 5), (140, 75), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {det_count}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(frame, f"Res: {w}x{h}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # YOLO indicator (top-right)
        cv2.putText(frame, "YOLOv8n", (w - 100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return frame
    
    def get_frame(self):
        """Get raw frame without detections"""
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
            'camera_ready': self.camera_ready
        }
    
    def cleanup(self):
        """Cleanup resources"""
        print("  🧹 Cleaning up camera...")
        self.stop_detection()
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
        print("  ✅ Camera cleanup complete")