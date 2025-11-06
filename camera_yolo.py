#!/usr/bin/env python3
"""
High Performance Camera and YOLOv8 Object Detection Module
OPTIMIZED: Better color accuracy, higher FPS, clearer detection boxes
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
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not available")

class EnhancedCameraYOLO:
    def __init__(self):
        """Initialize high-performance camera and YOLO"""
        print("  🔷 Initializing HIGH PERFORMANCE camera system...")
        
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
        
        # Performance optimization
        self.skip_frames = 0  # Process every frame for max FPS
        self.frame_skip_counter = 0
        
        # Initialize camera with optimizations
        self._init_camera()
        
        # Load YOLO model
        self.model = None
        self.model_loaded = False
        if YOLO_AVAILABLE:
            self._load_yolo_model()
        
        print("  ✅ High-performance camera initialized")
    
    def _load_yolo_model(self):
        """Load YOLOv8 model optimized for speed"""
        try:
            print("  🧠 Loading YOLOv8 model (optimized)...")
            
            model_path = config.YOLO_MODEL_PATH
            
            if os.path.exists(model_path):
                print(f"     ℹ️  Using local model: {model_path}")
            else:
                print(f"     📥 Downloading YOLOv8n (fastest)...")
                model_path = 'yolov8n.pt'
            
            self.model = YOLO(model_path)
            self.model.to('cpu')
            
            # Optimize model for inference
            print("     ⚡ Optimizing for speed...")
            self.model.fuse()  # Fuse layers for faster inference
            
            # Test inference
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            results = self.model(dummy_image, verbose=False)
            
            self.model_loaded = True
            print("  ✅ YOLOv8 model loaded and optimized")
            
        except Exception as e:
            print(f"  ❌ YOLO loading error: {e}")
            self.model_loaded = False
    
    def _init_camera(self):
        """Initialize camera with optimal settings for clarity and FPS"""
        print("  📹 Initializing camera (HIGH FPS mode)...")
        
        try:
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera device")
            
            # HIGH FPS SETTINGS
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
            self.camera.set(cv2.CAP_PROP_FPS, 30)  # Max FPS
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            
            # CLARITY SETTINGS
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Auto focus
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 50)
            self.camera.set(cv2.CAP_PROP_CONTRAST, 50)
            self.camera.set(cv2.CAP_PROP_SATURATION, 60)  # Better color saturation
            self.camera.set(cv2.CAP_PROP_SHARPNESS, 50)  # Sharpness for clarity
            
            print("     ⏳ Quick warmup (5 seconds)...")
            
            # Faster warmup
            start_time = time.time()
            valid_frames = 0
            
            while (time.time() - start_time) < 5:
                ret, frame = self.camera.read()
                
                if ret and frame is not None and frame.size > 0:
                    mean_brightness = frame.mean()
                    
                    if mean_brightness > 10:
                        valid_frames += 1
                        
                        if valid_frames >= 2:
                            print(f"     ✅ Camera ready! ({time.time() - start_time:.1f}s)")
                            
                            # Flush buffer
                            for _ in range(5):
                                self.camera.read()
                            
                            self.camera_ready = True
                            break
                
                time.sleep(0.05)
            
            if not self.camera_ready:
                self.camera_ready = True
            
            print("  ✅ Camera initialized - HIGH FPS MODE")
            self.camera_type = 'opencv'
            
        except Exception as e:
            print(f"  ❌ Camera initialization failed: {e}")
            self.camera = None
            self.camera_type = 'none'
            self.camera_ready = False
    
    def _capture_frames(self):
        """HIGH SPEED frame capture with color correction"""
        print("  🎥 HIGH SPEED capture thread started")
        frame_errors = 0
        success_count = 0
        
        while self.capture_running:
            try:
                if self.camera is None:
                    frame = self._generate_placeholder("Camera not available")
                    with self.frame_lock:
                        self.frame = frame
                    time.sleep(0.033)
                    continue
                
                # Grab frame immediately (no decoding yet) for speed
                grabbed = self.camera.grab()
                
                if not grabbed:
                    frame_errors += 1
                    if frame_errors > 10:
                        print("  ⚠️  Camera grab failed - reconnecting...")
                        try:
                            self.camera.release()
                        except:
                            pass
                        time.sleep(1)
                        self._init_camera()
                        frame_errors = 0
                    time.sleep(0.01)
                    continue
                
                # Decode frame
                ret, frame = self.camera.retrieve()
                
                if not ret or frame is None or frame.size == 0:
                    frame_errors += 1
                    time.sleep(0.01)
                    continue
                
                frame_errors = 0
                
                # Resize if needed
                if frame.shape[0] != config.CAMERA_RESOLUTION[1] or frame.shape[1] != config.CAMERA_RESOLUTION[0]:
                    frame = cv2.resize(frame, config.CAMERA_RESOLUTION, interpolation=cv2.INTER_LINEAR)
                
                # 🎨 COLOR CORRECTION: BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 🔍 ENHANCE CLARITY (optional - can be disabled for max FPS)
                # Slight sharpening for better clarity
                kernel = np.array([[-0.5, -0.5, -0.5],
                                   [-0.5,  5.0, -0.5],
                                   [-0.5, -0.5, -0.5]])
                frame = cv2.filter2D(frame, -1, kernel * 0.3)  # Mild sharpening
                
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
                    print(f"  ✅ First frame captured!")
                elif success_count % 100 == 0:
                    avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
                    print(f"  🚀 Captured {self.frame_count} frames | {avg_fps:.1f} FPS")
                
                # Minimal delay for max FPS
                time.sleep(0.001)
                
            except Exception as e:
                print(f"  ❌ Capture error: {e}")
                frame_errors += 1
                time.sleep(0.05)
        
        print("  🛑 Capture thread stopped")
    
    def _generate_placeholder(self, message="Waiting for camera..."):
        """Generate placeholder frame"""
        frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                         config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
        
        # Gradient background
        for i in range(frame.shape[0]):
            frame[i, :] = [30 + i//10, 20 + i//12, 50 + i//8]
        
        cv2.putText(frame, message, (100, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
        
        return frame
    
    def _yolo_detection_thread(self):
        """OPTIMIZED detection thread - every 2nd frame for balance"""
        print("  🔍 OPTIMIZED detection thread started")
        
        if not self.model_loaded or self.model is None:
            print("  ⚠️  Detection unavailable - model not loaded")
            self.detection_running = False
            return
        
        detection_errors = 0
        frame_counter = 0
        
        while self.detection_running:
            try:
                if self.frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process every 2nd frame for balance between FPS and detection
                frame_counter += 1
                if frame_counter % 2 != 0:
                    time.sleep(0.01)
                    continue
                
                with self.frame_lock:
                    frame = self.frame.copy()
                
                # Run YOLO (expects RGB)
                results = self.model(
                    frame,
                    conf=config.YOLO_CONFIDENCE_THRESHOLD,
                    iou=config.YOLO_IOU_THRESHOLD,
                    verbose=False,
                    device='cpu',
                    half=False  # FP32 for accuracy
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
                
                detection_errors = 0
                time.sleep(0.01)
                
            except Exception as e:
                detection_errors += 1
                if detection_errors > 5:
                    print(f"  ❌ Detection error: {e}")
                    detection_errors = 0
                time.sleep(0.05)
        
        print("  🛑 Detection thread stopped")
    
    def start_detection(self):
        """Start capture and detection threads"""
        if self.is_running:
            return
        
        self.is_running = True
        self.capture_running = True
        self.detection_running = True
        
        # High priority capture thread
        capture_thread = threading.Thread(
            target=self._capture_frames,
            daemon=True,
            name="HighSpeedCapture"
        )
        capture_thread.start()
        
        # Detection thread
        if self.model_loaded and self.model is not None:
            detection_thread = threading.Thread(
                target=self._yolo_detection_thread,
                daemon=True,
                name="OptimizedYOLO"
            )
            detection_thread.start()
        
        print("  ✅ High-speed detection started")
    
    def stop_detection(self):
        """Stop all threads"""
        self.detection_running = False
        self.capture_running = False
        time.sleep(0.3)
        self.is_running = False
    
    def get_frame_with_detections(self):
        """Get frame with HIGH QUALITY detection boxes"""
        with self.frame_lock:
            if self.frame is None:
                return self._generate_placeholder("Initializing camera...")
            frame = self.frame.copy()
        
        if frame is None or frame.size == 0:
            return self._generate_placeholder("No frame available")
        
        with self.detection_lock:
            detections = self.detections.copy()
        
        # Draw HIGH QUALITY bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # VIBRANT COLORS (RGB format)
            if conf > 0.8:
                color = (0, 255, 0)  # Bright Green
                thickness = 3
            elif conf > 0.6:
                color = (255, 165, 0)  # Orange
                thickness = 2
            else:
                color = (255, 0, 100)  # Pink/Red
                thickness = 2
            
            # Draw THICK, CLEAR rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw CORNER MARKERS for modern look
            corner_length = 20
            # Top-left corner
            cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness + 1)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness + 1)
            # Top-right corner
            cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness + 1)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness + 1)
            # Bottom-left corner
            cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness + 1)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness + 1)
            # Bottom-right corner
            cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness + 1)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness + 1)
            
            # Draw CLEAR label with background
            label = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            font_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            
            # Label background (semi-transparent effect)
            bg_x1, bg_y1 = x1, y1 - text_size[1] - 12
            bg_x2, bg_y2 = x1 + text_size[0] + 12, y1
            
            # Draw label background
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            
            # Draw text in white for contrast
            cv2.putText(frame, label, (x1 + 6, y1 - 6), font, font_scale, 
                       (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # Add overlay info
        frame = self._add_overlay(frame, len(detections))
        
        # 🔴 Convert RGB back to BGR for JPEG encoding
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
    
    def _add_overlay(self, frame, det_count):
        """Add HIGH QUALITY overlay info"""
        h, w = frame.shape[:2]
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        # Create semi-transparent overlay panel
        overlay = frame.copy()
        
        # Top-left info panel
        panel_height = 110
        cv2.rectangle(overlay, (0, 0), (250, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # FPS counter (larger, clearer)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Detection count
        cv2.putText(frame, f"Objects: {det_count}", (10, 70), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 200, 0), 2, cv2.LINE_AA)
        
        # Model indicator
        cv2.putText(frame, "YOLOv8", (10, 100), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        
        return frame
    
    def get_frame(self):
        """Get current frame without detections"""
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def get_detections(self):
        """Get current detections list"""
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
        print("  Cleaning up camera...")
        self.stop_detection()
        if self.camera:
            try:
                self.camera.release()
            except:
                pass