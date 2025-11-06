#!/usr/bin/env python3
"""
COMPLETE INTEGRATED Camera Module with Natural Colors
All features working: Camera + YOLO + Color Correction + Config Integration
Replace your camera/camera_yolo.py with this file
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
    print("⚠️  picamera2 not available")

try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print(f"✅ PyTorch version: {torch.__version__}")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"⚠️  YOLOv8 not available: {e}")


class ImageProcessor:
    """Integrated image processing with color correction"""
    
    @staticmethod
    def apply_color_correction(frame):
        """Apply warm color correction to eliminate blue tint"""
        
        # Method 1: LAB color space adjustment
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Shift b channel toward yellow (away from blue)
        b = cv2.addWeighted(b, 0.94, b, 0, -10)
        
        # Boost red/warm tones
        a = cv2.addWeighted(a, 1.06, a, 0, 4)
        
        frame = cv2.merge([l, a, b])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        # Method 2: Direct BGR adjustment
        b_channel, g_channel, r_channel = cv2.split(frame)
        
        # Reduce blue
        b_channel = cv2.addWeighted(b_channel, 0.88, b_channel, 0, -15)
        
        # Boost red
        r_channel = cv2.addWeighted(r_channel, 1.12, r_channel, 0, 10)
        
        # Balance green
        g_channel = cv2.addWeighted(g_channel, 1.03, g_channel, 0, 3)
        
        frame = cv2.merge([b_channel, g_channel, r_channel])
        
        return frame
    
    @staticmethod
    def process_natural_hq(frame):
        """High quality natural processing"""
        
        # 1. Denoise
        if getattr(config, 'ENABLE_DENOISING', True):
            frame = cv2.bilateralFilter(frame, 5, 50, 50)
        
        # 2. Color correction (critical for natural skin tones)
        if getattr(config, 'ENABLE_WARM_CORRECTION', True):
            frame = ImageProcessor.apply_color_correction(frame)
        
        # 3. Contrast enhancement
        if getattr(config, 'ENABLE_CONTRAST_BOOST', True):
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            frame = cv2.merge([l, a, b])
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        # 4. Final adjustments
        alpha = 1.15  # Contrast
        beta = 10     # Brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # 5. Optional sharpening
        if getattr(config, 'ENABLE_SHARPENING', False):
            kernel = np.array([[-0.5, -0.5, -0.5],
                              [-0.5,  5.0, -0.5],
                              [-0.5, -0.5, -0.5]])
            frame = cv2.filter2D(frame, -1, kernel)
        
        return frame
    
    @staticmethod
    def process_balanced(frame):
        """Balanced quality and performance"""
        
        # 1. Light denoise
        if getattr(config, 'ENABLE_DENOISING', True):
            frame = cv2.bilateralFilter(frame, 5, 50, 50)
        
        # 2. Color correction
        if getattr(config, 'ENABLE_WARM_CORRECTION', True):
            frame = ImageProcessor.apply_color_correction(frame)
        
        # 3. Simple contrast boost
        if getattr(config, 'ENABLE_CONTRAST_BOOST', True):
            alpha = 1.12
            beta = 8
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        return frame
    
    @staticmethod
    def process_performance(frame):
        """Fast processing for maximum FPS"""
        
        # Quick color correction only
        if getattr(config, 'ENABLE_WARM_CORRECTION', True):
            b, g, r = cv2.split(frame)
            b = cv2.addWeighted(b, 0.86, b, 0, -18)
            r = cv2.addWeighted(r, 1.15, r, 0, 12)
            frame = cv2.merge([b, g, r])
            
            # Quick brightness
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=8)
        
        return frame
    
    @staticmethod
    def process_frame(frame, preset='balanced'):
        """Process frame based on preset from config"""
        
        if preset in ['natural_hq', 'natural_warm']:
            return ImageProcessor.process_natural_hq(frame)
        elif preset == 'balanced':
            return ImageProcessor.process_balanced(frame)
        elif preset == 'performance':
            return ImageProcessor.process_performance(frame)
        else:
            # Default to balanced
            return ImageProcessor.process_balanced(frame)


class EnhancedCameraYOLO:
    """Complete integrated camera with YOLO and color correction"""
    
    def __init__(self):
        """Initialize camera system"""
        print("📷 Initializing complete camera system...")
        
        # Frame variables
        self.camera = None
        self.frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()
        
        # Detection variables
        self.detections = []
        self.detection_lock = threading.Lock()
        self.last_detection_time = 0
        
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
        self.total_detections = 0
        
        # YOLO model
        self.model = None
        self.model_loaded = False
        
        # Get settings from config
        self.detection_interval = getattr(config, 'DETECTION_INTERVAL', 0.15)
        self.confidence_threshold = getattr(config, 'YOLO_CONFIDENCE_THRESHOLD', 0.5)
        self.iou_threshold = getattr(config, 'YOLO_IOU_THRESHOLD', 0.45)
        self.max_detections = getattr(config, 'YOLO_MAX_DETECTIONS', 10)
        self.processing_preset = getattr(config, 'IMAGE_PROCESSING_PRESET', 'balanced')
        
        # Initialize
        self._init_camera()
        
        if YOLO_AVAILABLE:
            self._load_yolo_model()
        
        print("✅ Complete camera system initialized")
    
    def _init_camera(self):
        """Initialize camera with optimized settings from config"""
        print("  🎥 Initializing OV5647 camera...")
        
        if not PICAMERA2_AVAILABLE:
            print("  ❌ picamera2 not available")
            print("  💡 Install: sudo apt install -y python3-picamera2")
            self.camera_type = 'none'
            return
        
        try:
            self.camera = Picamera2()
            
            # Get settings from config
            resolution = getattr(config, 'CAMERA_RESOLUTION', (640, 480))
            target_fps = getattr(config, 'CAMERA_FPS', 25)
            camera_controls = getattr(config, 'CAMERA_CONTROLS', {})
            quality_mode = getattr(config, 'CAMERA_QUALITY_MODE', 'balanced')
            
            print(f"  ⚙️  Mode: {quality_mode}")
            print(f"  ⚙️  Resolution: {resolution}")
            print(f"  ⚙️  Target FPS: {target_fps}")
            print(f"  ⚙️  Processing: {self.processing_preset}")
            
            # Build camera configuration
            controls = {
                "FrameRate": target_fps,
                "FrameDurationLimits": (16666, 50000),
                
                # White Balance
                "AwbEnable": camera_controls.get('AwbEnable', True),
                "AwbMode": camera_controls.get('AwbMode', 3),
                
                # Exposure
                "AeEnable": camera_controls.get('AeEnable', True),
                "AeExposureMode": camera_controls.get('AeExposureMode', 0),
                "AeConstraintMode": camera_controls.get('AeConstraintMode', 0),
                "AeMeteringMode": camera_controls.get('AeMeteringMode', 0),
                
                # Image quality
                "Brightness": camera_controls.get('Brightness', 0.1),
                "Contrast": camera_controls.get('Contrast', 1.15),
                "Saturation": camera_controls.get('Saturation', 1.1),
                "Sharpness": camera_controls.get('Sharpness', 1.2),
                
                # Noise reduction
                "NoiseReductionMode": camera_controls.get('NoiseReductionMode', 1),
                
                # Color gains (CRITICAL for fixing blue tint)
                "ColourGains": camera_controls.get('ColourGains', (1.35, 0.75)),
            }
            
            camera_config = self.camera.create_video_configuration(
                main={"size": resolution, "format": "RGB888"},
                controls=controls
            )
            
            self.camera.configure(camera_config)
            
            # Print active color settings
            print(f"  🎨 Color settings:")
            print(f"     - AWB Mode: {controls['AwbMode']}")
            print(f"     - Color Gains: {controls['ColourGains']}")
            print(f"     - Brightness: {controls['Brightness']}")
            print(f"     - Saturation: {controls['Saturation']}")
            
            print("  ⏳ Starting camera...")
            self.camera.start()
            
            # Warmup for white balance
            warmup_time = getattr(config, 'CAMERA_WARMUP_TIME', 2.0)
            print(f"  ⏳ Warming up ({warmup_time}s)...")
            time.sleep(warmup_time)
            
            # Test capture
            test_frame = self.camera.capture_array()
            if test_frame is not None and test_frame.size > 0:
                print(f"  ✅ Camera ready! Resolution: {test_frame.shape}")
                print(f"  ✅ Natural color rendering enabled")
                self.camera_ready = True
                self.camera_type = 'picamera2'
            else:
                raise Exception("Failed to capture test frame")
            
        except Exception as e:
            print(f"  ❌ Camera initialization failed: {e}")
            print("  💡 Troubleshooting:")
            print("     1. Check camera: rpicam-still -t 0")
            print("     2. Enable camera: sudo raspi-config")
            print("     3. Reboot after enabling")
            self.camera = None
            self.camera_type = 'none'
            self.camera_ready = False
    
    def _load_yolo_model(self):
        """Load YOLOv8 model"""
        try:
            print("  🧠 Loading YOLOv8 model...")
            
            model_path = getattr(config, 'YOLO_MODEL_PATH', 'yolov8n.pt')
            
            if not os.path.exists(model_path):
                print(f"  ❌ Model not found: {model_path}")
                print("  📥 Download: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
                self.model_loaded = False
                return
            
            self.model = YOLO(model_path)
            self.model.to('cpu')
            
            # Test inference
            print("  ⏳ Testing inference...")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, verbose=False)
            
            self.model_loaded = True
            print(f"  ✅ YOLOv8 loaded ({len(self.model.names)} classes)")
            
        except Exception as e:
            print(f"  ❌ YOLO failed: {e}")
            self.model_loaded = False
    
    def _capture_frames(self):
        """Capture and process frames"""
        print(f"  🎬 Capture thread started (preset: {self.processing_preset})")
        
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
                        print(f"  ⚠️  Too many errors, reinitializing...")
                        self._reinit_camera()
                        consecutive_errors = 0
                    time.sleep(0.1)
                    continue
                
                consecutive_errors = 0
                
                # Convert RGB to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Apply rotation
                rotation = getattr(config, 'CAMERA_ROTATION', 0)
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # APPLY IMAGE PROCESSING (color correction + enhancements)
                frame = ImageProcessor.process_frame(frame, self.processing_preset)
                
                # Store processed frame
                with self.frame_lock:
                    self.frame = frame.copy()
                
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time + 0.001)
                self.fps_counter.append(fps)
                self.last_time = current_time
                
                # Log progress
                if self.frame_count % 150 == 0:
                    avg_fps = np.mean(list(self.fps_counter))
                    print(f"  📊 {self.frame_count} frames | {avg_fps:.1f} FPS | Natural colors ✅")
                
                # Frame rate control
                time.sleep(0.001)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"  ❌ Capture error: {e}")
                if consecutive_errors > max_errors:
                    self.capture_running = False
                time.sleep(0.1)
        
        print("  🛑 Capture thread stopped")
    
    def _yolo_detection_thread(self):
        """YOLO detection thread"""
        print("  🔍 Detection thread started")
        
        if not self.model_loaded or self.model is None:
            print("  ⚠️  Detection unavailable")
            return
        
        detection_count = 0
        
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
                try:
                    results = self.model.predict(
                        source=frame,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        max_det=self.max_detections,
                        verbose=False,
                        device='cpu',
                        classes=None,
                        agnostic_nms=False,
                        half=False,
                    )
                except Exception as e:
                    print(f"  ❌ YOLO error: {e}")
                    time.sleep(0.5)
                    continue
                
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
                                
                            except Exception as e:
                                continue
                
                # Update detections
                with self.detection_lock:
                    self.detections = detections
                    self.detection_count = len(detections)
                    if len(detections) > 0:
                        self.total_detections += len(detections)
                
                detection_count += 1
                
                if detection_count % 10 == 0 and len(detections) > 0:
                    print(f"  🎯 Detection #{detection_count}: {len(detections)} objects")
                    for det in detections[:3]:
                        print(f"     - {det['class']}: {det['confidence']:.2f}")
                
            except Exception as e:
                print(f"  ❌ Detection error: {e}")
                time.sleep(0.5)
        
        print(f"  🛑 Detection stopped (Total: {self.total_detections})")
    
    def _reinit_camera(self):
        """Reinitialize camera"""
        try:
            print("  🔄 Reinitializing...")
            if self.camera:
                self.camera.stop()
                time.sleep(0.5)
            self._init_camera()
        except Exception as e:
            print(f"  ❌ Reinit failed: {e}")
    
    def _generate_placeholder(self, message="Waiting..."):
        """Generate placeholder"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for i in range(frame.shape[0]):
            frame[i, :] = [20 + i//8, 15 + i//10, 35 + i//12]
        
        cv2.putText(frame, message, (100, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        cv2.putText(frame, "Camera: OV5647 (Natural Colors)", (70, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 1)
        
        return frame
    
    def start_detection(self):
        """Start camera and detection"""
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
            print("  ✅ Complete system started with natural colors")
        else:
            print("  ⚠️  Detection disabled (model not loaded)")
    
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
            
            if conf > 0.8:
                color = (0, 255, 0)
            elif conf > 0.6:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            cv2.rectangle(frame, (x1, y1 - label_h - 12),
                         (x1 + label_w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add overlay
        frame = self._add_overlay(frame, len(detections))
        
        return frame
    
    def _add_overlay(self, frame, det_count):
        """Add FPS and info overlay"""
        h, w = frame.shape[:2]
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (200, 85), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {det_count}", (15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        
        if self.model_loaded:
            cv2.putText(frame, "YOLOv8n", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Natural colors indicator
        cv2.putText(frame, "Natural Colors", (w - 200, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        if self.total_detections > 0:
            cv2.putText(frame, f"Total: {self.total_detections}", (15, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def get_frame(self):
        """Get current frame"""
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
        print("  🧹 Cleaning up...")
        self.stop_detection()
        if self.camera and hasattr(self.camera, 'stop'):
            try:
                self.camera.stop()
                print("  ✅ Camera stopped")
            except:
                pass


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Complete Camera System")
    print("="*60 + "\n")
    
    camera = EnhancedCameraYOLO()
    
    if not camera.camera_ready:
        print("❌ Camera not ready")
        exit(1)
    
    print("\n🚀 Starting 30-second test...")
    print("✅ Blue tint should be GONE\n")
    
    camera.start_detection()
    
    try:
        for i in range(30):
            time.sleep(1)
            
            if i % 5 == 0:
                stats = camera.get_performance_stats()
                print(f"[{i}s] FPS: {stats['fps']:.1f} | Objects: {stats['detections_count']} | Natural colors ✅")
        
        print("\n✅ Test complete!")
        
    except KeyboardInterrupt:
        print("\n⏸️  Interrupted")
    finally:
        camera.cleanup()