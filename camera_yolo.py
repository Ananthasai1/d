#!/usr/bin/env python3
"""
COMPLETE FIXED VERSION of camera/camera_yolo.py
Properly handles YOLO detection with PyTorch 2.6+ and optimizes performance
Replace your camera/camera_yolo.py with this entire file
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import picamera2
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️  picamera2 not available")

# Import YOLO with proper handling
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print(f"✅ PyTorch version: {torch.__version__}")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"⚠️  YOLOv8 not available: {e}")


class EnhancedCameraYOLO:
    """Enhanced camera with properly working YOLOv8 object detection"""
    
    def __init__(self):
        """Initialize camera and YOLO"""
        print("📷 Initializing camera system...")
        
        # Camera and frame variables
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
        
        # Detection settings
        self.detection_interval = getattr(config, 'DETECTION_INTERVAL', 0.2)
        self.confidence_threshold = getattr(config, 'YOLO_CONFIDENCE_THRESHOLD', 0.5)
        self.iou_threshold = getattr(config, 'YOLO_IOU_THRESHOLD', 0.45)
        self.max_detections = getattr(config, 'YOLO_MAX_DETECTIONS', 10)
        
        # Initialize camera
        self._init_camera()
        
        # Load YOLO model
        if YOLO_AVAILABLE:
            self._load_yolo_model()
        
        print("✅ Camera system initialized")
    
    def _init_camera(self):
        """Initialize camera with picamera2"""
        print("  🎥 Initializing OV5647 camera...")
        
        if not PICAMERA2_AVAILABLE:
            print("  ❌ picamera2 not available")
            print("  💡 Install: sudo apt install -y python3-picamera2")
            self.camera_type = 'none'
            return
        
        try:
            # Create Picamera2 instance
            self.camera = Picamera2()
            
            # Get resolution from config
            resolution = getattr(config, 'CAMERA_RESOLUTION', (640, 480))
            target_fps = getattr(config, 'CAMERA_FPS', 25)
            
            # Configure camera for video
            camera_config = self.camera.create_video_configuration(
                main={"size": resolution, "format": "RGB888"},
                controls={
                    "FrameRate": target_fps,
                    "FrameDurationLimits": (16666, 50000),  # 20-60 FPS range
                    "AwbEnable": True,   # Auto white balance
                    "AeEnable": True,    # Auto exposure
                    "NoiseReductionMode": 1,  # Fast noise reduction
                }
            )
            
            self.camera.configure(camera_config)
            
            print("  ⏳ Starting camera...")
            self.camera.start()
            
            # Quick warmup
            warmup_time = getattr(config, 'CAMERA_WARMUP_TIME', 1.0)
            print(f"  ⏳ Warming up camera ({warmup_time}s)...")
            time.sleep(warmup_time)
            
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
            print("  💡 Troubleshooting:")
            print("     1. Check camera: rpicam-still -t 0")
            print("     2. Enable camera: sudo raspi-config -> Interface Options -> Camera")
            print("     3. Reboot after enabling camera")
            self.camera = None
            self.camera_type = 'none'
            self.camera_ready = False
    
    def _load_yolo_model(self):
        """Load YOLOv8 model with proper PyTorch handling"""
        try:
            print("  🧠 Loading YOLOv8 model...")
            
            model_path = getattr(config, 'YOLO_MODEL_PATH', 'yolov8n.pt')
            
            if not os.path.exists(model_path):
                print(f"  ❌ Model file not found: {model_path}")
                print("  📥 Download with:")
                print(f"     wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
                self.model_loaded = False
                return
            
            print("  🔧 Configuring PyTorch security...")
            
            # Method 1: Try with safe globals (PyTorch 2.6+)
            try:
                # Add necessary safe globals for YOLO
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals([DetectionModel])
                print("  ✅ PyTorch security configured")
            except Exception as e:
                print(f"  ⚠️  Safe globals config: {e}")
            
            # Load model
            print("  ⏳ Loading YOLO model...")
            
            try:
                # Try loading normally first
                self.model = YOLO(model_path)
            except Exception as e:
                # If that fails, use weights_only=False workaround
                print(f"  ⚠️  Normal load failed: {type(e).__name__}")
                print("  🔄 Trying alternative loading method...")
                
                # Patch torch.load temporarily
                original_load = torch.load
                
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = patched_load
                
                try:
                    self.model = YOLO(model_path)
                    print("  ⚠️  Using legacy loading mode...")
                finally:
                    torch.load = original_load
            
            # Set to CPU and verify
            self.model.to('cpu')
            
            # Test inference with proper error handling
            print("  ⏳ Testing model inference...")
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            try:
                results = self.model(test_image, verbose=False, conf=0.5)
                print(f"  ✅ Model inference successful")
                self.model_loaded = True
                
                # Print model info
                print(f"  📊 Model classes: {len(self.model.names)}")
                print(f"  📊 Sample classes: {list(self.model.names.values())[:5]}...")
                
            except Exception as e:
                print(f"  ❌ Model inference test failed: {e}")
                self.model_loaded = False
                return
            
            print("  ✅ YOLOv8 loaded successfully")
            
        except Exception as e:
            print(f"  ❌ YOLO loading failed: {e}")
            print(f"  💡 Error type: {type(e).__name__}")
            self.model_loaded = False
    
    def _capture_frames(self):
        """Optimized frame capture thread"""
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
                
                # Capture frame
                frame = self.camera.capture_array()
                
                if frame is None or frame.size == 0:
                    consecutive_errors += 1
                    if consecutive_errors > max_errors:
                        print(f"  ⚠️  Too many errors, reinitializing camera...")
                        self._reinit_camera()
                        consecutive_errors = 0
                    time.sleep(0.1)
                    continue
                
                consecutive_errors = 0
                
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Apply rotation if configured
                rotation = getattr(config, 'CAMERA_ROTATION', 0)
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Store frame
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
                    print(f"  📊 Captured {self.frame_count} frames | {avg_fps:.1f} FPS")
                
                # Frame rate control
                time.sleep(0.001)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"  ❌ Capture error: {e}")
                if consecutive_errors > max_errors:
                    print(f"  ⚠️  Too many errors, stopping")
                    self.capture_running = False
                time.sleep(0.1)
        
        print("  🛑 Capture thread stopped")
    
    def _yolo_detection_thread(self):
        """PROPERLY WORKING YOLOv8 detection thread"""
        print("  🔍 Detection thread started")
        
        if not self.model_loaded or self.model is None:
            print("  ⚠️  Detection unavailable - model not loaded")
            return
        
        detection_frame_count = 0
        
        while self.detection_running:
            try:
                current_time = time.time()
                
                # Throttle detection rate
                if current_time - self.last_detection_time < self.detection_interval:
                    time.sleep(0.05)
                    continue
                
                self.last_detection_time = current_time
                
                # Get current frame
                with self.frame_lock:
                    if self.frame is None:
                        time.sleep(0.1)
                        continue
                    frame = self.frame.copy()
                
                # Run YOLO detection with proper settings
                try:
                    results = self.model.predict(
                        source=frame,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        max_det=self.max_detections,
                        verbose=False,
                        device='cpu',
                        classes=None,  # Detect all classes
                        agnostic_nms=False,
                        half=False,  # Don't use FP16 on CPU
                    )
                    
                except Exception as e:
                    print(f"  ❌ YOLO predict error: {e}")
                    time.sleep(0.5)
                    continue
                
                # Parse detections
                detections = []
                
                if results and len(results) > 0:
                    result = results[0]  # Get first result
                    
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        
                        for i in range(len(boxes)):
                            try:
                                # Get box coordinates
                                box = boxes.xyxy[i].cpu().numpy()
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Get confidence and class
                                conf = float(boxes.conf[i].cpu().numpy())
                                cls_id = int(boxes.cls[i].cpu().numpy())
                                
                                # Get class name
                                class_name = self.model.names[cls_id]
                                
                                # Create detection object
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
                                print(f"  ⚠️  Box parsing error: {e}")
                                continue
                
                # Update detections
                with self.detection_lock:
                    self.detections = detections
                    self.detection_count = len(detections)
                    if len(detections) > 0:
                        self.total_detections += len(detections)
                
                detection_frame_count += 1
                
                # Log detection info
                if detection_frame_count % 10 == 0 and len(detections) > 0:
                    print(f"  🎯 Detection #{detection_frame_count}: Found {len(detections)} objects")
                    for det in detections[:3]:  # Show first 3
                        print(f"     - {det['class']}: {det['confidence']:.2f}")
                
            except Exception as e:
                print(f"  ❌ Detection thread error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
        
        print(f"  🛑 Detection thread stopped (Total: {self.total_detections} detections)")
    
    def _reinit_camera(self):
        """Reinitialize camera after errors"""
        try:
            print("  🔄 Reinitializing camera...")
            if self.camera:
                self.camera.stop()
                time.sleep(0.5)
            self._init_camera()
        except Exception as e:
            print(f"  ❌ Camera reinit failed: {e}")
    
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
        else:
            print("  ⚠️  Detection disabled (model not loaded)")
    
    def stop_detection(self):
        """Stop all threads"""
        self.detection_running = False
        self.capture_running = False
        time.sleep(0.5)
        self.is_running = False
    
    def get_frame_with_detections(self):
        """Get frame with bounding boxes drawn"""
        # Get current frame
        with self.frame_lock:
            if self.frame is None:
                return self._generate_placeholder("Initializing...")
            frame = self.frame.copy()
        
        # Get current detections
        with self.detection_lock:
            detections = self.detections.copy()
        
        # Draw detections on frame
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Color by confidence (BGR format)
            if conf > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif conf > 0.6:
                color = (0, 165, 255)  # Orange - medium
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            # Draw bounding box (thicker for better visibility)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label
            label = f"{class_name}: {conf:.2f}"
            
            # Get label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (x1, y1 - label_h - 12), 
                (x1 + label_w + 10, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, 
                label, 
                (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 0), 
                2
            )
        
        # Add overlay info
        frame = self._add_overlay(frame, len(detections))
        
        return frame
    
    def _add_overlay(self, frame, det_count):
        """Add FPS and info overlay"""
        h, w = frame.shape[:2]
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        # Semi-transparent overlay for better text visibility
        overlay = frame.copy()
        
        # FPS counter (top-left)
        cv2.rectangle(overlay, (5, 5), (150, 70), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {det_count}", (15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        
        # Model status (top-right)
        if self.model_loaded:
            status_text = "YOLOv8n"
            status_color = (255, 0, 255)
        else:
            status_text = "No Model"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Total detections (bottom-left)
        if self.total_detections > 0:
            cv2.putText(frame, f"Total: {self.total_detections}", (15, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def get_frame(self):
        """Get current frame without detections"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_detections(self):
        """Get current detections list"""
        with self.detection_lock:
            return self.detections.copy()
    
    def get_performance_stats(self):
        """Get performance statistics"""
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
        """Cleanup resources"""
        print("  🧹 Cleaning up camera...")
        self.stop_detection()
        if self.camera and hasattr(self.camera, 'stop'):
            try:
                self.camera.stop()
                print("  ✅ Camera stopped")
            except:
                pass


# Test function
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Enhanced Camera with YOLO Detection")
    print("="*60 + "\n")
    
    camera = EnhancedCameraYOLO()
    
    if not camera.camera_ready:
        print("❌ Camera not ready. Check connection.")
        exit(1)
    
    print("\n🚀 Starting detection test (30 seconds)...")
    print("Place objects in view: person, cup, phone, etc.\n")
    
    camera.start_detection()
    
    try:
        for i in range(30):
            time.sleep(1)
            
            if i % 5 == 0:
                stats = camera.get_performance_stats()
                detections = camera.get_detections()
                
                print(f"[{i}s] FPS: {stats['fps']:.1f} | "
                      f"Objects: {stats['detections_count']} | "
                      f"Total: {stats['total_detections']}")
                
                if detections:
                    for det in detections:
                        print(f"  → {det['class']}: {det['confidence']:.2f}")
        
        print("\n✅ Test complete!")
        
    except KeyboardInterrupt:
        print("\n⏸️  Test interrupted")
    finally:
        camera.cleanup()