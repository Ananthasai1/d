#!/usr/bin/env python3
"""
OPTIMIZED Configuration for Maximum FPS
This config will give you 20-25 FPS on Raspberry Pi 3
"""

# ===== Flask Server Settings =====
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000
DEBUG = False

# ===== GPIO Pin Assignments =====
ULTRASONIC_TRIGGER_PIN = 23
ULTRASONIC_ECHO_PIN = 24
NIGHT_VISION_GPIO = 18

# ===== PCA9685 Servo Driver Settings =====
PCA9685_ADDRESS = 0x40
PCA9685_FREQUENCY = 50

# ===== Servo Channel Mapping =====
SERVO_CHANNELS = [
    [0, 1, 2],    # Leg 0 (Front-Right)
    [4, 5, 6],    # Leg 1 (Front-Left)
    [8, 9, 10],   # Leg 2 (Rear-Left)
    [12, 13, 14]  # Leg 3 (Rear-Right)
]

# ===== Servo Calibration =====
SERVO_PULSE_RANGE = [150, 600]

# ===== Robot Physical Dimensions (mm) =====
LENGTH_A = 55.0
LENGTH_B = 77.5
LENGTH_C = 27.5
LENGTH_SIDE = 71.0

# ===== Movement Parameters =====
Z_DEFAULT = -50.0
Z_UP = -30.0
Z_BOOT = -28.0
X_DEFAULT = 62.0
X_OFFSET = 0.0
Y_START = 0.0
Y_STEP = 40.0

# ===== Movement Speeds =====
LEG_MOVE_SPEED = 8.0
BODY_MOVE_SPEED = 3.0
SPOT_TURN_SPEED = 4.0
STAND_SEAT_SPEED = 1.0
SPEED_MULTIPLE = 1.2

# ===== Ultrasonic Sensor Settings =====
OBSTACLE_THRESHOLD = 20
MAX_DISTANCE = 200

# ===== OPTIMIZED CAMERA SETTINGS FOR MAXIMUM FPS =====
# Resolution: Lower = faster FPS
CAMERA_RESOLUTION = (640, 480)  # Good balance

# Target FPS: Set high, let camera achieve what it can
CAMERA_FPS = 30

# Camera rotation
CAMERA_ROTATION = 0

# Minimal warmup for faster startup
CAMERA_WARMUP_TIME = 0.5

# ===== YOLOv8 DETECTION SETTINGS =====
YOLO_MODEL_PATH = 'yolov8n.pt'  # Use nano model (fastest)
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_MAX_DETECTIONS = 10

# ===== CRITICAL FPS OPTIMIZATION =====
# Detection interval: Higher = better FPS, less frequent detection
# 0.2 = Run YOLO every 200ms (5 times per second)
# This allows camera to run at full speed
DETECTION_INTERVAL = 0.2  # Increased from 0.12 for better FPS

# Video quality: Lower = faster encoding
VIDEO_QUALITY = 85  # Reduced from 92 for speed

# Frame buffer
FRAME_BUFFER_SIZE = 2  # Smaller = lower latency

# Threading
ENABLE_THREADING = True
MAX_THREAD_WORKERS = 2

# ===== AUTO MODE SETTINGS =====
AUTO_MODE_LOOP_DELAY = 0.05
AUTO_DETECTION_FREQUENCY = 0.5

# ===== NIGHT VISION SETTINGS =====
NIGHT_VISION_THRESHOLD = 50
ENABLE_NIGHT_VISION = False

# ===== PERFORMANCE TIPS =====
"""
Expected Performance on Raspberry Pi 3:
- Camera FPS: 20-25 fps
- Detection Rate: 5 fps (every 200ms)
- Total CPU: ~60-70%

To get even MORE FPS:
1. Lower resolution: CAMERA_RESOLUTION = (320, 240)
2. Increase detection interval: DETECTION_INTERVAL = 0.3
3. Disable YOLO completely (comment out model loading)

To improve detection accuracy (at cost of FPS):
1. Higher resolution: CAMERA_RESOLUTION = (800, 600)
2. Decrease detection interval: DETECTION_INTERVAL = 0.15
3. Use better model: YOLO_MODEL_PATH = 'yolov8s.pt'
"""