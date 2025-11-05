#!/usr/bin/env python3
"""
Enhanced Configuration for CyberCrawl Spider Robot
Optimized for superior image quality and colors
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

# ===== ENHANCED CAMERA SETTINGS =====
# Resolution: Higher = better quality, lower FPS
# Options: (640, 480), (800, 600), (1280, 720), (1920, 1080)
CAMERA_RESOLUTION = (640, 480)  # Good balance for Pi 3

# Frame rate: Lower = better quality per frame
# Recommended: 10-15 for Pi 3, 20-30 for Pi 4/5
CAMERA_FPS = 12  # Reduced slightly for better processing time

# Camera warmup time
CAMERA_WARMUP_TIME = 2

# Camera rotation (0, 90, 180, 270)
CAMERA_ROTATION = 0

# ===== IMAGE ENHANCEMENT SETTINGS =====
# Enable/disable real-time image enhancement
ENABLE_IMAGE_ENHANCEMENT = True

# Brightness adjustment (-100 to 100)
# Positive = brighter, Negative = darker
BRIGHTNESS_ADJUST = 15

# Contrast adjustment (-100 to 100)
# Higher = more contrast
CONTRAST_ADJUST = 20

# Color saturation (0.5 to 2.0)
# 1.0 = normal, >1.0 = more vibrant
SATURATION_FACTOR = 1.3

# Sharpness
ENABLE_SHARPENING = True

# Noise reduction (reduces FPS but cleaner image)
ENABLE_DENOISING = False

# ===== VIDEO STREAMING QUALITY =====
# JPEG compression quality (1-100)
# Higher = better quality but more bandwidth
# Recommended: 85-95 for good quality, 70-85 for performance
VIDEO_QUALITY = 92  # High quality

# ===== YOLOv8 DETECTION SETTINGS =====
# Model options:
# - yolov8n.pt (nano - fastest, ~6MB)
# - yolov8s.pt (small - balanced, ~22MB)
# - yolov8m.pt (medium - slower, ~52MB)
YOLO_MODEL_PATH = 'yolov8n.pt'

# Confidence threshold (0.0 to 1.0)
# Lower = more detections (but more false positives)
YOLO_CONFIDENCE_THRESHOLD = 0.5

# IoU threshold for NMS
YOLO_IOU_THRESHOLD = 0.45

# Maximum detections per frame
YOLO_MAX_DETECTIONS = 10

# ===== PERFORMANCE TUNING =====
# Time between YOLO inferences (seconds)
# Higher = lower CPU usage, less frequent detection
# Lower = more frequent detection, higher CPU
DETECTION_INTERVAL = 0.15  # Slightly increased for better frame processing

# Enable multi-threading
ENABLE_THREADING = True

# ===== AUTO MODE SETTINGS =====
AUTO_MODE_LOOP_DELAY = 0.05
AUTO_DETECTION_FREQUENCY = 0.5

# ===== NIGHT VISION SETTINGS =====
NIGHT_VISION_THRESHOLD = 50
ENABLE_NIGHT_VISION = False

# ===== COLOR GRADING PRESETS =====
# Available presets: 'natural', 'vibrant', 'warm', 'cool', 'cinematic'
COLOR_PRESET = 'vibrant'

# Preset definitions
COLOR_PRESETS = {
    'natural': {
        'brightness': 0,
        'contrast': 10,
        'saturation': 1.0,
        'description': 'Neutral, realistic colors'
    },
    'vibrant': {
        'brightness': 15,
        'contrast': 20,
        'saturation': 1.3,
        'description': 'Punchy, vivid colors'
    },
    'warm': {
        'brightness': 10,
        'contrast': 15,
        'saturation': 1.2,
        'temperature': 'warm',
        'description': 'Warm, orange-shifted tones'
    },
    'cool': {
        'brightness': 5,
        'contrast': 15,
        'saturation': 1.1,
        'temperature': 'cool',
        'description': 'Cool, blue-shifted tones'
    },
    'cinematic': {
        'brightness': -5,
        'contrast': 30,
        'saturation': 1.4,
        'description': 'Film-like with crushed blacks'
    }
}

# ===== ADVANCED CAMERA CONTROLS =====
# Fine-tune these if needed
CAMERA_AWB_MODE = 0  # 0=Auto, 1=Tungsten, 2=Fluorescent, 3=Indoor, 4=Daylight, 5=Cloudy
CAMERA_AE_MODE = 0   # 0=Normal, 1=Short, 2=Long, 3=Custom
CAMERA_METERING_MODE = 0  # 0=Centre, 1=Spot, 2=Matrix, 3=Custom

# ===== PERFORMANCE PROFILES =====
# Uncomment one profile or create your own

# HIGH QUALITY (slower, best image)
# CAMERA_RESOLUTION = (800, 600)
# CAMERA_FPS = 10
# VIDEO_QUALITY = 95
# DETECTION_INTERVAL = 0.2
# ENABLE_IMAGE_ENHANCEMENT = True

# BALANCED (default - good quality and speed)
CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS = 12
VIDEO_QUALITY = 92
DETECTION_INTERVAL = 0.15
ENABLE_IMAGE_ENHANCEMENT = True

# PERFORMANCE (faster, lower quality)
# CAMERA_RESOLUTION = (320, 240)
# CAMERA_FPS = 20
# VIDEO_QUALITY = 80
# DETECTION_INTERVAL = 0.1
# ENABLE_IMAGE_ENHANCEMENT = False

# ===== DISPLAY SETTINGS =====
# Show FPS on overlay
SHOW_FPS_OVERLAY = True

# Show detection count
SHOW_DETECTION_COUNT = True

# Bounding box thickness
BBOX_THICKNESS = 3

# Label font scale
LABEL_FONT_SCALE = 0.7

# ===== DEBUGGING =====
# Print frame capture stats
DEBUG_FRAME_CAPTURE = False

# Print detection results
DEBUG_DETECTIONS = False

# Verbose YOLO output
YOLO_VERBOSE = False