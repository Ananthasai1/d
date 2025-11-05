#!/usr/bin/env python3
"""
Configuration settings for CyberCrawl Spider Robot
OPTIMIZED for better FPS and natural image quality
"""

# ===== Flask Server Settings =====
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000
DEBUG = False

# ===== GPIO Pin Assignments =====
ULTRASONIC_TRIGGER_PIN = 23
ULTRASONIC_ECHO_PIN = 24
NIGHT_VISION_GPIO = 18  # Optional IR LED control

# ===== PCA9685 Servo Driver Settings =====
PCA9685_ADDRESS = 0x40
PCA9685_FREQUENCY = 50  # Hz (standard for servos)

# ===== Servo Channel Mapping =====
# Leg numbering: 0=Front-Right, 1=Front-Left, 2=Rear-Left, 3=Rear-Right
# Joint numbering: 0=Coxa, 1=Femur, 2=Tibia
SERVO_CHANNELS = [
    [0, 1, 2],    # Leg 0 (Front-Right)
    [4, 5, 6],    # Leg 1 (Front-Left)
    [8, 9, 10],   # Leg 2 (Rear-Left)
    [12, 13, 14]  # Leg 3 (Rear-Right)
]

# ===== Servo Calibration =====
SERVO_PULSE_RANGE = [150, 600]  # Pulse width for 0-180 degrees

# ===== Robot Physical Dimensions (mm) =====
LENGTH_A = 55.0      # Upper leg length
LENGTH_B = 77.5      # Lower leg length
LENGTH_C = 27.5      # Coxa length
LENGTH_SIDE = 71.0   # Body side length

# ===== Movement Parameters =====
Z_DEFAULT = -50.0    # Default standing height
Z_UP = -30.0         # Leg lift height
Z_BOOT = -28.0       # Boot/sit position
X_DEFAULT = 62.0     # Default X position
X_OFFSET = 0.0       # X offset for body
Y_START = 0.0        # Starting Y position
Y_STEP = 40.0        # Step length

# ===== Movement Speeds =====
LEG_MOVE_SPEED = 8.0
BODY_MOVE_SPEED = 3.0
SPOT_TURN_SPEED = 4.0
STAND_SEAT_SPEED = 1.0
SPEED_MULTIPLE = 1.2

# ===== Ultrasonic Sensor Settings =====
OBSTACLE_THRESHOLD = 20  # cm - distance to trigger avoidance
MAX_DISTANCE = 200       # cm - maximum detection range

# ===== OPTIMIZED CAMERA SETTINGS =====
"""
Quality Modes:
- 'high': Best quality, natural image, ~15-20 FPS (800x600)
- 'balanced': Good quality, smooth video, ~20-25 FPS (640x480) [RECOMMENDED]
- 'performance': Maximum FPS, ~25-30 FPS (320x240)
"""
CAMERA_QUALITY_MODE = 'balanced'  # Change to 'high' or 'performance' as needed

# Resolution (auto-adjusted based on quality mode)
CAMERA_RESOLUTION = (640, 480)  # Balanced: good quality + performance

# FPS Target (higher = smoother video)
CAMERA_FPS = 25  # Optimized for Raspberry Pi 3/4

# Camera warmup time
CAMERA_WARMUP_TIME = 1  # Reduced from 2 seconds

# Camera rotation
CAMERA_ROTATION = 0  # 0, 90, 180, 270

# ===== IMAGE PROCESSING SETTINGS =====
"""
Processing Presets:
- 'natural_hq': Best quality, natural colors, slight performance impact
- 'balanced': Good quality with minimal processing [RECOMMENDED]
- 'performance': Minimal processing for maximum FPS
"""
IMAGE_PROCESSING_PRESET = 'balanced'

# Enable specific enhancements (only used in 'balanced' mode)
ENABLE_DENOISING = True        # Reduces grain/noise
ENABLE_CONTRAST_BOOST = True   # Better visibility
ENABLE_SHARPENING = False      # Keep False for natural look

# ===== YOLOv8 DETECTION SETTINGS =====
YOLO_MODEL_PATH = 'yolov8n.pt'  # Nano model (fastest)
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_MAX_DETECTIONS = 10  # Limit for performance

# ===== PERFORMANCE OPTIMIZATION =====
"""
Key settings for FPS improvement:
- DETECTION_INTERVAL: Time between YOLO inferences
  Lower = more frequent detection but lower FPS
  Higher = less frequent detection but higher FPS
"""
DETECTION_INTERVAL = 0.15  # Optimized: run YOLO every 150ms

# Video streaming quality
VIDEO_QUALITY = 90  # JPEG compression (85-95 recommended)
                    # Higher = better quality, larger bandwidth

# Frame buffer settings
FRAME_BUFFER_SIZE = 3  # Smaller buffer = lower latency

# Threading optimization
ENABLE_THREADING = True
MAX_THREAD_WORKERS = 2

# ===== CAMERA HARDWARE CONTROLS =====
"""
These control the actual camera sensor behavior
Set to 'auto' for natural-looking images
"""
CAMERA_CONTROLS = {
    # Auto White Balance (AWB)
    'AwbEnable': True,
    'AwbMode': 0,  # 0=Auto, 1=Tungsten, 2=Fluorescent, 3=Indoor, 4=Daylight, 5=Cloudy
    
    # Auto Exposure (AE)
    'AeEnable': True,
    'AeExposureMode': 0,  # 0=Normal, 1=Short, 2=Long, 3=Custom
    'AeConstraintMode': 0,  # 0=Normal, 1=Highlight, 2=Shadow
    'AeMeteringMode': 0,   # 0=Centre, 1=Spot, 2=Matrix, 3=Custom
    
    # Image adjustments (neutral values for natural look)
    'Brightness': 0.0,     # -1.0 to 1.0 (0.0 = neutral)
    'Contrast': 1.0,       # 0.0 to 2.0 (1.0 = neutral)
    'Saturation': 1.0,     # 0.0 to 2.0 (1.0 = neutral)
    'Sharpness': 1.0,      # 0.0 to 16.0 (1.0 = moderate)
    
    # Noise reduction
    'NoiseReductionMode': 1,  # 0=Off, 1=Fast, 2=HighQuality
    
    # Color gains (for fine-tuning white balance)
    'ColourGains': (1.0, 1.0),  # (red_gain, blue_gain)
}

# ===== AUTO MODE SETTINGS =====
AUTO_MODE_LOOP_DELAY = 0.05  # seconds between sensor readings
AUTO_DETECTION_FREQUENCY = 0.5  # seconds between object checks

# ===== NIGHT VISION SETTINGS =====
NIGHT_VISION_THRESHOLD = 50  # Brightness threshold
ENABLE_NIGHT_VISION = False  # Set to True if IR LEDs are connected

# ===== ADVANCED PERFORMANCE TUNING =====
"""
For Raspberry Pi 3:
- Use 'balanced' or 'performance' mode
- CAMERA_FPS = 20-25
- DETECTION_INTERVAL = 0.15-0.2

For Raspberry Pi 4/5:
- Can use 'high' mode
- CAMERA_FPS = 25-30
- DETECTION_INTERVAL = 0.1
"""

# Adaptive quality (experimental)
ENABLE_ADAPTIVE_QUALITY = False  # Automatically adjust based on CPU load

# Debug mode
SHOW_FPS_IN_CONSOLE = False  # Print FPS to console
SHOW_DETECTION_LOGS = False  # Print detection info

# ===== QUICK PRESETS =====
"""
Copy one of these preset configurations:

PRESET 1 - MAXIMUM QUALITY (for photography/recording):
    CAMERA_QUALITY_MODE = 'high'
    CAMERA_RESOLUTION = (800, 600)
    CAMERA_FPS = 15
    IMAGE_PROCESSING_PRESET = 'natural_hq'
    DETECTION_INTERVAL = 0.2
    VIDEO_QUALITY = 95

PRESET 2 - BALANCED (RECOMMENDED for general use):
    CAMERA_QUALITY_MODE = 'balanced'
    CAMERA_RESOLUTION = (640, 480)
    CAMERA_FPS = 25
    IMAGE_PROCESSING_PRESET = 'balanced'
    DETECTION_INTERVAL = 0.15
    VIDEO_QUALITY = 90

PRESET 3 - MAXIMUM FPS (for fast-moving objects):
    CAMERA_QUALITY_MODE = 'performance'
    CAMERA_RESOLUTION = (320, 240)
    CAMERA_FPS = 30
    IMAGE_PROCESSING_PRESET = 'performance'
    DETECTION_INTERVAL = 0.1
    VIDEO_QUALITY = 85
"""