#!/usr/bin/env python3
"""
Configuration settings for CyberCrawl Spider Robot
FIXED: Corrected white balance to eliminate blue tint
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

# ===== OPTIMIZED CAMERA SETTINGS =====
CAMERA_QUALITY_MODE = 'balanced'
CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS = 25
CAMERA_WARMUP_TIME = 2
CAMERA_ROTATION = 0

# ===== IMAGE PROCESSING SETTINGS =====
IMAGE_PROCESSING_PRESET = 'natural_warm'  # New preset to fix blue tint
ENABLE_DENOISING = True
ENABLE_CONTRAST_BOOST = True
ENABLE_SHARPENING = False
ENABLE_WARM_CORRECTION = True  # New: Enable warm color correction

# ===== YOLOv8 DETECTION SETTINGS =====
YOLO_MODEL_PATH = 'yolov8n.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_MAX_DETECTIONS = 10

# ===== PERFORMANCE OPTIMIZATION =====
DETECTION_INTERVAL = 0.15
VIDEO_QUALITY = 90
FRAME_BUFFER_SIZE = 3
ENABLE_THREADING = True
MAX_THREAD_WORKERS = 2

# ===== FIXED CAMERA HARDWARE CONTROLS =====
"""
FIXED: Corrected white balance settings to eliminate blue tint
- Changed AWB mode to Indoor (3) for better LED/fluorescent handling
- Adjusted color gains to warm up the image
- Increased brightness and saturation slightly
"""
CAMERA_CONTROLS = {
    # Auto White Balance - FIXED for indoor lighting
    'AwbEnable': True,
    'AwbMode': 3,  # CHANGED: 3 = Indoor (best for LED/fluorescent lights)
                   # Try: 1 = Tungsten for incandescent bulbs
                   # Try: 4 = Daylight if near windows
    
    # Auto Exposure
    'AeEnable': True,
    'AeExposureMode': 0,
    'AeConstraintMode': 0,
    'AeMeteringMode': 0,
    
    # Image adjustments - FIXED for warmer, more natural skin tones
    'Brightness': 0.1,      # CHANGED: Slightly brighter (was 0.0)
    'Contrast': 1.15,       # CHANGED: Better definition (was 1.0)
    'Saturation': 1.1,      # CHANGED: More color (was 1.0)
    'Sharpness': 1.2,       # CHANGED: Slightly sharper (was 1.0)
    
    # Noise reduction
    'NoiseReductionMode': 1,
    
    # Color gains - FIXED: Reduce blue, increase red/warmth
    'ColourGains': (1.35, 0.75),  # CHANGED: (red_gain, blue_gain)
                                   # red: 1.35 = 35% more warmth
                                   # blue: 0.75 = 25% less blue cast
                                   # Adjust if needed:
                                   # More warmth: (1.4, 0.7) or (1.5, 0.65)
                                   # Less warmth: (1.2, 0.85) or (1.1, 0.9)
}

# ===== AUTO MODE SETTINGS =====
AUTO_MODE_LOOP_DELAY = 0.05
AUTO_DETECTION_FREQUENCY = 0.5

# ===== NIGHT VISION SETTINGS =====
NIGHT_VISION_THRESHOLD = 50
ENABLE_NIGHT_VISION = False

# ===== DEBUG SETTINGS =====
SHOW_FPS_IN_CONSOLE = False
SHOW_DETECTION_LOGS = False

# ===== COLOR CORRECTION PRESETS =====
"""
Choose the best preset for your lighting conditions:
"""

# Preset 1: Indoor LED/Fluorescent (Most Common - RECOMMENDED)
COLOR_CORRECTION_PRESET_INDOOR = {
    'AwbMode': 3,
    'ColourGains': (1.35, 0.75),
    'Brightness': 0.1,
    'Saturation': 1.1
}

# Preset 2: Incandescent/Tungsten Bulbs (Warm Yellow Light)
COLOR_CORRECTION_PRESET_TUNGSTEN = {
    'AwbMode': 1,
    'ColourGains': (1.2, 0.85),
    'Brightness': 0.05,
    'Saturation': 1.05
}

# Preset 3: Natural Daylight (Near Window)
COLOR_CORRECTION_PRESET_DAYLIGHT = {
    'AwbMode': 4,
    'ColourGains': (1.1, 0.9),
    'Brightness': 0.0,
    'Saturation': 1.0
}

# Preset 4: Very Blue Room (Strong Blue Cast)
COLOR_CORRECTION_PRESET_STRONG_BLUE = {
    'AwbMode': 1,  # Tungsten mode adds yellow/red
    'ColourGains': (1.5, 0.65),  # Strong red boost, heavy blue reduction
    'Brightness': 0.15,
    'Saturation': 1.15
}

# Active preset (change this to test different presets)
ACTIVE_COLOR_PRESET = 'indoor'  # Options: 'indoor', 'tungsten', 'daylight', 'strong_blue'

# ===== APPLY PRESET =====
def apply_color_preset():
    """Apply the selected color preset to CAMERA_CONTROLS"""
    global CAMERA_CONTROLS
    
    presets = {
        'indoor': COLOR_CORRECTION_PRESET_INDOOR,
        'tungsten': COLOR_CORRECTION_PRESET_TUNGSTEN,
        'daylight': COLOR_CORRECTION_PRESET_DAYLIGHT,
        'strong_blue': COLOR_CORRECTION_PRESET_STRONG_BLUE
    }
    
    if ACTIVE_COLOR_PRESET in presets:
        preset = presets[ACTIVE_COLOR_PRESET]
        CAMERA_CONTROLS.update(preset)
        print(f"✅ Applied color preset: {ACTIVE_COLOR_PRESET}")

# Apply preset on import
apply_color_preset()

# ===== QUICK ADJUSTMENT GUIDE =====
"""
IF STILL TOO BLUE:
1. Increase red gain: ColourGains = (1.4, 0.7) or (1.5, 0.65)
2. Try Tungsten mode: AwbMode = 1
3. Use strong_blue preset: ACTIVE_COLOR_PRESET = 'strong_blue'

IF TOO YELLOW/ORANGE:
1. Decrease red gain: ColourGains = (1.2, 0.85) or (1.1, 0.9)
2. Try Indoor mode: AwbMode = 3
3. Use daylight preset: ACTIVE_COLOR_PRESET = 'daylight'

IF TOO DARK:
1. Increase brightness: Brightness = 0.15 or 0.2

IF TOO BRIGHT/WASHED OUT:
1. Decrease brightness: Brightness = 0.0 or -0.05
2. Increase contrast: Contrast = 1.2
"""