


# ============================================================================
# STEP 2: Add these settings to config.py
# ============================================================================
"""
Add these lines to your config.py:
"""

# ===== ENHANCED CAMERA SETTINGS =====
"""
Quality Modes:
- 'high': Best quality, natural image, ~18-22 FPS (800x600)
- 'balanced': Good quality, smooth video, ~22-28 FPS (640x480) [RECOMMENDED]
- 'performance': Maximum FPS, ~28-32 FPS (320x240)
"""
CAMERA_QUALITY_MODE = 'balanced'  # Change to 'high' or 'performance' as needed

# Image Processing Preset
"""
- 'natural_hq': Best quality with denoising and enhancement
- 'balanced': Light processing for good quality [RECOMMENDED]
- 'performance': No processing for maximum speed
"""
IMAGE_PROCESSING_PRESET = 'balanced'

# Camera warmup time (reduced for faster startup)
CAMERA_WARMUP_TIME = 1  # seconds

# YOLO detection interval (optimized)
DETECTION_INTERVAL = 0.12  # Run YOLO every 120ms for better performance

# Video streaming quality
VIDEO_QUALITY = 92  # JPEG compression (85-95 recommended)


# ============================================================================
# INTEGRATION COMPLETE
# ============================================================================
"""
USAGE:
1. Replace camera/camera_yolo.py with the enhanced version above
2. Add the new settings to config.py
3. Restart your application:
   cd ~/cybercrawl
   source venv/bin/activate
   python app.py

EXPECTED IMPROVEMENTS:
- FPS: 3-5 fps → 22-28 fps (balanced mode)
- Quality: Natural colors, better lighting, reduced noise
- Latency: Lower lag, smoother video
- Processing: Optimized for Raspberry Pi

QUALITY MODES:
# For best quality (photography/recording):
CAMERA_QUALITY_MODE = 'high'
IMAGE_PROCESSING_PRESET = 'natural_hq'

# For balanced use (RECOMMENDED):
CAMERA_QUALITY_MODE = 'balanced'
IMAGE_PROCESSING_PRESET = 'balanced'

# For maximum FPS (fast movement):
CAMERA_QUALITY_MODE = 'performance'
IMAGE_PROCESSING_PRESET = 'performance'
"""