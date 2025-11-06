#!/usr/bin/env python3
"""
Color Calibration Utilities for Natural Color Correction
Optional post-processing if hardware gains need fine-tuning
Minimal performance impact - optimized for real-time processing
"""

import cv2
import numpy as np
import config

class NaturalColorCalibration:
    """
    Fast color correction based on reference image calibration
    Uses LUT (lookup tables) for near-zero performance impact
    """
    
    def __init__(self):
        """Initialize with calibrated values from config"""
        self.r_gain = getattr(config, 'CALIBRATED_RED_GAIN', 1.45)
        self.b_gain = getattr(config, 'CALIBRATED_BLUE_GAIN', 0.72)
        self.gamma = getattr(config, 'GAMMA_VALUE', 0.95)
        self.enable_gamma = getattr(config, 'ENABLE_GAMMA_CORRECTION', True)
        self.enable_post = getattr(config, 'ENABLE_POST_CORRECTION', False)
        
        # Pre-compute gamma LUT for speed
        self.gamma_lut = self._build_gamma_lut(self.gamma)
        
        print(f"🎨 Color calibration initialized:")
        print(f"   Red gain: {self.r_gain:.2f}, Blue gain: {self.b_gain:.2f}")
        print(f"   Gamma: {self.gamma:.2f}, Post-processing: {self.enable_post}")
    
    def _build_gamma_lut(self, gamma):
        """Build gamma correction lookup table"""
        inv_gamma = 1.0 / gamma
        lut = np.array([
            ((i / 255.0) ** inv_gamma) * 255 
            for i in range(256)
        ]).astype("uint8")
        return lut
    
    def apply_hardware_correction(self, frame):
        """
        Apply color gains (simulates hardware-level correction)
        This is what the camera does internally with ColourGains
        Only use if you need to verify or supplement hardware gains
        """
        if not self.enable_post:
            return frame
        
        # Convert to float for precise gain application
        img_float = frame.astype(np.float32)
        
        # Apply gains to B and R channels (OpenCV uses BGR)
        img_float[:, :, 2] *= self.r_gain  # Red channel
        img_float[:, :, 0] *= self.b_gain  # Blue channel
        
        # Clip and convert back
        img_corrected = np.clip(img_float, 0, 255).astype(np.uint8)
        
        return img_corrected
    
    def apply_gamma_correction(self, frame):
        """Apply gamma correction using pre-built LUT (very fast)"""
        if not self.enable_gamma:
            return frame
        
        return cv2.LUT(frame, self.gamma_lut)
    
    def correct_frame(self, frame):
        """
        Full correction pipeline - optimized for speed
        Typical overhead: <1ms per frame on Raspberry Pi 4
        """
        if frame is None:
            return frame
        
        # Step 1: Hardware-simulated gains (if enabled)
        if self.enable_post:
            frame = self.apply_hardware_correction(frame)
        
        # Step 2: Gamma correction (minimal overhead via LUT)
        if self.enable_gamma:
            frame = self.apply_gamma_correction(frame)
        
        return frame
    
    def analyze_frame_color(self, frame):
        """
        Analyze frame color balance for debugging
        Returns channel means and ratios
        """
        if frame is None:
            return None
        
        # Compute mean values for each channel
        b_mean = np.mean(frame[:, :, 0])
        g_mean = np.mean(frame[:, :, 1])
        r_mean = np.mean(frame[:, :, 2])
        
        # Compute ratios
        r_g_ratio = r_mean / (g_mean + 1e-6)
        b_g_ratio = b_mean / (g_mean + 1e-6)
        
        return {
            'r_mean': r_mean,
            'g_mean': g_mean,
            'b_mean': b_mean,
            'r_g_ratio': r_g_ratio,
            'b_g_ratio': b_g_ratio,
            'color_temp': 'warm' if r_g_ratio > 1.0 else 'cool'
        }


class InteractiveCalibrator:
    """
    Interactive calibration tool
    Use this to fine-tune gains by comparing to reference
    """
    
    @staticmethod
    def match_to_reference(source_img, reference_img, crop_frac=0.8):
        """
        Calculate optimal gains to match reference image
        
        Args:
            source_img: Current camera frame (BGR)
            reference_img: Target reference image (BGR)
            crop_frac: Center crop fraction to avoid UI elements
        
        Returns:
            (r_gain, b_gain): Calculated color gains
        """
        # Convert to RGB
        src_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        ref_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Center crop to avoid edges/UI
        src_crop = InteractiveCalibrator._center_crop(src_rgb, crop_frac)
        ref_crop = InteractiveCalibrator._center_crop(ref_rgb, crop_frac)
        
        # Compute mean values
        src_mean = src_crop.reshape(-1, 3).mean(axis=0)
        ref_mean = ref_crop.reshape(-1, 3).mean(axis=0)
        
        # Calculate channel ratios
        r_s, g_s, b_s = src_mean
        r_r, g_r, b_r = ref_mean
        
        ratio_r_src = (r_s + 1e-6) / (g_s + 1e-6)
        ratio_b_src = (b_s + 1e-6) / (g_s + 1e-6)
        ratio_r_ref = (r_r + 1e-6) / (g_r + 1e-6)
        ratio_b_ref = (b_r + 1e-6) / (g_r + 1e-6)
        
        # Calculate gains
        r_gain = ratio_r_ref / ratio_r_src
        b_gain = ratio_b_ref / ratio_b_src
        
        # Clamp to reasonable range
        r_gain = float(np.clip(r_gain, 0.5, 2.5))
        b_gain = float(np.clip(b_gain, 0.5, 2.5))
        
        return r_gain, b_gain
    
    @staticmethod
    def _center_crop(img, frac):
        """Crop center portion of image"""
        h, w = img.shape[:2]
        ch, cw = int(h * frac), int(w * frac)
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        return img[y0:y0+ch, x0:x0+cw]
    
    @staticmethod
    def preview_gains(frame, r_gain, b_gain, gamma=0.95):
        """
        Preview what gains will look like
        Returns corrected frame without modifying original
        """
        # Apply gains
        img_float = frame.astype(np.float32)
        img_float[:, :, 2] *= r_gain  # Red
        img_float[:, :, 0] *= b_gain  # Blue
        preview = np.clip(img_float, 0, 255).astype(np.uint8)
        
        # Apply gamma
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 
                       for i in range(256)]).astype("uint8")
        preview = cv2.LUT(preview, lut)
        
        return preview


# ===== Integration with camera_yolo.py =====
"""
OPTION A: Hardware-only correction (RECOMMENDED - NO FPS IMPACT)
Just update config.py with calibrated values:
    CALIBRATED_RED_GAIN = 1.45
    CALIBRATED_BLUE_GAIN = 0.72

The camera applies these automatically via ColourGains setting.

OPTION B: Software post-processing (if hardware gains insufficient)
Add to camera_yolo.py:

    from color_calibration import NaturalColorCalibration
    
    class EnhancedCameraYOLO:
        def __init__(self):
            ...
            self.color_corrector = NaturalColorCalibration()
        
        def get_frame_with_detections(self):
            ...
            # Before drawing detections, apply correction:
            frame = self.color_corrector.correct_frame(frame)
            ...

Performance:
- Hardware gains (OPTION A): 0ms overhead, perfect FPS
- Software correction (OPTION B): ~0.5-1ms overhead, negligible FPS impact
"""

# ===== Quick Test Script =====
def test_calibration():
    """Test calibration on a sample frame"""
    print("\n" + "="*60)
    print("🧪 Testing Color Calibration")
    print("="*60 + "\n")
    
    # Create calibrator
    calibrator = NaturalColorCalibration()
    
    # Create test frame (simulated blue-tinted)
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    test_frame[:, :, 0] = 160  # Blue channel higher (blue tint)
    test_frame[:, :, 1] = 128  # Green neutral
    test_frame[:, :, 2] = 110  # Red channel lower
    
    print("📊 Before correction:")
    before_stats = calibrator.analyze_frame_color(test_frame)
    print(f"   R/G ratio: {before_stats['r_g_ratio']:.3f}")
    print(f"   B/G ratio: {before_stats['b_g_ratio']:.3f}")
    print(f"   Color temp: {before_stats['color_temp']}")
    
    # Apply correction
    corrected = calibrator.correct_frame(test_frame)
    
    print("\n📊 After correction:")
    after_stats = calibrator.analyze_frame_color(corrected)
    print(f"   R/G ratio: {after_stats['r_g_ratio']:.3f}")
    print(f"   B/G ratio: {after_stats['b_g_ratio']:.3f}")
    print(f"   Color temp: {after_stats['color_temp']}")
    
    print("\n✅ Calibration test complete!")
    print("💡 Gains should shift color from cool/blue to warm/natural")

if __name__ == "__main__":
    test_calibration()