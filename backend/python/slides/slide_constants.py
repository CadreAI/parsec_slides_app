"""
Constants and utility functions for Google Slides creation
"""
from typing import Dict, Any

# EMU unit conversion (1 EMU = 1/914400 inch)
EMU_PER_INCH = 914400
SLIDE_WIDTH_EMU = 10 * EMU_PER_INCH
SLIDE_HEIGHT_EMU = (SLIDE_WIDTH_EMU * 9) / 16  # 16:9 aspect ratio

# Parsec blue color (#0094bd)
PARSEC_BLUE = {
    'red': 0.0,
    'green': 0.5803921568627451,
    'blue': 0.7411764705882353
}


def hex_to_rgb_color(hex_str: str) -> Dict[str, Any]:
    """Convert hex color to RGB format for Slides API"""
    clean_hex = hex_str.replace('#', '')
    r = int(clean_hex[0:2], 16) / 255.0
    g = int(clean_hex[2:4], 16) / 255.0
    b = int(clean_hex[4:6], 16) / 255.0
    return {
        'opaqueColor': {
            'rgbColor': {'red': r, 'green': g, 'blue': b}
        }
    }

