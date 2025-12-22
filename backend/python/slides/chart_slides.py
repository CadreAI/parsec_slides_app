"""
Chart slide creation functions for Google Slides presentations
"""
from typing import List, Dict, Optional, Any
from .slide_constants import SLIDE_WIDTH_EMU, SLIDE_HEIGHT_EMU, PARSEC_BLUE, hex_to_rgb_color


def hex_to_rgb_dict(hex_str: str) -> Dict[str, float]:
    """Convert hex color string to RGB dict for Slides API"""
    clean_hex = hex_str.replace('#', '')
    r = int(clean_hex[0:2], 16) / 255.0
    g = int(clean_hex[2:4], 16) / 255.0
    b = int(clean_hex[4:6], 16) / 255.0
    return {'red': r, 'green': g, 'blue': b}


def create_chart_slide_request(presentation_id: str, slide_object_id: str, insertion_index: int) -> Dict[str, Any]:
    """Create a blank slide for charts"""
    return {
        'createSlide': {
            'objectId': slide_object_id,
            'slideLayoutReference': {'predefinedLayout': 'BLANK'},
            'insertionIndex': insertion_index
        }
    }


def create_single_chart_slide_requests(
    slide_object_id: str,
    chart_url: str,
    title: str,
    slide_width_emu: float = SLIDE_WIDTH_EMU,
    slide_height_emu: float = SLIDE_HEIGHT_EMU,
    start_index: int = 0,
    summary: Optional[str] = None,
    theme_color: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Create requests for a single chart slide with maximized chart size"""
    # Use provided theme_color or default to Parsec blue
    # Handle both None and empty string cases
    final_theme_color = theme_color if theme_color and theme_color.strip() else '#0094bd'
    
    requests = []
    
    if not chart_url:
        return requests
    
    # Minimal margins for maximum chart size
    margin_emu = 50000
    top_bar_height_emu = 150000
    
    # Top bar
    top_bar_object_id = f'TopBar_{start_index}'
    requests.extend([
        {
            'createShape': {
                'objectId': top_bar_object_id,
                'shapeType': 'RECTANGLE',
                'elementProperties': {
                    'pageObjectId': slide_object_id,
                    'size': {'width': {'magnitude': slide_width_emu, 'unit': 'EMU'}, 'height': {'magnitude': top_bar_height_emu, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': 0, 'translateY': 0, 'unit': 'EMU'}
                }
            }
        },
        {
            'updateShapeProperties': {
                'objectId': top_bar_object_id,
                'shapeProperties': {'shapeBackgroundFill': {'solidFill': {'color': {'rgbColor': hex_to_rgb_dict(final_theme_color)}}}},
                'fields': 'shapeBackgroundFill.solidFill.color'
            }
        }
    ])
    
    # Chart - maximize size to fill almost entire slide
    chart_y = top_bar_height_emu + 10000  # Minimal spacing after top bar
    
    # Calculate available dimensions with minimal margins
    bottom_margin_emu = 20000
    available_height = slide_height_emu - chart_y - bottom_margin_emu
    available_width = slide_width_emu - margin_emu * 2
    
    # Use maximum available width and height
    chart_width = available_width
    chart_height = available_height
    
    chart_object_id = f'Chart_{start_index}'
    chart_x = margin_emu
    
    requests.append({
        'createImage': {
            'objectId': chart_object_id,
            'url': chart_url,
            'elementProperties': {
                'pageObjectId': slide_object_id,
                'size': {'width': {'magnitude': chart_width, 'unit': 'EMU'}, 'height': {'magnitude': chart_height, 'unit': 'EMU'}},
                'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': chart_x, 'translateY': chart_y, 'unit': 'EMU'}
            }
        }
    })
    
    return requests

