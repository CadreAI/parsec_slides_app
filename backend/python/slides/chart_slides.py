"""
Chart slide creation functions for Google Slides presentations
"""
from typing import List, Dict, Optional, Any
from .slide_constants import SLIDE_WIDTH_EMU, SLIDE_HEIGHT_EMU, PARSEC_BLUE, hex_to_rgb_color


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
    summary: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Create requests for a single chart slide with title and summary"""
    requests = []
    
    if not chart_url:
        return requests
    
    # Reduced margins and spacing to maximize chart size
    margin_emu = 200000  # Reduced from 500000
    title_height_emu = 800000  # Reduced from 1200000
    top_bar_height_emu = 150000  # Reduced from 200000
    
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
                'shapeProperties': {'shapeBackgroundFill': {'solidFill': {'color': {'rgbColor': PARSEC_BLUE}}}},
                'fields': 'shapeBackgroundFill.solidFill.color'
            }
        }
    ])
    
    # Title - reduced spacing
    title_object_id = f'Title_{start_index}'
    title_width = 9000000
    title_x = (slide_width_emu - title_width) / 2
    title_y = top_bar_height_emu + 3000  # Reduced from 100000
    
    requests.extend([
        {
            'createShape': {
                'objectId': title_object_id,
                'shapeType': 'TEXT_BOX',
                'elementProperties': {
                    'pageObjectId': slide_object_id,
                    'size': {'width': {'magnitude': title_width, 'unit': 'EMU'}, 'height': {'magnitude': title_height_emu, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': title_x, 'translateY': title_y, 'unit': 'EMU'}
                }
            }
        },
        {
            'insertText': {'objectId': title_object_id, 'text': title, 'insertionIndex': 0}
        },
        {
            'updateTextStyle': {
                'objectId': title_object_id,
                'style': {'fontSize': {'magnitude': 19, 'unit': 'PT'}, 'bold': True},
                'fields': 'fontSize,bold',
                'textRange': {'type': 'ALL'}
            }
        },
        {
            'updateParagraphStyle': {
                'objectId': title_object_id,
                'style': {'alignment': 'CENTER'},
                'fields': 'alignment',
                'textRange': {'type': 'ALL'}
            }
        }
    ])
    
    # Chart - maximize size and move higher
    available_width = slide_width_emu - margin_emu * 2
    available_height = slide_height_emu - margin_emu - title_height_emu - top_bar_height_emu - 50000  # Minimal bottom margin
    # Use almost full available width and height
    chart_width = available_width - margin_emu  # Use most of width
    chart_height = available_height*1.10  # Use full available height
    
    chart_object_id = f'Chart_{start_index}'
    chart_x = margin_emu  # Start at margin
    chart_y = top_bar_height_emu + title_height_emu + 2000  # Move higher - minimal spacing after title
    
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


def create_dual_chart_slide_requests(
    slide_object_id: str,
    chart_url1: str,
    chart_url2: str,
    title: str,
    slide_width_emu: float = SLIDE_WIDTH_EMU,
    slide_height_emu: float = SLIDE_HEIGHT_EMU,
    start_index: int = 0,
    insight1: Optional[Dict[str, Any]] = None,
    insight2: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Create requests for a dual chart slide (math + reading side by side) with context text below each chart"""
    requests = []
    
    if not chart_url1 or not chart_url2:
        return requests
    
    # Reduced margins and spacing to maximize chart size
    margin_emu = 200000  # Reduced from 400000
    title_height_emu = 800000  # Reduced from 1200000
    chart_spacing_emu = 150000  # Reduced from 200000
    top_bar_height_emu = 150000  # Reduced from 200000
    
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
                'shapeProperties': {'shapeBackgroundFill': {'solidFill': {'color': {'rgbColor': PARSEC_BLUE}}}},
                'fields': 'shapeBackgroundFill.solidFill.color'
            }
        }
    ])
    
    # Title - reduced spacing
    title_object_id = f'Title_{start_index}'
    title_width = 9000000
    title_x = (slide_width_emu - title_width) / 2
    title_y = top_bar_height_emu + 50000  # Reduced from 100000
    
    requests.extend([
        {
            'createShape': {
                'objectId': title_object_id,
                'shapeType': 'TEXT_BOX',
                'elementProperties': {
                    'pageObjectId': slide_object_id,
                    'size': {'width': {'magnitude': title_width, 'unit': 'EMU'}, 'height': {'magnitude': title_height_emu, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': title_x, 'translateY': title_y, 'unit': 'EMU'}
                }
            }
        },
        {
            'insertText': {'objectId': title_object_id, 'text': title, 'insertionIndex': 0}
        },
        {
            'updateTextStyle': {
                'objectId': title_object_id,
                'style': {'fontSize': {'magnitude': 19, 'unit': 'PT'}, 'bold': True},
                'fields': 'fontSize,bold',
                'textRange': {'type': 'ALL'}
            }
        },
        {
            'updateParagraphStyle': {
                'objectId': title_object_id,
                'style': {'alignment': 'CENTER'},
                'fields': 'alignment',
                'textRange': {'type': 'ALL'}
            }
        }
    ])
    
    # Charts side by side - maximize size and move higher
    available_width = slide_width_emu - margin_emu * 2
    available_height = slide_height_emu - margin_emu - title_height_emu - top_bar_height_emu - 50000  # Minimal bottom margin
    chart_width = (available_width - chart_spacing_emu) / 2
    chart_height = available_height  # Use full available height
    
    # Move charts higher - minimal spacing after title
    chart1_object_id = f'Chart_{start_index}'
    chart1_x = margin_emu
    chart1_y = top_bar_height_emu + title_height_emu + 10000  # Move higher - minimal spacing
    
    chart2_object_id = f'Chart_{start_index + 1}'
    chart2_x = margin_emu + chart_width + chart_spacing_emu
    chart2_y = top_bar_height_emu + title_height_emu + 10000  # Move higher - minimal spacing
    
    requests.extend([
        {
            'createImage': {
                'objectId': chart1_object_id,
                'url': chart_url1,
                'elementProperties': {
                    'pageObjectId': slide_object_id,
                    'size': {'width': {'magnitude': chart_width, 'unit': 'EMU'}, 'height': {'magnitude': chart_height, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': chart1_x, 'translateY': chart1_y, 'unit': 'EMU'}
                }
            }
        },
        {
            'createImage': {
                'objectId': chart2_object_id,
                'url': chart_url2,
                'elementProperties': {
                    'pageObjectId': slide_object_id,
                    'size': {'width': {'magnitude': chart_width, 'unit': 'EMU'}, 'height': {'magnitude': chart_height, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': chart2_x, 'translateY': chart2_y, 'unit': 'EMU'}
                }
            }
        }
    ])
    
    return requests

