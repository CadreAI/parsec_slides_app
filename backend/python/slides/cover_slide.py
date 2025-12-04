"""
Cover slide creation for Google Slides presentations
"""
from typing import List, Dict, Any
from .slide_constants import SLIDE_WIDTH_EMU, SLIDE_HEIGHT_EMU, PARSEC_BLUE, hex_to_rgb_color


def create_cover_slide_requests(cover_slide_id: str) -> List[Dict[str, Any]]:
    """Create cover slide requests (ported from coverSlide.ts)"""
    slide_width_emu = SLIDE_WIDTH_EMU
    slide_height_emu = SLIDE_HEIGHT_EMU
    header_height_emu = 1500000  # ~1.64 inches
    
    requests = [
        # Create cover slide
        {
            'createSlide': {
                'objectId': cover_slide_id,
                'slideLayoutReference': {'predefinedLayout': 'BLANK'},
                'insertionIndex': 0
            }
        },
        # White header background
        {
            'createShape': {
                'objectId': 'header_background',
                'shapeType': 'RECTANGLE',
                'elementProperties': {
                    'pageObjectId': cover_slide_id,
                    'size': {
                        'width': {'magnitude': slide_width_emu, 'unit': 'EMU'},
                        'height': {'magnitude': header_height_emu, 'unit': 'EMU'}
                    },
                    'transform': {
                        'scaleX': 1, 'scaleY': 1, 'translateX': 0, 'translateY': 0, 'unit': 'EMU'
                    }
                }
            }
        },
        {
            'updateShapeProperties': {
                'objectId': 'header_background',
                'shapeProperties': {
                    'shapeBackgroundFill': {
                        'solidFill': {
                            'color': {'rgbColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}
                        }
                    },
                    'outline': {'propertyState': 'NOT_RENDERED'}
                },
                'fields': 'shapeBackgroundFill.solidFill.color,outline'
            }
        },
        # Parsec logo circle (left)
        {
            'createShape': {
                'objectId': 'parsec_logo_circle',
                'shapeType': 'ELLIPSE',
                'elementProperties': {
                    'pageObjectId': cover_slide_id,
                    'size': {'width': {'magnitude': 800000, 'unit': 'EMU'}, 'height': {'magnitude': 800000, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': 1000000, 'translateY': 350000, 'unit': 'EMU'}
                }
            }
        },
        {
            'updateShapeProperties': {
                'objectId': 'parsec_logo_circle',
                'shapeProperties': {
                    'shapeBackgroundFill': {'solidFill': {'color': {'rgbColor': PARSEC_BLUE}}},
                    'outline': {'propertyState': 'NOT_RENDERED'}
                },
                'fields': 'shapeBackgroundFill.solidFill.color,outline'
            }
        },
        # Parsec text
        {
            'createShape': {
                'objectId': 'parsec_text',
                'shapeType': 'TEXT_BOX',
                'elementProperties': {
                    'pageObjectId': cover_slide_id,
                    'size': {'width': {'magnitude': 2000000, 'unit': 'EMU'}, 'height': {'magnitude': 600000, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': 2000000, 'translateY': 500000, 'unit': 'EMU'}
                }
            }
        },
        {
            'insertText': {'objectId': 'parsec_text', 'text': 'parsec\neducation', 'insertionIndex': 0}
        },
        {
            'updateTextStyle': {
                'objectId': 'parsec_text',
                'style': {'fontSize': {'magnitude': 20, 'unit': 'PT'}, 'bold': True, 'foregroundColor': hex_to_rgb_color('0094bd')},
                'fields': 'fontSize,bold,foregroundColor',
                'textRange': {'type': 'FIXED_RANGE', 'startIndex': 0, 'endIndex': 6}
            }
        },
        {
            'updateTextStyle': {
                'objectId': 'parsec_text',
                'style': {'fontSize': {'magnitude': 16, 'unit': 'PT'}, 'foregroundColor': hex_to_rgb_color('0094bd')},
                'fields': 'fontSize,foregroundColor',
                'textRange': {'type': 'FIXED_RANGE', 'startIndex': 7, 'endIndex': 16}
            }
        },
        {
            'updateShapeProperties': {
                'objectId': 'parsec_text',
                'shapeProperties': {'outline': {'propertyState': 'NOT_RENDERED'}},
                'fields': 'outline'
            }
        },
        # School logo circle (right)
        {
            'createShape': {
                'objectId': 'school_logo_circle',
                'shapeType': 'ELLIPSE',
                'elementProperties': {
                    'pageObjectId': cover_slide_id,
                    'size': {'width': {'magnitude': 1000000, 'unit': 'EMU'}, 'height': {'magnitude': 1000000, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': slide_width_emu - 2000000, 'translateY': 250000, 'unit': 'EMU'}
                }
            }
        },
        {
            'updateShapeProperties': {
                'objectId': 'school_logo_circle',
                'shapeProperties': {
                    'shapeBackgroundFill': {'solidFill': {'color': {'rgbColor': {'red': 0.8, 'green': 0.8, 'blue': 0.8}}}},
                    'outline': {'propertyState': 'NOT_RENDERED'}
                },
                'fields': 'shapeBackgroundFill.solidFill.color,outline'
            }
        },
        # Header separator line
        {
            'createShape': {
                'objectId': 'header_separator',
                'shapeType': 'RECTANGLE',
                'elementProperties': {
                    'pageObjectId': cover_slide_id,
                    'size': {'width': {'magnitude': slide_width_emu, 'unit': 'EMU'}, 'height': {'magnitude': 20000, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': 0, 'translateY': header_height_emu - 10000, 'unit': 'EMU'}
                }
            }
        },
        {
            'updateShapeProperties': {
                'objectId': 'header_separator',
                'shapeProperties': {
                    'shapeBackgroundFill': {'solidFill': {'color': {'rgbColor': {'red': 0.0, 'green': 0.0, 'blue': 0.0}}}},
                    'outline': {'propertyState': 'NOT_RENDERED'}
                },
                'fields': 'shapeBackgroundFill.solidFill.color,outline'
            }
        },
        # Blue content background
        {
            'createShape': {
                'objectId': 'content_background',
                'shapeType': 'RECTANGLE',
                'elementProperties': {
                    'pageObjectId': cover_slide_id,
                    'size': {'width': {'magnitude': slide_width_emu, 'unit': 'EMU'}, 'height': {'magnitude': slide_height_emu - header_height_emu, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': 0, 'translateY': header_height_emu, 'unit': 'EMU'}
                }
            }
        },
        {
            'updateShapeProperties': {
                'objectId': 'content_background',
                'shapeProperties': {
                    'shapeBackgroundFill': {'solidFill': {'color': {'rgbColor': PARSEC_BLUE}}},
                    'outline': {'propertyState': 'NOT_RENDERED'}
                },
                'fields': 'shapeBackgroundFill.solidFill.color,outline'
            }
        },
        # Title: "Quarterly Insights"
        {
            'createShape': {
                'objectId': 'cover_title',
                'shapeType': 'TEXT_BOX',
                'elementProperties': {
                    'pageObjectId': cover_slide_id,
                    'size': {'width': {'magnitude': 8000000, 'unit': 'EMU'}, 'height': {'magnitude': 2000000, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': (slide_width_emu - 8000000) / 2, 'translateY': header_height_emu + 500000, 'unit': 'EMU'}
                }
            }
        },
        {
            'insertText': {'objectId': 'cover_title', 'text': 'Quarterly Insights', 'insertionIndex': 0}
        },
        {
            'updateTextStyle': {
                'objectId': 'cover_title',
                'style': {'fontSize': {'magnitude': 60, 'unit': 'PT'}, 'bold': True, 'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}}},
                'fields': 'fontSize,bold,foregroundColor',
                'textRange': {'type': 'ALL'}
            }
        },
        {
            'updateParagraphStyle': {
                'objectId': 'cover_title',
                'style': {'alignment': 'CENTER'},
                'fields': 'alignment',
                'textRange': {'type': 'ALL'}
            }
        },
        {
            'updateShapeProperties': {
                'objectId': 'cover_title',
                'shapeProperties': {'outline': {'propertyState': 'NOT_RENDERED'}},
                'fields': 'outline'
            }
        },
        # Subtitle
        {
            'createShape': {
                'objectId': 'cover_subtitle',
                'shapeType': 'TEXT_BOX',
                'elementProperties': {
                    'pageObjectId': cover_slide_id,
                    'size': {'width': {'magnitude': 9000000, 'unit': 'EMU'}, 'height': {'magnitude': 1000000, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': (slide_width_emu - 9000000) / 2, 'translateY': header_height_emu + 5500000, 'unit': 'EMU'}
                }
            }
        },
        {
            'insertText': {'objectId': 'cover_subtitle', 'text': 'Making data meaningful with a true partner in education', 'insertionIndex': 0}
        },
        {
            'updateTextStyle': {
                'objectId': 'cover_subtitle',
                'style': {'fontSize': {'magnitude': 24, 'unit': 'PT'}, 'italic': True, 'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}}},
                'fields': 'fontSize,italic,foregroundColor',
                'textRange': {'type': 'ALL'}
            }
        },
        {
            'updateParagraphStyle': {
                'objectId': 'cover_subtitle',
                'style': {'alignment': 'CENTER'},
                'fields': 'alignment',
                'textRange': {'type': 'ALL'}
            }
        },
        {
            'updateShapeProperties': {
                'objectId': 'cover_subtitle',
                'shapeProperties': {'outline': {'propertyState': 'NOT_RENDERED'}},
                'fields': 'outline'
            }
        },
        # Date: "JUNE 10, 2025"
        {
            'createShape': {
                'objectId': 'cover_date',
                'shapeType': 'TEXT_BOX',
                'elementProperties': {
                    'pageObjectId': cover_slide_id,
                    'size': {'width': {'magnitude': 3000000, 'unit': 'EMU'}, 'height': {'magnitude': 800000, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': (slide_width_emu - 3000000) / 2, 'translateY': header_height_emu + 2000000, 'unit': 'EMU'}
                }
            }
        },
        {
            'insertText': {'objectId': 'cover_date', 'text': 'JUNE 10, 2025', 'insertionIndex': 0}
        },
        {
            'updateTextStyle': {
                'objectId': 'cover_date',
                'style': {'fontSize': {'magnitude': 18, 'unit': 'PT'}, 'italic': True, 'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}}},
                'fields': 'fontSize,italic,foregroundColor',
                'textRange': {'type': 'ALL'}
            }
        },
        {
            'updateParagraphStyle': {
                'objectId': 'cover_date',
                'style': {'alignment': 'CENTER'},
                'fields': 'alignment',
                'textRange': {'type': 'ALL'}
            }
        },
        {
            'updateShapeProperties': {
                'objectId': 'cover_date',
                'shapeProperties': {'outline': {'propertyState': 'NOT_RENDERED'}},
                'fields': 'outline'
            }
        }
    ]
    
    return requests

