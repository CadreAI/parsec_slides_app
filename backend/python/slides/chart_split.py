"""
Cover slide and section divider slide creation for Google Slides presentations
"""
from typing import List, Dict, Any, Optional
from .slide_constants import SLIDE_WIDTH_EMU, SLIDE_HEIGHT_EMU, PARSEC_BLUE, hex_to_rgb_color


def hex_to_rgb_dict(hex_str: str) -> Dict[str, float]:
    """Convert hex color string to RGB dict for Slides API"""
    clean_hex = hex_str.replace('#', '')
    r = int(clean_hex[0:2], 16) / 255.0
    g = int(clean_hex[2:4], 16) / 255.0
    b = int(clean_hex[4:6], 16) / 255.0
    return {'red': r, 'green': g, 'blue': b}


def create_section_divider_slide_requests(
    slide_object_id: str,
    test_type: str,
    sections: List[str],
    insertion_index: int = 0,
    theme_color: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Create a section divider slide with fully blue background.
    Displays test type and sections separated.
    
    Args:
        slide_object_id: Unique ID for the slide
        test_type: Type of test (NWEA, iReady, STAR)
        sections: List of section names/numbers to display (e.g., ["Section 1", "Section 2", "Section 3"])
        insertion_index: Index where to insert the slide
    
    Returns:
        List of Google Slides API requests
    """
    # Use provided theme_color or default to Parsec blue
    # Handle both None and empty string cases
    final_theme_color = theme_color if theme_color and theme_color.strip() else '#0094bd'
    
    slide_width_emu = SLIDE_WIDTH_EMU
    slide_height_emu = SLIDE_HEIGHT_EMU
    requests = []
    
    # Create blank slide
    requests.append({
        'createSlide': {
            'objectId': slide_object_id,
            'slideLayoutReference': {'predefinedLayout': 'BLANK'},
            'insertionIndex': insertion_index
        }
    })
    
    # Full blue background covering entire slide
    requests.append({
        'createShape': {
            'objectId': f'{slide_object_id}_background',
            'shapeType': 'RECTANGLE',
            'elementProperties': {
                'pageObjectId': slide_object_id,
                'size': {
                    'width': {'magnitude': slide_width_emu, 'unit': 'EMU'},
                    'height': {'magnitude': slide_height_emu, 'unit': 'EMU'}
                },
                'transform': {
                    'scaleX': 1, 'scaleY': 1, 'translateX': 0, 'translateY': 0, 'unit': 'EMU'
                }
            }
        }
    })
    requests.append({
        'updateShapeProperties': {
            'objectId': f'{slide_object_id}_background',
            'shapeProperties': {
                'shapeBackgroundFill': {'solidFill': {'color': {'rgbColor': hex_to_rgb_dict(final_theme_color)}}},
                'outline': {'propertyState': 'NOT_RENDERED'}
            },
            'fields': 'shapeBackgroundFill.solidFill.color,outline'
        }
    })
    
    # Test type title at top (vertically centered in upper portion)
    test_type_object_id = f'{slide_object_id}_test_type'
    test_type_y = 600000  # Position from top
    test_type_width = 9000000
    test_type_height = 1500000  # Reduced height for tighter spacing
    requests.append({
        'createShape': {
            'objectId': test_type_object_id,
            'shapeType': 'TEXT_BOX',
            'elementProperties': {
                'pageObjectId': slide_object_id,
                'size': {
                    'width': {'magnitude': test_type_width, 'unit': 'EMU'},
                    'height': {'magnitude': test_type_height, 'unit': 'EMU'}
                },
                'transform': {
                    'scaleX': 1, 'scaleY': 1,
                    'translateX': (slide_width_emu - test_type_width) / 2,
                    'translateY': test_type_y,
                    'unit': 'EMU'
                }
            }
        }
    })
    requests.append({
        'insertText': {'objectId': test_type_object_id, 'text': test_type, 'insertionIndex': 0 }
    })
    requests.append({
        'updateTextStyle': {
            'objectId': test_type_object_id,
            'style': {
                'fontSize': {'magnitude': 72, 'unit': 'PT'},
                'bold': True,
                'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}}
            },
            'fields': 'fontSize,bold,foregroundColor',
            'textRange': {'type': 'ALL'}
        }
    })
    requests.append({
        'updateParagraphStyle': {
            'objectId': test_type_object_id,
            'style': {'alignment': 'CENTER'},
            'fields': 'alignment',
            'textRange': {'type': 'ALL'}
        }
    })
    requests.append({
        'updateShapeProperties': {
            'objectId': test_type_object_id,
            'shapeProperties': {'outline': {'propertyState': 'NOT_RENDERED'}},
            'fields': 'outline'
        }
    })
    
    # Sections displayed vertically, directly below test type
    num_sections = len(sections)
    if num_sections > 0:
        # Start sections right below the test type with minimal gap
        section_start_y = test_type_y + test_type_height + 200000  # Small gap between title and first section
        available_height = slide_height_emu - section_start_y - 800000  # Leave bottom margin
        
        # Ensure available_height is positive and set minimum section height
        if available_height <= 0:
            available_height = 2000000  # Fallback minimum height
        
        # Calculate section height with minimum guarantee
        min_section_height = 800000  # Minimum section height
        calculated_height = available_height / num_sections if num_sections > 0 else min_section_height
        section_height = max(min_section_height, min(1800000, calculated_height))  # Between 800K and 1.8M EMU
        
        # Adjust spacing if needed to fit
        section_spacing = 200000  # Reduced space between sections
        total_needed = (section_height * num_sections) + (section_spacing * (num_sections - 1))
        if total_needed > available_height:
            # Reduce spacing if needed
            if num_sections > 1:
                section_spacing = max(100000, (available_height - (section_height * num_sections)) / (num_sections - 1))
        
        for i, section in enumerate(sections):
            section_object_id = f'{slide_object_id}_section_{i}'
            section_y = section_start_y + i * (section_height + section_spacing)
            
            # Create section text box
            requests.append({
                'createShape': {
                    'objectId': section_object_id,
                    'shapeType': 'TEXT_BOX',
                    'elementProperties': {
                        'pageObjectId': slide_object_id,
                        'size': {
                            'width': {'magnitude': 8000000, 'unit': 'EMU'},
                            'height': {'magnitude': section_height, 'unit': 'EMU'}
                        },
                        'transform': {
                            'scaleX': 1, 'scaleY': 1,
                            'translateX': (slide_width_emu - 8000000) / 2,
                            'translateY': section_y,
                            'unit': 'EMU'
                        }
                    }
                }
            })
            requests.append({
                'insertText': {'objectId': section_object_id, 'text': section, 'insertionIndex': 0}
            })
            requests.append({
                'updateTextStyle': {
                    'objectId': section_object_id,
                    'style': {
                        'fontSize': {'magnitude': 48, 'unit': 'PT'},
                        'bold': True,
                        'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}}
                    },
                    'fields': 'fontSize,bold,foregroundColor',
                    'textRange': {'type': 'ALL'}
                }
            })
            requests.append({
                'updateParagraphStyle': {
                    'objectId': section_object_id,
                    'style': {'alignment': 'CENTER'},
                    'fields': 'alignment',
                    'textRange': {'type': 'ALL'}
                }
            })
            requests.append({
                'updateShapeProperties': {
                    'objectId': section_object_id,
                    'shapeProperties': {'outline': {'propertyState': 'NOT_RENDERED'}},
                    'fields': 'outline'
                }
            })
            
            # Add separator line below each section (except the last one)
            if i < num_sections - 1:
                separator_object_id = f'{slide_object_id}_separator_{i}'
                separator_y = section_y + section_height + section_spacing / 2 - 10000
                requests.append({
                    'createShape': {
                        'objectId': separator_object_id,
                        'shapeType': 'RECTANGLE',
                        'elementProperties': {
                            'pageObjectId': slide_object_id,
                            'size': {
                                'width': {'magnitude': 6000000, 'unit': 'EMU'},
                                'height': {'magnitude': 20000, 'unit': 'EMU'}
                            },
                            'transform': {
                                'scaleX': 1, 'scaleY': 1,
                                'translateX': (slide_width_emu - 6000000) / 2,
                                'translateY': separator_y,
                                'unit': 'EMU'
                            }
                        }
                    }
                })
                requests.append({
                    'updateShapeProperties': {
                        'objectId': separator_object_id,
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
                })
    
    return requests
