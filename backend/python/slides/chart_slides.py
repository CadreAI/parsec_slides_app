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
    
    margin_emu = 500000
    title_height_emu = 1200000
    summary_width_emu = 3000000
    top_bar_height_emu = 200000
    
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
    
    # Title
    title_object_id = f'Title_{start_index}'
    title_width = 9000000
    title_x = (slide_width_emu - title_width) / 2
    title_y = top_bar_height_emu + 100000
    
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
                'style': {'fontSize': {'magnitude': 28, 'unit': 'PT'}, 'bold': True},
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
    
    # Chart
    available_width = slide_width_emu - margin_emu * 2
    available_height = slide_height_emu - margin_emu * 2 - title_height_emu - top_bar_height_emu
    base_chart_width = available_width - summary_width_emu - margin_emu
    base_chart_height = available_height
    chart_width = base_chart_width * 1.25
    chart_height = base_chart_height * 1.25
    
    chart_object_id = f'Chart_{start_index}'
    chart_x = margin_emu * 0.5
    chart_y = top_bar_height_emu + title_height_emu + margin_emu * 0.3
    
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
    
    # Summary text box
    default_summary = '• In Reading, both 5th and 6th grade fell short of typical Fall to Spring growth targets.\n\n• In Math, 5th grade matched typical Fall to Spring growth targets while 6th graders exceeded typical growth.'
    summary_text = summary or default_summary
    is_default_text = not summary or summary == default_summary
    
    summary_object_id = f'Summary_{start_index}'
    requests.extend([
        {
            'createShape': {
                'objectId': summary_object_id,
                'shapeType': 'TEXT_BOX',
                'elementProperties': {
                    'pageObjectId': slide_object_id,
                    'size': {'width': {'magnitude': summary_width_emu, 'unit': 'EMU'}, 'height': {'magnitude': 4000000, 'unit': 'EMU'}},
                    'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': slide_width_emu - 3200000, 'translateY': top_bar_height_emu + title_height_emu + 100000, 'unit': 'EMU'}
                }
            }
        },
        {
            'insertText': {'objectId': summary_object_id, 'text': summary_text, 'insertionIndex': 0}
        },
        {
            'updateTextStyle': {
                'objectId': summary_object_id,
                'style': {'fontSize': {'magnitude': 12, 'unit': 'PT'}},
                'fields': 'fontSize',
                'textRange': {'type': 'ALL'}
            }
        },
        {
            'updateParagraphStyle': {
                'objectId': summary_object_id,
                'style': {'lineSpacing': 150, 'spaceAbove': {'magnitude': 6, 'unit': 'PT'}},
                'fields': 'lineSpacing,spaceAbove',
                'textRange': {'type': 'ALL'}
            }
        }
    ])
    
    # Only apply hardcoded styling if using default text
    if is_default_text:
        requests.extend([
            {
                'updateTextStyle': {
                    'objectId': summary_object_id,
                    'style': {'bold': True, 'foregroundColor': hex_to_rgb_color('0094bd')},
                    'fields': 'bold,foregroundColor',
                    'textRange': {'type': 'FIXED_RANGE', 'startIndex': 37, 'endIndex': 47}
                }
            },
            {
                'updateTextStyle': {
                    'objectId': summary_object_id,
                    'style': {'bold': True, 'foregroundColor': hex_to_rgb_color('0094bd')},
                    'fields': 'bold,foregroundColor',
                    'textRange': {'type': 'FIXED_RANGE', 'startIndex': 112, 'endIndex': 119}
                }
            },
            {
                'updateTextStyle': {
                    'objectId': summary_object_id,
                    'style': {'bold': True, 'foregroundColor': hex_to_rgb_color('0094bd')},
                    'fields': 'bold,foregroundColor',
                    'textRange': {'type': 'FIXED_RANGE', 'startIndex': 176, 'endIndex': 199}
                }
            }
        ])
    
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
    
    margin_emu = 400000
    title_height_emu = 1200000
    chart_spacing_emu = 200000
    top_bar_height_emu = 200000
    context_text_height_emu = 600000  # Height for context text box below each chart
    
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
    
    # Title
    title_object_id = f'Title_{start_index}'
    title_width = 9000000
    title_x = (slide_width_emu - title_width) / 2
    title_y = top_bar_height_emu + 100000
    
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
                'style': {'fontSize': {'magnitude': 28, 'unit': 'PT'}, 'bold': True},
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
    
    # Charts side by side - moved up by reducing margin
    available_width = slide_width_emu - margin_emu * 2
    # Reserve space for context text below charts
    available_height = slide_height_emu - margin_emu * 2 - title_height_emu - top_bar_height_emu - context_text_height_emu - 100000
    chart_width = (available_width - chart_spacing_emu) / 2
    chart_height = available_height
    
    # Move charts up by reducing Y position (reduce margin from margin_emu to margin_emu * 0.5)
    chart1_object_id = f'Chart_{start_index}'
    chart1_x = margin_emu
    chart1_y = top_bar_height_emu + title_height_emu + margin_emu * 0.5
    
    chart2_object_id = f'Chart_{start_index + 1}'
    chart2_x = margin_emu + chart_width + chart_spacing_emu
    chart2_y = top_bar_height_emu + title_height_emu + margin_emu * 0.5
    
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
    
    # Add context text boxes below each chart with the most important insight
    # Context for chart 1 (left chart)
    context1_text = ""
    if insight1 and insight1.get('insights'):
        insights_list = insight1.get('insights', [])
        if insights_list:
            # Use the first insight as the most important
            context1_text = insights_list[0]
    
    if context1_text:
        context1_object_id = f'Context1_{start_index}'
        context1_y = chart1_y + chart_height + 50000  # Position below chart 1
        requests.extend([
            {
                'createShape': {
                    'objectId': context1_object_id,
                    'shapeType': 'TEXT_BOX',
                    'elementProperties': {
                        'pageObjectId': slide_object_id,
                        'size': {'width': {'magnitude': chart_width, 'unit': 'EMU'}, 'height': {'magnitude': context_text_height_emu, 'unit': 'EMU'}},
                        'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': chart1_x, 'translateY': context1_y, 'unit': 'EMU'}
                    }
                }
            },
            {
                'insertText': {'objectId': context1_object_id, 'text': context1_text, 'insertionIndex': 0}
            },
            {
                'updateTextStyle': {
                    'objectId': context1_object_id,
                    'style': {'fontSize': {'magnitude': 12, 'unit': 'PT'}, 'italic': True},
                    'fields': 'fontSize,italic',
                    'textRange': {'type': 'ALL'}
                }
            },
            {
                'updateParagraphStyle': {
                    'objectId': context1_object_id,
                    'style': {'alignment': 'CENTER', 'lineSpacing': 120},
                    'fields': 'alignment,lineSpacing',
                    'textRange': {'type': 'ALL'}
                }
            }
        ])
    
    # Context for chart 2 (right chart)
    context2_text = ""
    if insight2 and insight2.get('insights'):
        insights_list = insight2.get('insights', [])
        if insights_list:
            # Use the first insight as the most important
            context2_text = insights_list[0]
    
    if context2_text:
        context2_object_id = f'Context2_{start_index}'
        context2_y = chart2_y + chart_height + 50000  # Position below chart 2
        requests.extend([
            {
                'createShape': {
                    'objectId': context2_object_id,
                    'shapeType': 'TEXT_BOX',
                    'elementProperties': {
                        'pageObjectId': slide_object_id,
                        'size': {'width': {'magnitude': chart_width, 'unit': 'EMU'}, 'height': {'magnitude': context_text_height_emu, 'unit': 'EMU'}},
                        'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': chart2_x, 'translateY': context2_y, 'unit': 'EMU'}
                    }
                }
            },
            {
                'insertText': {'objectId': context2_object_id, 'text': context2_text, 'insertionIndex': 0}
            },
            {
                'updateTextStyle': {
                    'objectId': context2_object_id,
                    'style': {'fontSize': {'magnitude': 12, 'unit': 'PT'}, 'italic': True},
                    'fields': 'fontSize,italic',
                    'textRange': {'type': 'ALL'}
                }
            },
            {
                'updateParagraphStyle': {
                    'objectId': context2_object_id,
                    'style': {'alignment': 'CENTER', 'lineSpacing': 120},
                    'fields': 'alignment,lineSpacing',
                    'textRange': {'type': 'ALL'}
                }
            }
        ])
    
    return requests

