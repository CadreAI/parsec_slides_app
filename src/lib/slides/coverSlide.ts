// EMU unit conversion (1 EMU = 1/914400 inch)
const EMU_PER_INCH = 914400
const SLIDE_WIDTH_EMU = 10 * EMU_PER_INCH

/**
 * Converts a hex color string to the required RGB format for the Slides API.
 */
function hexToRgbColor(hex: string) {
    const cleanHex = hex.replace('#', '')
    const r = parseInt(cleanHex.substring(0, 2), 16) / 255.0
    const g = parseInt(cleanHex.substring(2, 4), 16) / 255.0
    const b = parseInt(cleanHex.substring(4, 6), 16) / 255.0
    return {
        opaqueColor: { rgbColor: { red: r, green: g, blue: b } }
    }
}

/**
 * Creates the cover slide requests for the presentation.
 * Returns an array of batch update requests for the cover slide.
 */
export function createCoverSlideRequests(coverSlideId: string): any[] {
    const slideWidthEMU = SLIDE_WIDTH_EMU
    const slideHeightEMU = (slideWidthEMU * 9) / 16 // 16:9 aspect ratio
    const headerHeightEMU = 1500000 // ~1.64 inches for header

    return [
        // Cover slide - insert at index 0 (first slide)
        {
            createSlide: {
                objectId: coverSlideId,
                slideLayoutReference: {
                    predefinedLayout: 'BLANK'
                },
                insertionIndex: 0
            }
        },
        // White header background
        {
            createShape: {
                objectId: 'header_background',
                shapeType: 'RECTANGLE',
                elementProperties: {
                    pageObjectId: coverSlideId,
                    size: {
                        width: { magnitude: slideWidthEMU, unit: 'EMU' },
                        height: { magnitude: headerHeightEMU, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: 0,
                        translateY: 0,
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            updateShapeProperties: {
                objectId: 'header_background',
                shapeProperties: {
                    shapeBackgroundFill: {
                        solidFill: {
                            color: {
                                rgbColor: {
                                    red: 1.0,
                                    green: 1.0,
                                    blue: 1.0
                                }
                            }
                        }
                    },
                    outline: {
                        propertyState: 'NOT_RENDERED'
                    }
                },
                fields: 'shapeBackgroundFill.solidFill.color,outline'
            }
        },
        // Parsec Education logo placeholder (left side)
        {
            createShape: {
                objectId: 'parsec_logo_circle',
                shapeType: 'ELLIPSE',
                elementProperties: {
                    pageObjectId: coverSlideId,
                    size: {
                        width: { magnitude: 800000, unit: 'EMU' },
                        height: { magnitude: 800000, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: 1000000,
                        translateY: 350000,
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            updateShapeProperties: {
                objectId: 'parsec_logo_circle',
                shapeProperties: {
                    shapeBackgroundFill: {
                        solidFill: {
                            color: {
                                rgbColor: {
                                    red: 0.0, // Blue #0094bd
                                    green: 0.5803921568627451,
                                    blue: 0.7411764705882353
                                }
                            }
                        }
                    },
                    outline: {
                        propertyState: 'NOT_RENDERED'
                    }
                },
                fields: 'shapeBackgroundFill.solidFill.color,outline'
            }
        },
        // Parsec Education text (left side)
        {
            createShape: {
                objectId: 'parsec_text',
                shapeType: 'TEXT_BOX',
                elementProperties: {
                    pageObjectId: coverSlideId,
                    size: {
                        width: { magnitude: 2000000, unit: 'EMU' },
                        height: { magnitude: 600000, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: 2000000,
                        translateY: 500000,
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            insertText: {
                objectId: 'parsec_text',
                text: 'parsec\neducation',
                insertionIndex: 0
            }
        },
        // Format "parsec" text (dark blue) and "education" text (lighter blue)
        // Text: "parsec\neducation" (16 characters: parsec=6, \n=1, education=9)
        {
            updateTextStyle: {
                objectId: 'parsec_text',
                style: {
                    fontSize: { magnitude: 20, unit: 'PT' },
                    bold: true,
                    foregroundColor: hexToRgbColor('0094bd') // Dark blue for "parsec"
                },
                fields: 'fontSize,bold,foregroundColor',
                textRange: {
                    type: 'FIXED_RANGE',
                    startIndex: 0,
                    endIndex: 6 // "parsec" (indices 0-5)
                }
            }
        },
        {
            updateTextStyle: {
                objectId: 'parsec_text',
                style: {
                    fontSize: { magnitude: 16, unit: 'PT' },
                    foregroundColor: hexToRgbColor('0094bd') // Same blue but smaller for "education"
                },
                fields: 'fontSize,foregroundColor',
                textRange: {
                    type: 'FIXED_RANGE',
                    startIndex: 7, // After "parsec\n" (position 7)
                    endIndex: 16 // End of "education" (indices 7-15, total length is 16)
                }
            }
        },
        {
            updateShapeProperties: {
                objectId: 'parsec_text',
                shapeProperties: {
                    outline: {
                        propertyState: 'NOT_RENDERED'
                    }
                },
                fields: 'outline'
            }
        },
        // School district logo placeholder (right side)
        {
            createShape: {
                objectId: 'school_logo_circle',
                shapeType: 'ELLIPSE',
                elementProperties: {
                    pageObjectId: coverSlideId,
                    size: {
                        width: { magnitude: 1000000, unit: 'EMU' },
                        height: { magnitude: 1000000, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: slideWidthEMU - 2000000,
                        translateY: 250000,
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            updateShapeProperties: {
                objectId: 'school_logo_circle',
                shapeProperties: {
                    shapeBackgroundFill: {
                        solidFill: {
                            color: {
                                rgbColor: {
                                    red: 0.8,
                                    green: 0.8,
                                    blue: 0.8
                                }
                            }
                        }
                    },
                    outline: {
                        propertyState: 'NOT_RENDERED'
                    }
                },
                fields: 'shapeBackgroundFill.solidFill.color,outline'
            }
        },
        // Horizontal line separator (using a thin rectangle instead)
        {
            createShape: {
                objectId: 'header_separator',
                shapeType: 'RECTANGLE',
                elementProperties: {
                    pageObjectId: coverSlideId,
                    size: {
                        width: { magnitude: slideWidthEMU, unit: 'EMU' },
                        height: { magnitude: 20000, unit: 'EMU' } // Very thin line
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: 0,
                        translateY: headerHeightEMU - 10000,
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            updateShapeProperties: {
                objectId: 'header_separator',
                shapeProperties: {
                    shapeBackgroundFill: {
                        solidFill: {
                            color: {
                                rgbColor: {
                                    red: 0.0,
                                    green: 0.0,
                                    blue: 0.0
                                }
                            }
                        }
                    },
                    outline: {
                        propertyState: 'NOT_RENDERED'
                    }
                },
                fields: 'shapeBackgroundFill.solidFill.color,outline'
            }
        },
        // Blue main content background
        {
            createShape: {
                objectId: 'content_background',
                shapeType: 'RECTANGLE',
                elementProperties: {
                    pageObjectId: coverSlideId,
                    size: {
                        width: { magnitude: slideWidthEMU, unit: 'EMU' },
                        height: { magnitude: slideHeightEMU - headerHeightEMU, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: 0,
                        translateY: headerHeightEMU,
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            updateShapeProperties: {
                objectId: 'content_background',
                shapeProperties: {
                    shapeBackgroundFill: {
                        solidFill: {
                            color: {
                                rgbColor: {
                                    red: 0.0, // Blue #0094bd
                                    green: 0.5803921568627451,
                                    blue: 0.7411764705882353
                                }
                            }
                        }
                    },
                    outline: {
                        propertyState: 'NOT_RENDERED'
                    }
                },
                fields: 'shapeBackgroundFill.solidFill.color,outline'
            }
        },
        // Title: "Quarterly Insights"
        {
            createShape: {
                objectId: 'cover_title',
                shapeType: 'TEXT_BOX',
                elementProperties: {
                    pageObjectId: coverSlideId,
                    size: {
                        width: { magnitude: 8000000, unit: 'EMU' },
                        height: { magnitude: 2000000, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: (slideWidthEMU - 8000000) / 2,
                        translateY: headerHeightEMU + 500000, // Moved up significantly to avoid overlap
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            insertText: {
                objectId: 'cover_title',
                text: 'Quarterly Insights',
                insertionIndex: 0
            }
        },
        {
            updateTextStyle: {
                objectId: 'cover_title',
                style: {
                    fontSize: { magnitude: 60, unit: 'PT' },
                    bold: true,
                    foregroundColor: {
                        opaqueColor: {
                            rgbColor: {
                                red: 1.0,
                                green: 1.0,
                                blue: 1.0
                            }
                        }
                    }
                },
                fields: 'fontSize,bold,foregroundColor',
                textRange: { type: 'ALL' }
            }
        },
        {
            updateParagraphStyle: {
                objectId: 'cover_title',
                style: {
                    alignment: 'CENTER'
                },
                fields: 'alignment',
                textRange: { type: 'ALL' }
            }
        },
        {
            updateShapeProperties: {
                objectId: 'cover_title',
                shapeProperties: {
                    outline: {
                        propertyState: 'NOT_RENDERED'
                    }
                },
                fields: 'outline'
            }
        },
        // Subtitle: "Making data meaningful with a true partner in education"
        {
            createShape: {
                objectId: 'cover_subtitle',
                shapeType: 'TEXT_BOX',
                elementProperties: {
                    pageObjectId: coverSlideId,
                    size: {
                        width: { magnitude: 9000000, unit: 'EMU' },
                        height: { magnitude: 1000000, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: (slideWidthEMU - 9000000) / 2,
                        translateY: headerHeightEMU + 5500000, // Position below title
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            insertText: {
                objectId: 'cover_subtitle',
                text: 'Making data meaningful with a true partner in education',
                insertionIndex: 0
            }
        },
        {
            updateTextStyle: {
                objectId: 'cover_subtitle',
                style: {
                    fontSize: { magnitude: 24, unit: 'PT' },
                    italic: true,
                    foregroundColor: {
                        opaqueColor: {
                            rgbColor: {
                                red: 1.0,
                                green: 1.0,
                                blue: 1.0
                            }
                        }
                    }
                },
                fields: 'fontSize,italic,foregroundColor',
                textRange: { type: 'ALL' }
            }
        },
        {
            updateParagraphStyle: {
                objectId: 'cover_subtitle',
                style: {
                    alignment: 'CENTER'
                },
                fields: 'alignment',
                textRange: { type: 'ALL' }
            }
        },
        {
            updateShapeProperties: {
                objectId: 'cover_subtitle',
                shapeProperties: {
                    outline: {
                        propertyState: 'NOT_RENDERED'
                    }
                },
                fields: 'outline'
            }
        },
        // Date: "JUNE 10, 2025" - positioned above title
        {
            createShape: {
                objectId: 'cover_date',
                shapeType: 'TEXT_BOX',
                elementProperties: {
                    pageObjectId: coverSlideId,
                    size: {
                        width: { magnitude: 3000000, unit: 'EMU' },
                        height: { magnitude: 800000, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: (slideWidthEMU - 3000000) / 2,
                        translateY: headerHeightEMU + 2000000, // Original position
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            insertText: {
                objectId: 'cover_date',
                text: 'JUNE 10, 2025',
                insertionIndex: 0
            }
        },
        {
            updateTextStyle: {
                objectId: 'cover_date',
                style: {
                    fontSize: { magnitude: 18, unit: 'PT' },
                    italic: true,
                    foregroundColor: {
                        opaqueColor: {
                            rgbColor: {
                                red: 1.0,
                                green: 1.0,
                                blue: 1.0
                            }
                        }
                    }
                },
                fields: 'fontSize,italic,foregroundColor',
                textRange: { type: 'ALL' }
            }
        },
        {
            updateParagraphStyle: {
                objectId: 'cover_date',
                style: {
                    alignment: 'CENTER'
                },
                fields: 'alignment',
                textRange: { type: 'ALL' }
            }
        },
        {
            updateShapeProperties: {
                objectId: 'cover_date',
                shapeProperties: {
                    outline: {
                        propertyState: 'NOT_RENDERED'
                    }
                },
                fields: 'outline'
            }
        }
    ]
}
