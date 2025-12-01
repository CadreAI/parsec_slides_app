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
 * Creates the NWEA summary slide requests for the presentation.
 * Returns an array of batch update requests for the NWEA slide.
 */
export function createNweaSlideRequests(nweaSlideId: string): any[] {
    const slideWidthEMU = SLIDE_WIDTH_EMU

    return [
        // Always create the NWEA summary slide after cover slide
        {
            createSlide: {
                objectId: nweaSlideId,
                slideLayoutReference: {
                    predefinedLayout: 'BLANK'
                },
                insertionIndex: 1 // Insert after cover slide
            }
        },
        // Top bar 1 (left side) for NWEA slide
        {
            createShape: {
                objectId: 'top_bar_1',
                shapeType: 'RECTANGLE',
                elementProperties: {
                    pageObjectId: nweaSlideId,
                    size: {
                        width: { magnitude: slideWidthEMU / 2, unit: 'EMU' },
                        height: { magnitude: 200000, unit: 'EMU' } // ~0.22 inches
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
                objectId: 'top_bar_1',
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
                    }
                },
                fields: 'shapeBackgroundFill.solidFill.color'
            }
        },
        // Top bar 2 (right side)
        {
            createShape: {
                objectId: 'top_bar_2',
                shapeType: 'RECTANGLE',
                elementProperties: {
                    pageObjectId: nweaSlideId,
                    size: {
                        width: { magnitude: slideWidthEMU / 2, unit: 'EMU' },
                        height: { magnitude: 200000, unit: 'EMU' } // ~0.22 inches
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: slideWidthEMU / 2,
                        translateY: 0,
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            updateShapeProperties: {
                objectId: 'top_bar_2',
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
                    }
                },
                fields: 'shapeBackgroundFill.solidFill.color'
            }
        },
        // Title box at top center
        {
            createShape: {
                objectId: 'title_box',
                shapeType: 'TEXT_BOX',
                elementProperties: {
                    pageObjectId: nweaSlideId,
                    size: {
                        width: { magnitude: 9000000, unit: 'EMU' },
                        height: { magnitude: 1200000, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: (slideWidthEMU - 9000000) / 2, // Center horizontally
                        translateY: 300000, // Near top
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            insertText: {
                objectId: 'title_box',
                text: 'NWEA Fall to Spring Growth: Fairmead Elementary',
                insertionIndex: 0
            }
        },
        {
            updateTextStyle: {
                objectId: 'title_box',
                style: {
                    fontSize: { magnitude: 28, unit: 'PT' },
                    bold: true
                },
                fields: 'fontSize,bold',
                textRange: { type: 'ALL' }
            }
        },
        {
            updateParagraphStyle: {
                objectId: 'title_box',
                style: {
                    alignment: 'CENTER'
                },
                fields: 'alignment',
                textRange: { type: 'ALL' }
            }
        },
        // Summary text box on the right side
        {
            createShape: {
                objectId: 'summary_text',
                shapeType: 'TEXT_BOX',
                elementProperties: {
                    pageObjectId: nweaSlideId,
                    size: {
                        width: { magnitude: 3000000, unit: 'EMU' }, // ~3.3 inches wide
                        height: { magnitude: 4000000, unit: 'EMU' } // ~4.4 inches tall
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: slideWidthEMU - 3200000, // Right side with margin
                        translateY: 2000000, // Below title
                        unit: 'EMU'
                    }
                }
            }
        },
        // Insert summary text
        {
            insertText: {
                objectId: 'summary_text',
                text: '• In Reading, both 5th and 6th grade fell short of typical Fall to Spring growth targets.\n\n• In Math, 5th grade matched typical Fall to Spring growth targets while 6th graders exceeded typical growth.',
                insertionIndex: 0
            }
        },
        // Set default font size
        {
            updateTextStyle: {
                objectId: 'summary_text',
                style: {
                    fontSize: { magnitude: 14, unit: 'PT' }
                },
                fields: 'fontSize',
                textRange: { type: 'ALL' }
            }
        },
        // Highlight "fell short" in Reading bullet (blue, bold)
        // Character positions: 37-47
        {
            updateTextStyle: {
                objectId: 'summary_text',
                style: {
                    bold: true,
                    foregroundColor: hexToRgbColor('0094bd') // Blue - returns {opaqueColor: {rgbColor: {...}}}
                },
                fields: 'bold,foregroundColor',
                textRange: {
                    type: 'FIXED_RANGE',
                    startIndex: 37,
                    endIndex: 47
                }
            }
        },
        // Highlight "matched" in Math bullet (blue, bold)
        // Character positions: 112-119
        {
            updateTextStyle: {
                objectId: 'summary_text',
                style: {
                    bold: true,
                    foregroundColor: hexToRgbColor('0094bd') // Blue
                },
                fields: 'bold,foregroundColor',
                textRange: {
                    type: 'FIXED_RANGE',
                    startIndex: 112,
                    endIndex: 119
                }
            }
        },
        // Highlight "exceeded typical growth" in Math bullet (blue, bold)
        // Character positions: 176-199
        {
            updateTextStyle: {
                objectId: 'summary_text',
                style: {
                    bold: true,
                    foregroundColor: hexToRgbColor('0094bd') // Blue
                },
                fields: 'bold,foregroundColor',
                textRange: {
                    type: 'FIXED_RANGE',
                    startIndex: 176,
                    endIndex: 199
                }
            }
        },
        {
            updateParagraphStyle: {
                objectId: 'summary_text',
                style: {
                    lineSpacing: 150,
                    spaceAbove: { magnitude: 6, unit: 'PT' }
                },
                fields: 'lineSpacing,spaceAbove',
                textRange: { type: 'ALL' }
            }
        },
        // Reading section label (left side)
        {
            createShape: {
                objectId: 'reading_label',
                shapeType: 'TEXT_BOX',
                elementProperties: {
                    pageObjectId: nweaSlideId,
                    size: {
                        width: { magnitude: 2500000, unit: 'EMU' },
                        height: { magnitude: 500000, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: 500000, // Left side
                        translateY: 2000000, // Below title
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            insertText: {
                objectId: 'reading_label',
                text: 'Reading Student Norms',
                insertionIndex: 0
            }
        },
        {
            updateTextStyle: {
                objectId: 'reading_label',
                style: {
                    fontSize: { magnitude: 18, unit: 'PT' },
                    bold: true
                },
                fields: 'fontSize,bold',
                textRange: { type: 'ALL' }
            }
        },
        // Math section label (middle)
        {
            createShape: {
                objectId: 'math_label',
                shapeType: 'TEXT_BOX',
                elementProperties: {
                    pageObjectId: nweaSlideId,
                    size: {
                        width: { magnitude: 2500000, unit: 'EMU' },
                        height: { magnitude: 500000, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: slideWidthEMU / 2 - 1250000, // Center
                        translateY: 2000000, // Below title
                        unit: 'EMU'
                    }
                }
            }
        },
        {
            insertText: {
                objectId: 'math_label',
                text: 'Math Student Norms',
                insertionIndex: 0
            }
        },
        {
            updateTextStyle: {
                objectId: 'math_label',
                style: {
                    fontSize: { magnitude: 18, unit: 'PT' },
                    bold: true
                },
                fields: 'fontSize,bold',
                textRange: { type: 'ALL' }
            }
        }
        // Note: Images are commented out by default as they require publicly accessible URLs
        // Uncomment and provide valid image URLs when ready
        // Reading Chart 1: Fall-to-Spring Growth %tile (left side, top)
        // {
        //     createImage: {
        //         objectId: 'reading_chart1',
        //         url: 'YOUR_READING_CHART1_URL_HERE',
        //         elementProperties: {
        //             pageObjectId: nweaSlideId,
        //             size: {
        //                 width: { magnitude: 2500000, unit: 'EMU' },
        //                 height: { magnitude: 2000000, unit: 'EMU' }
        //             },
        //             transform: {
        //                 scaleX: 1,
        //                 scaleY: 1,
        //                 translateX: 500000,
        //                 translateY: 2600000,
        //                 unit: 'EMU'
        //             }
        //         }
        //     }
        // },
        // Reading Chart 2: Fall-to-Spring Student CGI (left side, bottom)
        // {
        //     createImage: {
        //         objectId: 'reading_chart2',
        //         url: 'YOUR_READING_CHART2_URL_HERE',
        //         elementProperties: {
        //             pageObjectId: nweaSlideId,
        //             size: {
        //                 width: { magnitude: 2500000, unit: 'EMU' },
        //                 height: { magnitude: 2000000, unit: 'EMU' }
        //             },
        //             transform: {
        //                 scaleX: 1,
        //                 scaleY: 1,
        //                 translateX: 500000,
        //                 translateY: 4700000,
        //                 unit: 'EMU'
        //             }
        //         }
        //     }
        // },
        // Math Chart 1: Fall-to-Spring Growth %tile (center, top)
        // {
        //     createImage: {
        //         objectId: 'math_chart1',
        //         url: 'YOUR_MATH_CHART1_URL_HERE',
        //         elementProperties: {
        //             pageObjectId: nweaSlideId,
        //             size: {
        //                 width: { magnitude: 2500000, unit: 'EMU' },
        //                 height: { magnitude: 2000000, unit: 'EMU' }
        //             },
        //             transform: {
        //                 scaleX: 1,
        //                 scaleY: 1,
        //                 translateX: slideWidthEMU / 2 - 1250000,
        //                 translateY: 2600000,
        //                 unit: 'EMU'
        //             }
        //         }
        //     }
        // },
        // Math Chart 2: Fall-to-Spring Student CGI (center, bottom)
        // {
        //     createImage: {
        //         objectId: 'math_chart2',
        //         url: 'YOUR_MATH_CHART2_URL_HERE',
        //         elementProperties: {
        //             pageObjectId: nweaSlideId,
        //             size: {
        //                 width: { magnitude: 2500000, unit: 'EMU' },
        //                 height: { magnitude: 2000000, unit: 'EMU' }
        //             },
        //             transform: {
        //                 scaleX: 1,
        //                 scaleY: 1,
        //                 translateX: slideWidthEMU / 2 - 1250000,
        //                 translateY: 4700000,
        //                 unit: 'EMU'
        //             }
        //         }
        //     }
        // }
    ]
}
