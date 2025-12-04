const EMU_PER_INCH = 914400
const SLIDE_WIDTH_EMU = 10 * EMU_PER_INCH
const SLIDE_HEIGHT_EMU = (SLIDE_WIDTH_EMU * 9) / 16 // 16:9 aspect ratio

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

interface CreateSlideRequest {
    createSlide: {
        objectId: string
        slideLayoutReference: {
            predefinedLayout: string
        }
        insertionIndex: number
    }
}

interface ChartSlideRequest {
    createImage: {
        objectId: string
        url: string
        elementProperties: {
            pageObjectId: string
            size: {
                width: { magnitude: number; unit: string }
                height: { magnitude: number; unit: string }
            }
            transform: {
                scaleX: number
                scaleY: number
                translateX: number
                translateY: number
                unit: string
            }
        }
    }
}

/**
 * Create slide requests for adding charts to a slide
 * chartUrls should be publicly accessible URLs (e.g., from Google Drive)
 * startIndex: Starting index for unique object IDs across all slides
 */
export function createChartSlideRequests(
    slideObjectId: string,
    chartUrls: string[],
    slideWidthEMU: number = SLIDE_WIDTH_EMU,
    slideHeightEMU: number = SLIDE_HEIGHT_EMU,
    startIndex: number = 0
): ChartSlideRequest[] {
    const requests: ChartSlideRequest[] = []

    if (chartUrls.length === 0) {
        return requests
    }

    const marginEMU = 500000 // ~0.5 inch margin
    const chartsPerRow = Math.min(2, chartUrls.length)
    const chartsPerCol = Math.ceil(chartUrls.length / chartsPerRow)

    // Calculate chart dimensions
    const availableWidth = slideWidthEMU - marginEMU * 2
    const availableHeight = slideHeightEMU - marginEMU * 2
    const chartWidth = (availableWidth - marginEMU * (chartsPerRow - 1)) / chartsPerRow
    const chartHeight = (availableHeight - marginEMU * (chartsPerCol - 1)) / chartsPerCol

    chartUrls.forEach((chartUrl, index) => {
        if (!chartUrl) {
            console.warn(`Chart URL is empty at index ${index}`)
            return
        }

        const row = Math.floor(index / chartsPerRow)
        const col = index % chartsPerRow

        const x = marginEMU + col * (chartWidth + marginEMU)
        const y = marginEMU + row * (chartHeight + marginEMU)

        // Use startIndex + index to ensure unique IDs across all slides
        const imageObjectId = `Chart_${startIndex + index}`

        requests.push({
            createImage: {
                objectId: imageObjectId,
                url: chartUrl,
                elementProperties: {
                    pageObjectId: slideObjectId,
                    size: {
                        width: { magnitude: chartWidth, unit: 'EMU' },
                        height: { magnitude: chartHeight, unit: 'EMU' }
                    },
                    transform: {
                        scaleX: 1,
                        scaleY: 1,
                        translateX: x,
                        translateY: y,
                        unit: 'EMU'
                    }
                }
            }
        })
    })

    return requests
}

/**
 * Create a new slide with charts
 */
export function createChartSlideRequest(presentationId: string, slideObjectId: string, insertionIndex: number): CreateSlideRequest {
    return {
        createSlide: {
            objectId: slideObjectId,
            slideLayoutReference: {
                predefinedLayout: 'BLANK'
            },
            insertionIndex: insertionIndex
        }
    }
}

interface DualChartSlideRequest {
    createImage?: {
        objectId: string
        url: string
        elementProperties: {
            pageObjectId: string
            size: {
                width: { magnitude: number; unit: string }
                height: { magnitude: number; unit: string }
            }
            transform: {
                scaleX: number
                scaleY: number
                translateX: number
                translateY: number
                unit: string
            }
        }
    }
    createShape?: {
        objectId: string
        shapeType: string
        elementProperties: {
            pageObjectId: string
            size: {
                width: { magnitude: number; unit: string }
                height: { magnitude: number; unit: string }
            }
            transform: {
                scaleX: number
                scaleY: number
                translateX: number
                translateY: number
                unit: string
            }
        }
    }
    updateShapeProperties?: {
        objectId: string
        shapeProperties: {
            shapeBackgroundFill?: {
                solidFill: {
                    color: {
                        rgbColor: {
                            red: number
                            green: number
                            blue: number
                        }
                    }
                }
            }
        }
        fields: string
    }
    insertText?: {
        objectId: string
        text: string
        insertionIndex: number
    }
    updateTextStyle?: {
        objectId: string
        style: {
            fontSize?: { magnitude: number; unit: string }
            bold?: boolean
            foregroundColor?: {
                opaqueColor: { rgbColor: { red: number; green: number; blue: number } }
            }
        }
        fields: string
        textRange?: {
            type: string
            startIndex?: number
            endIndex?: number
        }
    }
    updateParagraphStyle?: {
        objectId: string
        style: {
            alignment?: string
            lineSpacing?: number
            spaceAbove?: { magnitude: number; unit: string }
        }
        fields: string
        textRange?: {
            type: string
        }
    }
}

/**
 * Create slide requests for a single-chart slide with title/description
 * This template inserts one graph with a title at the top and summary text
 *
 * @param slideObjectId - The object ID of the slide
 * @param chartUrl - URL of the chart
 * @param title - Title/description text
 * @param slideWidthEMU - Slide width in EMU units
 * @param slideHeightEMU - Slide height in EMU units
 * @param startIndex - Starting index for unique object IDs
 */
export function createSingleChartSlideRequests(
    slideObjectId: string,
    chartUrl: string,
    title: string,
    slideWidthEMU: number = SLIDE_WIDTH_EMU,
    slideHeightEMU: number = SLIDE_HEIGHT_EMU,
    startIndex: number = 0
): DualChartSlideRequest[] {
    const requests: DualChartSlideRequest[] = []

    if (!chartUrl) {
        console.warn('Chart URL is required for single chart slide')
        return requests
    }

    const marginEMU = 500000 // ~0.5 inch margin
    const titleHeightEMU = 1200000 // ~1.3 inch for title area (matching nweaSlide)
    const summaryWidthEMU = 3000000 // ~3.3 inches for summary text box
    const topBarHeightEMU = 200000 // ~0.22 inches for top bar

    // Top bar (full width) matching nweaSlide
    const topBarObjectId = `TopBar_${startIndex}`
    requests.push({
        createShape: {
            objectId: topBarObjectId,
            shapeType: 'RECTANGLE',
            elementProperties: {
                pageObjectId: slideObjectId,
                size: {
                    width: { magnitude: slideWidthEMU, unit: 'EMU' },
                    height: { magnitude: topBarHeightEMU, unit: 'EMU' }
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
    })
    requests.push({
        updateShapeProperties: {
            objectId: topBarObjectId,
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
    })

    // Calculate chart dimensions (single chart with summary on the right)
    // Double the chart size (2x width and height)
    const availableWidth = slideWidthEMU - marginEMU * 2
    const availableHeight = slideHeightEMU - marginEMU * 2 - titleHeightEMU - topBarHeightEMU
    const baseChartWidth = availableWidth - summaryWidthEMU - marginEMU
    const baseChartHeight = availableHeight
    // Double both dimensions (2x width and 2x height = 4x area)
    const chartWidth = baseChartWidth * 1.25
    const chartHeight = baseChartHeight * 1.25

    // Create title text box shape (centered, matching nweaSlide pattern)
    const titleObjectId = `Title_${startIndex}`
    const titleWidth = 9000000 // ~9.8 inches (matching nweaSlide)
    const titleX = (slideWidthEMU - titleWidth) / 2 // Center horizontally
    const titleY = topBarHeightEMU + 100000 // Below top bar

    requests.push({
        createShape: {
            objectId: titleObjectId,
            shapeType: 'TEXT_BOX',
            elementProperties: {
                pageObjectId: slideObjectId,
                size: {
                    width: { magnitude: titleWidth, unit: 'EMU' },
                    height: { magnitude: titleHeightEMU, unit: 'EMU' }
                },
                transform: {
                    scaleX: 1,
                    scaleY: 1,
                    translateX: titleX,
                    translateY: titleY,
                    unit: 'EMU'
                }
            }
        }
    })

    // Insert title text
    requests.push({
        insertText: {
            objectId: titleObjectId,
            text: title,
            insertionIndex: 0
        }
    })

    // Title text style (matching nweaSlide: 28pt, bold)
    requests.push({
        updateTextStyle: {
            objectId: titleObjectId,
            style: {
                fontSize: { magnitude: 28, unit: 'PT' },
                bold: true
            },
            fields: 'fontSize,bold',
            textRange: { type: 'ALL' }
        }
    })

    // Center align title text (matching nweaSlide pattern)
    requests.push({
        updateParagraphStyle: {
            objectId: titleObjectId,
            style: {
                alignment: 'CENTER'
            },
            fields: 'alignment',
            textRange: { type: 'ALL' }
        }
    })

    // Single chart (left side, positioned higher and more to the left)
    const chartObjectId = `Chart_${startIndex}`
    const chartX = marginEMU * 0.5 // Move left (reduce margin)
    const chartY = topBarHeightEMU + titleHeightEMU + marginEMU * 0.3 // Move up (reduce spacing below title)

    requests.push({
        createImage: {
            objectId: chartObjectId,
            url: chartUrl,
            elementProperties: {
                pageObjectId: slideObjectId,
                size: {
                    width: { magnitude: chartWidth, unit: 'EMU' },
                    height: { magnitude: chartHeight, unit: 'EMU' }
                },
                transform: {
                    scaleX: 1,
                    scaleY: 1,
                    translateX: chartX,
                    translateY: chartY,
                    unit: 'EMU'
                }
            }
        }
    })

    // Summary text box on the right side (matching nweaSlide)
    const summaryTextObjectId = `Summary_${startIndex}`
    requests.push({
        createShape: {
            objectId: summaryTextObjectId,
            shapeType: 'TEXT_BOX',
            elementProperties: {
                pageObjectId: slideObjectId,
                size: {
                    width: { magnitude: summaryWidthEMU, unit: 'EMU' }, // ~3.3 inches wide
                    height: { magnitude: 4000000, unit: 'EMU' } // ~4.4 inches tall
                },
                transform: {
                    scaleX: 1,
                    scaleY: 1,
                    translateX: slideWidthEMU - 3200000, // Right side with margin
                    translateY: topBarHeightEMU + titleHeightEMU + 100000, // Below title
                    unit: 'EMU'
                }
            }
        }
    })

    // Insert summary text (same as nweaSlide)
    requests.push({
        insertText: {
            objectId: summaryTextObjectId,
            text: '• In Reading, both 5th and 6th grade fell short of typical Fall to Spring growth targets.\n\n• In Math, 5th grade matched typical Fall to Spring growth targets while 6th graders exceeded typical growth.',
            insertionIndex: 0
        }
    })

    // Set default font size
    requests.push({
        updateTextStyle: {
            objectId: summaryTextObjectId,
            style: {
                fontSize: { magnitude: 14, unit: 'PT' }
            },
            fields: 'fontSize',
            textRange: { type: 'ALL' }
        }
    })

    // Highlight "fell short" in Reading bullet (blue, bold)
    requests.push({
        updateTextStyle: {
            objectId: summaryTextObjectId,
            style: {
                bold: true,
                foregroundColor: hexToRgbColor('0094bd') // Blue
            },
            fields: 'bold,foregroundColor',
            textRange: {
                type: 'FIXED_RANGE',
                startIndex: 37,
                endIndex: 47
            }
        }
    })

    // Highlight "matched" in Math bullet (blue, bold)
    requests.push({
        updateTextStyle: {
            objectId: summaryTextObjectId,
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
    })

    // Highlight "exceeded typical growth" in Math bullet (blue, bold)
    requests.push({
        updateTextStyle: {
            objectId: summaryTextObjectId,
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
    })

    // Set paragraph style with line spacing
    requests.push({
        updateParagraphStyle: {
            objectId: summaryTextObjectId,
            style: {
                lineSpacing: 150,
                spaceAbove: { magnitude: 6, unit: 'PT' }
            },
            fields: 'lineSpacing,spaceAbove',
            textRange: { type: 'ALL' }
        }
    })

    return requests
}

/**
 * Create slide requests for a dual-chart slide with title/description
 * This template inserts two graphs side by side with a title at the top
 *
 * @param slideObjectId - The object ID of the slide
 * @param chartUrl1 - URL of the first chart (left side)
 * @param chartUrl2 - URL of the second chart (right side)
 * @param title - Title/description text (typically from one of the chart titles)
 * @param slideWidthEMU - Slide width in EMU units
 * @param slideHeightEMU - Slide height in EMU units
 * @param startIndex - Starting index for unique object IDs
 */
export function createDualChartSlideRequests(
    slideObjectId: string,
    chartUrl1: string,
    chartUrl2: string,
    title: string,
    slideWidthEMU: number = SLIDE_WIDTH_EMU,
    slideHeightEMU: number = SLIDE_HEIGHT_EMU,
    startIndex: number = 0
): DualChartSlideRequest[] {
    const requests: DualChartSlideRequest[] = []

    if (!chartUrl1 || !chartUrl2) {
        console.warn('Both chart URLs are required for dual chart slide')
        return requests
    }

    const marginEMU = 400000 // ~0.44 inch margin (smaller for bigger charts)
    const titleHeightEMU = 1200000 // ~1.3 inch for title area (matching nweaSlide)
    const chartSpacingEMU = 200000 // ~0.22 inch spacing between charts (smaller spacing)
    const topBarHeightEMU = 200000 // ~0.22 inches for top bar

    // Top bar (full width) matching nweaSlide
    const topBarObjectId = `TopBar_${startIndex}`
    requests.push({
        createShape: {
            objectId: topBarObjectId,
            shapeType: 'RECTANGLE',
            elementProperties: {
                pageObjectId: slideObjectId,
                size: {
                    width: { magnitude: slideWidthEMU, unit: 'EMU' },
                    height: { magnitude: topBarHeightEMU, unit: 'EMU' }
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
    })
    requests.push({
        updateShapeProperties: {
            objectId: topBarObjectId,
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
    })

    // Calculate chart dimensions (two charts side by side, no summary text)
    const availableWidth = slideWidthEMU - marginEMU * 2
    const availableHeight = slideHeightEMU - marginEMU * 2 - titleHeightEMU - topBarHeightEMU
    const chartWidth = (availableWidth - chartSpacingEMU) / 2
    const chartHeight = availableHeight

    // Create title text box shape (centered, matching nweaSlide pattern)
    const titleObjectId = `Title_${startIndex}`
    const titleWidth = 9000000 // ~9.8 inches (matching nweaSlide)
    const titleX = (slideWidthEMU - titleWidth) / 2 // Center horizontally
    const titleY = topBarHeightEMU + 100000 // Below top bar

    requests.push({
        createShape: {
            objectId: titleObjectId,
            shapeType: 'TEXT_BOX',
            elementProperties: {
                pageObjectId: slideObjectId,
                size: {
                    width: { magnitude: titleWidth, unit: 'EMU' },
                    height: { magnitude: titleHeightEMU, unit: 'EMU' }
                },
                transform: {
                    scaleX: 1,
                    scaleY: 1,
                    translateX: titleX,
                    translateY: titleY,
                    unit: 'EMU'
                }
            }
        }
    })

    // Insert title text
    requests.push({
        insertText: {
            objectId: titleObjectId,
            text: title,
            insertionIndex: 0
        }
    })

    // Title text style (matching nweaSlide: 28pt, bold)
    requests.push({
        updateTextStyle: {
            objectId: titleObjectId,
            style: {
                fontSize: { magnitude: 28, unit: 'PT' },
                bold: true
            },
            fields: 'fontSize,bold',
            textRange: { type: 'ALL' }
        }
    })

    // Center align title text (matching nweaSlide pattern)
    requests.push({
        updateParagraphStyle: {
            objectId: titleObjectId,
            style: {
                alignment: 'CENTER'
            },
            fields: 'alignment',
            textRange: { type: 'ALL' }
        }
    })

    // First chart (left side)
    const chart1ObjectId = `Chart_${startIndex}`
    const chart1X = marginEMU
    const chart1Y = topBarHeightEMU + titleHeightEMU + marginEMU

    requests.push({
        createImage: {
            objectId: chart1ObjectId,
            url: chartUrl1,
            elementProperties: {
                pageObjectId: slideObjectId,
                size: {
                    width: { magnitude: chartWidth, unit: 'EMU' },
                    height: { magnitude: chartHeight, unit: 'EMU' }
                },
                transform: {
                    scaleX: 1,
                    scaleY: 1,
                    translateX: chart1X,
                    translateY: chart1Y,
                    unit: 'EMU'
                }
            }
        }
    })

    // Second chart (right side)
    const chart2ObjectId = `Chart_${startIndex + 1}`
    const chart2X = marginEMU + chartWidth + chartSpacingEMU
    const chart2Y = topBarHeightEMU + titleHeightEMU + marginEMU

    requests.push({
        createImage: {
            objectId: chart2ObjectId,
            url: chartUrl2,
            elementProperties: {
                pageObjectId: slideObjectId,
                size: {
                    width: { magnitude: chartWidth, unit: 'EMU' },
                    height: { magnitude: chartHeight, unit: 'EMU' }
                },
                transform: {
                    scaleX: 1,
                    scaleY: 1,
                    translateX: chart2X,
                    translateY: chart2Y,
                    unit: 'EMU'
                }
            }
        }
    })

    return requests
}
