import fs from 'fs'
import path from 'path'

const EMU_PER_INCH = 914400
const SLIDE_WIDTH_EMU = 10 * EMU_PER_INCH
const SLIDE_HEIGHT_EMU = (SLIDE_WIDTH_EMU * 9) / 16 // 16:9 aspect ratio

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
): any[] {
    const requests: any[] = []

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
export function createChartSlideRequest(presentationId: string, slideObjectId: string, insertionIndex: number): any {
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
