import { extractFolderIdFromUrl, getSlidesClient, uploadImageToDrive } from '@/lib/googleSlides'
import { createChartSlideRequest, createDualChartSlideRequests, createSingleChartSlideRequests } from '@/lib/slides/chartSlide'
import { createCoverSlideRequests } from '@/lib/slides/coverSlide'
import type { SlideData, TextSegment } from '@/types/slide'
import { NextRequest, NextResponse } from 'next/server'
import path from 'path'

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

export async function POST(req: NextRequest) {
    try {
        const body = await req.json()
        const { title, assessments, slides: slidesData, charts: chartPaths, driveFolderUrl } = body

        if (!title) {
            return NextResponse.json({ error: 'title is required' }, { status: 400 })
        }

        // Extract folder ID from URL if provided
        const folderId = driveFolderUrl ? extractFolderIdFromUrl(driveFolderUrl) : null
        if (driveFolderUrl && !folderId) {
            console.warn(`Could not extract folder ID from URL: ${driveFolderUrl}`)
        }

        // If slides array is provided, use it; otherwise fall back to assessments
        const slideData: SlideData[] = slidesData || []

        // Normalize chart paths to absolute paths
        const normalizedCharts: string[] = []
        if (chartPaths && Array.isArray(chartPaths)) {
            for (const chartPath of chartPaths) {
                if (typeof chartPath === 'string') {
                    // If relative path, resolve relative to project root
                    const resolvedPath = path.isAbsolute(chartPath) ? chartPath : path.resolve(process.cwd(), chartPath)
                    normalizedCharts.push(resolvedPath)
                }
            }
        }

        const slidesClient = await getSlidesClient()

        // Create a new presentation
        const response = await slidesClient.presentations.create({
            requestBody: {
                title: title
            }
        })

        const presentationId = response.data.presentationId

        if (!presentationId) {
            throw new Error('Failed to create presentation: No presentation ID returned')
        }

        // Get the first slide (title slide) - we'll keep it as is
        const presentation = await slidesClient.presentations.get({
            presentationId: presentationId
        })

        const slides_data = presentation.data.slides
        if (!slides_data || slides_data.length === 0) {
            throw new Error('No slides found in presentation')
        }

        // Helper function to get layout based on slide type
        const getLayoutForSlideType = (slideType: string): string => {
            switch (slideType.toLowerCase()) {
                case 'intro':
                    return 'TITLE_ONLY'
                case 'graph':
                case 'chart':
                    return 'BLANK'
                case 'content':
                case 'assessment':
                default:
                    return 'TITLE_AND_BODY'
            }
        }

        // Helper function to parse location string to coordinates (in EMU)
        const parseLocationEMU = (location: string) => {
            const slideWidthEMU = SLIDE_WIDTH_EMU
            const slideHeightEMU = (slideWidthEMU * 9) / 16 // 16:9 aspect ratio
            const marginEMU = 500000 // ~0.5 inch margin

            const locations: Record<string, { x: number; y: number }> = {
                'top-left': { x: marginEMU, y: marginEMU },
                'top-center': { x: slideWidthEMU / 2, y: marginEMU },
                'top-right': { x: slideWidthEMU - marginEMU, y: marginEMU },
                center: { x: slideWidthEMU / 2, y: slideHeightEMU / 2 },
                'center-left': { x: marginEMU, y: slideHeightEMU / 2 },
                'center-right': { x: slideWidthEMU - marginEMU, y: slideHeightEMU / 2 },
                'bottom-left': { x: marginEMU, y: slideHeightEMU - marginEMU },
                'bottom-center': { x: slideWidthEMU / 2, y: slideHeightEMU - marginEMU },
                'bottom-right': { x: slideWidthEMU - marginEMU, y: slideHeightEMU - marginEMU },
                title: { x: slideWidthEMU / 2, y: marginEMU },
                body: { x: slideWidthEMU / 2, y: slideHeightEMU / 2 }
            }
            return locations[location.toLowerCase()] || locations['center']
        }

        // Helper function to parse location string to coordinates (in PT for backward compatibility)
        const parseLocation = (location: string, slideWidth: number = 720, slideHeight: number = 405) => {
            const locations: Record<string, { x: number; y: number }> = {
                'top-left': { x: 50, y: 50 },
                'top-center': { x: slideWidth / 2, y: 50 },
                'top-right': { x: slideWidth - 50, y: 50 },
                center: { x: slideWidth / 2, y: slideHeight / 2 },
                'center-left': { x: 50, y: slideHeight / 2 },
                'center-right': { x: slideWidth - 50, y: slideHeight / 2 },
                'bottom-left': { x: 50, y: slideHeight - 50 },
                'bottom-center': { x: slideWidth / 2, y: slideHeight - 50 },
                'bottom-right': { x: slideWidth - 50, y: slideHeight - 50 },
                title: { x: slideWidth / 2, y: 50 },
                body: { x: slideWidth / 2, y: slideHeight / 2 }
            }
            return locations[location.toLowerCase()] || locations['center']
        }

        // Create slides based on slideData array (assessments are no longer used)
        const slidesToCreate = slideData.length > 0 ? slideData : []

        // Create cover slide as the very first slide
        const coverSlideId = 'cover_slide_001'

        // Create slide requests using the extracted functions
        const createSlideRequests: Array<{ createSlide?: unknown; insertText?: unknown; updateTextStyle?: unknown; updateParagraphStyle?: unknown }> = [
            ...createCoverSlideRequests(coverSlideId)
        ]

        // Add chart slides if charts are provided
        // Calculate slides needed based on subject graph detection
        let chartSlideCount = 0
        if (normalizedCharts.length > 0) {
            // Helper function to check if charts are subject graphs (math and reading)
            const isSubjectGraphPair = (chartPaths: string[]): boolean => {
                if (chartPaths.length < 2) return false
                const chartNames = chartPaths.map((p) => path.basename(p, path.extname(p)).toLowerCase())
                const hasMath = chartNames.some((name) => name.includes('math'))
                const hasReading = chartNames.some((name) => name.includes('reading') || name.includes('read'))
                return hasMath && hasReading
            }

            // Calculate how many slides we need
            let totalSlidesNeeded = 0
            for (let i = 0; i < normalizedCharts.length; ) {
                const currentCharts = normalizedCharts.slice(i, i + 2)
                if (isSubjectGraphPair(currentCharts) && i + 1 < normalizedCharts.length) {
                    totalSlidesNeeded++ // One slide for the pair
                    i += 2
                } else {
                    totalSlidesNeeded++ // One slide per chart
                    i += 1
                }
            }

            console.log(`[Slides] Creating ${totalSlidesNeeded} chart slide(s) for ${normalizedCharts.length} chart(s)`)
            for (let i = 0; i < totalSlidesNeeded; i++) {
                const chartSlideId = `chart_slide_${i}`
                const insertionIndex = 1 + i // After cover slide
                createSlideRequests.push(createChartSlideRequest(presentationId, chartSlideId, insertionIndex))
                chartSlideCount++
            }
            console.log(`[Slides] Created ${chartSlideCount} chart slide(s)`)
        }

        // Create additional slides if provided
        if (slidesToCreate.length > 0) {
            // Create all additional slides
            slidesToCreate.forEach((slide: SlideData, index: number) => {
                const slideObjectId = `Slide_${index}`
                createSlideRequests.push({
                    createSlide: {
                        objectId: slideObjectId,
                        slideLayoutReference: {
                            predefinedLayout: getLayoutForSlideType(slide.slideType)
                        },
                        insertionIndex: index + 1 + chartSlideCount // Insert after title slide and chart slides
                    }
                })
            })
        }

        // Execute slide creation
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (slidesClient.presentations.batchUpdate as any)({
            presentationId: presentationId,
            requestBody: {
                requests: createSlideRequests
            }
        })

        // Get updated presentation to work with created slides
        const updatedPresentation = await slidesClient.presentations.get({
            presentationId: presentationId
        })

        const allSlides = updatedPresentation.data.slides || []

        // Define slide dimensions for chart placement
        const slideWidthEMU = SLIDE_WIDTH_EMU
        const slideHeightEMU = (slideWidthEMU * 9) / 16 // 16:9 aspect ratio

        // Add charts to chart slides
        if (normalizedCharts.length > 0) {
            // Upload charts to Google Drive and get publicly accessible URLs
            // Note: Google Slides API requires publicly accessible URLs to fetch images
            // Once embedded, images become part of the Slides presentation file
            console.log(`Uploading ${normalizedCharts.length} charts to Google Drive...`)
            const chartUrls: string[] = []
            for (const chartPath of normalizedCharts) {
                try {
                    const fileName = path.basename(chartPath)
                    // makePublic: true - required for Google Slides API to access the images
                    // Once embedded in Slides, images become part of the presentation file
                    const driveUrl = await uploadImageToDrive(chartPath, fileName, folderId, true)
                    chartUrls.push(driveUrl)
                    const folderInfo = folderId ? ` (in folder ${folderId})` : ''
                    console.log(`✓ Uploaded ${fileName} to Drive${folderInfo}`)
                } catch (error: unknown) {
                    const errorMessage = error instanceof Error ? error.message : 'Unknown error'
                    console.error(`Failed to upload chart ${chartPath}:`, errorMessage)
                    // Continue with other charts even if one fails
                }
            }

            // Helper function to check if charts are subject graphs (math and reading)
            const isSubjectGraphPair = (chartPaths: string[]): boolean => {
                if (chartPaths.length < 2) return false

                const chartNames = chartPaths.map((p) => path.basename(p, path.extname(p)).toLowerCase())

                // Check if we have both math and reading charts
                const hasMath = chartNames.some((name) => name.includes('math'))
                const hasReading = chartNames.some((name) => name.includes('reading') || name.includes('read'))

                return hasMath && hasReading
            }

            let globalChartIndex = 0 // Track chart index across all slides
            const chartsAddedToSlides: string[] = []
            let slideIndex = 1 // Start after cover slide

            // Process charts, grouping subject pairs (math+reading) together
            for (let i = 0; i < chartUrls.length; ) {
                // Check if current chart and next chart form a subject pair
                const currentCharts = normalizedCharts.slice(i, i + 2)
                const isSubjectPair = isSubjectGraphPair(currentCharts)

                let slideChartUrls: string[] = []
                let slideChartPaths: string[] = []

                if (isSubjectPair && i + 1 < chartUrls.length) {
                    // Use dual template for subject pairs (math + reading)
                    slideChartUrls = chartUrls.slice(i, i + 2)
                    slideChartPaths = normalizedCharts.slice(i, i + 2)
                    i += 2 // Process both charts together
                } else {
                    // Use single template for individual charts
                    slideChartUrls = [chartUrls[i]]
                    slideChartPaths = [normalizedCharts[i]]
                    i += 1 // Process one chart at a time
                }

                if (slideIndex < allSlides.length) {
                    const chartSlide = allSlides[slideIndex]
                    if (!chartSlide.objectId) {
                        console.warn(`Chart slide at index ${slideIndex} has no objectId, skipping`)
                        slideIndex++
                        continue
                    }

                    // Log which charts are being added to this slide
                    const chartNames = slideChartUrls.map((url) => {
                        // Extract filename from URL if possible, otherwise use index
                        try {
                            const urlObj = new URL(url)
                            return urlObj.pathname.split('/').pop() || `chart_${i}`
                        } catch {
                            return `chart_${i}`
                        }
                    })
                    console.log(`[Slides] Adding ${slideChartUrls.length} chart(s) to slide ${slideIndex}:`, chartNames)
                    chartsAddedToSlides.push(...chartNames)

                    let chartRequests: unknown[] = []

                    // Extract grade from chart filename for dual chart slides
                    const extractGradeFromFilename = (filename: string): string => {
                        const name = path.basename(filename, path.extname(filename)).toLowerCase()

                        // Try to find grade patterns: "grade_5", "grade5", "grade 5", "5th", "k", "kindergarten", etc.
                        // Priority: Look for "grade" followed by number, then standalone grade numbers
                        // Avoid matching numbers from "section3" or other non-grade patterns
                        let gradeMatch = name.match(/grade[_\s]*(\d+|k|kindergarten)/i)
                        if (!gradeMatch) {
                            // Try standalone grade patterns (but avoid matching section numbers)
                            gradeMatch = name.match(/(?:^|[^a-z])(\d+)(?:th|st|nd|rd)(?:[^a-z]|$)/i)
                        }
                        if (!gradeMatch) {
                            // Last resort: look for k or kindergarten
                            gradeMatch = name.match(/\b(k|kindergarten)\b/i)
                        }

                        if (gradeMatch) {
                            const grade = gradeMatch[1].toLowerCase()
                            if (grade === 'k' || grade === 'kindergarten') {
                                return 'Kindergarten'
                            }
                            const num = parseInt(grade, 10)
                            if (!isNaN(num)) {
                                // Handle Kindergarten (grade 0)
                                if (num === 0) {
                                    return 'Kindergarten'
                                }
                                // Convert number to ordinal (1st, 2nd, 3rd, 4th, etc.)
                                const getOrdinalSuffix = (n: number): string => {
                                    const j = n % 10
                                    const k = n % 100
                                    if (j === 1 && k !== 11) return 'st'
                                    if (j === 2 && k !== 12) return 'nd'
                                    if (j === 3 && k !== 13) return 'rd'
                                    return 'th'
                                }
                                return `${num}${getOrdinalSuffix(num)} Grade`
                            }
                        }
                        return 'Grade' // Default fallback
                    }

                    // Extract graph name from filename for single chart slides
                    const extractGraphNameFromFilename = (filename: string): string => {
                        const name = path.basename(filename, path.extname(filename))
                        // Remove common prefixes/suffixes and clean up
                        let graphName = name
                            .replace(/^.*_section\d+_/, '') // Remove section prefixes
                            .replace(/_fall_trends|_winter_trends|_spring_trends|_trends/gi, '') // Remove trend suffixes
                            .replace(/_/g, ' ') // Replace underscores with spaces
                            .replace(/\b(grade|gr)\s*\d+/gi, '') // Remove grade references
                            .replace(/\s+/g, ' ') // Normalize spaces
                            .trim()

                        // Capitalize first letter of each word
                        graphName = graphName
                            .split(' ')
                            .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                            .join(' ')

                        return graphName || 'Chart' // Default fallback
                    }

                    // Determine template based on subject graph detection
                    if (isSubjectPair && slideChartUrls.length === 2) {
                        // Extract grade from first chart filename for dual chart slides
                        const firstChartPath = slideChartPaths[0] || ''
                        const grade = extractGradeFromFilename(firstChartPath)
                        const title = grade // Just the grade, no "Trends"

                        // Use dual chart template for subject graphs (math & reading)
                        chartRequests = createDualChartSlideRequests(
                            chartSlide.objectId,
                            slideChartUrls[0],
                            slideChartUrls[1],
                            title,
                            slideWidthEMU,
                            slideHeightEMU,
                            globalChartIndex
                        )
                        console.log(`[Slides] Using dual chart template for subject graphs (math/reading) with title: "${title}"`)
                    } else {
                        // Extract graph name from filename for single chart slides
                        const firstChartPath = slideChartPaths[0] || ''
                        const title = extractGraphNameFromFilename(firstChartPath)

                        // Use single chart template for all other cases
                        chartRequests = createSingleChartSlideRequests(
                            chartSlide.objectId,
                            slideChartUrls[0],
                            title,
                            slideWidthEMU,
                            slideHeightEMU,
                            globalChartIndex
                        )
                        console.log(`[Slides] Using single chart template with title: "${title}"`)
                    }

                    globalChartIndex += slideChartUrls.length // Update global index

                    if (chartRequests.length > 0) {
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        await (slidesClient.presentations.batchUpdate as any)({
                            presentationId: presentationId,
                            requestBody: {
                                requests: chartRequests
                            }
                        })
                        console.log(`[Slides] ✓ Successfully added ${chartRequests.length} chart element(s) to slide ${slideIndex}`)
                    }

                    slideIndex++ // Move to next slide
                } else {
                    console.warn(
                        `[Slides] Slide index ${slideIndex} is out of bounds (total slides: ${allSlides.length}). Skipping ${slideChartUrls.length} chart(s).`
                    )
                    break // Stop processing if we run out of slides
                }
            }

            console.log(`[Slides] Summary: Added ${chartsAddedToSlides.length} chart(s) to ${slideIndex - 2} slide(s)`)
            console.log(`[Slides] Charts added to slides:`, chartsAddedToSlides)
        }

        // Process each additional slide and add content (skip title slide)
        if (slidesToCreate.length > 0) {
            for (let i = 0; i < slidesToCreate.length; i++) {
                const slideIndex = i + 1 + chartSlideCount // Skip title slide (0) and chart slides
                if (slideIndex < allSlides.length) {
                    const slide = allSlides[slideIndex]
                    const slideInfo = slidesToCreate[i]
                    const slideObjectId = slide.objectId
                    const updateRequests: Array<Record<string, unknown>> = []

                    // Add text elements with formatting support
                    if (slideInfo.text && slideInfo.text.length > 0) {
                        for (const textElement of slideInfo.text) {
                            const locationEMU = parseLocationEMU(textElement.location)
                            const textBoxObjectId = `TextBox_${i}_${updateRequests.length}`

                            // Determine if we should use formatted segments or simple text
                            const hasSegments = textElement.segments && textElement.segments.length > 0
                            const textContent = hasSegments ? textElement.segments!.map((s: TextSegment) => s.text).join('') : textElement.context

                            // Create text box using EMU for precise positioning
                            const textBoxWidth = textElement.segments ? 4000000 : 5000000 // ~4-5 inches
                            const textBoxHeight = textElement.segments ? 3000000 : 2000000 // ~2-3 inches

                            updateRequests.push({
                                createShape: {
                                    objectId: textBoxObjectId,
                                    shapeType: 'TEXT_BOX',
                                    elementProperties: {
                                        pageObjectId: slideObjectId,
                                        size: {
                                            width: { magnitude: textBoxWidth, unit: 'EMU' },
                                            height: { magnitude: textBoxHeight, unit: 'EMU' }
                                        },
                                        transform: {
                                            scaleX: 1.0,
                                            scaleY: 1.0,
                                            translateX: locationEMU.x - textBoxWidth / 2,
                                            translateY: locationEMU.y - textBoxHeight / 2,
                                            unit: 'EMU'
                                        }
                                    }
                                }
                            })

                            if (hasSegments && textElement.segments) {
                                // Insert formatted text segments
                                let cursorIndex = 0
                                for (const segment of textElement.segments) {
                                    const segmentLength = segment.text.length

                                    // Insert text
                                    updateRequests.push({
                                        insertText: {
                                            objectId: textBoxObjectId,
                                            text: segment.text,
                                            insertionIndex: cursorIndex
                                        }
                                    })

                                    // Apply formatting
                                    const styleFields: string[] = []
                                    const style: Record<string, unknown> = {}

                                    if (segment.fontSize !== undefined) {
                                        styleFields.push('fontSize')
                                        style.fontSize = { magnitude: segment.fontSize, unit: 'PT' }
                                    } else if (textElement.fontSize !== undefined) {
                                        styleFields.push('fontSize')
                                        style.fontSize = { magnitude: textElement.fontSize, unit: 'PT' }
                                    }

                                    if (segment.bold !== undefined) {
                                        styleFields.push('bold')
                                        style.bold = segment.bold
                                    } else if (textElement.bold !== undefined) {
                                        styleFields.push('bold')
                                        style.bold = textElement.bold
                                    }

                                    if (segment.italic !== undefined) {
                                        styleFields.push('italic')
                                        style.italic = segment.italic
                                    }

                                    if (segment.color) {
                                        styleFields.push('foregroundColor')
                                        style.foregroundColor = hexToRgbColor(segment.color)
                                    }

                                    if (styleFields.length > 0) {
                                        updateRequests.push({
                                            updateTextStyle: {
                                                objectId: textBoxObjectId,
                                                fields: styleFields.join(','),
                                                textRange: {
                                                    type: 'FIXED_RANGE',
                                                    startIndex: cursorIndex,
                                                    endIndex: cursorIndex + segmentLength
                                                },
                                                style: style
                                            }
                                        })
                                    }

                                    cursorIndex += segmentLength
                                }

                                // Apply alignment if specified
                                if (textElement.alignment) {
                                    updateRequests.push({
                                        updateParagraphStyle: {
                                            objectId: textBoxObjectId,
                                            fields: 'alignment',
                                            textRange: { type: 'ALL' },
                                            style: {
                                                alignment: textElement.alignment
                                            }
                                        }
                                    })
                                }
                            } else {
                                // Simple text insertion
                                updateRequests.push({
                                    insertText: {
                                        objectId: textBoxObjectId,
                                        insertionIndex: 0,
                                        text: textContent
                                    }
                                })

                                // Apply default formatting if specified
                                const styleFields: string[] = []
                                const style: Record<string, unknown> = {}

                                if (textElement.fontSize !== undefined) {
                                    styleFields.push('fontSize')
                                    style.fontSize = { magnitude: textElement.fontSize, unit: 'PT' }
                                }

                                if (textElement.bold !== undefined) {
                                    styleFields.push('bold')
                                    style.bold = textElement.bold
                                }

                                if (styleFields.length > 0) {
                                    updateRequests.push({
                                        updateTextStyle: {
                                            objectId: textBoxObjectId,
                                            fields: styleFields.join(','),
                                            textRange: { type: 'ALL' },
                                            style: style
                                        }
                                    })
                                }

                                if (textElement.alignment) {
                                    updateRequests.push({
                                        updateParagraphStyle: {
                                            objectId: textBoxObjectId,
                                            fields: 'alignment',
                                            textRange: { type: 'ALL' },
                                            style: {
                                                alignment: textElement.alignment
                                            }
                                        }
                                    })
                                }
                            }
                        }
                    }

                    // Add table if provided
                    if (slideInfo.table) {
                        const table = slideInfo.table
                        const locationEMU = parseLocationEMU(table.location)
                        const tableObjectId = `Table_${i}`

                        const tableRows = table.data.length
                        const tableCols = table.data[0]?.length || 0

                        if (tableRows > 0 && tableCols > 0) {
                            const tableWidth = table.width || 4500000 // ~5 inches default
                            const tableHeight = table.height || 2000000 // ~2 inches default

                            // Create table
                            updateRequests.push({
                                createTable: {
                                    objectId: tableObjectId,
                                    elementProperties: {
                                        pageObjectId: slideObjectId,
                                        transform: {
                                            scaleX: 1.0,
                                            scaleY: 1.0,
                                            translateX: locationEMU.x - tableWidth / 2,
                                            translateY: locationEMU.y - tableHeight / 2,
                                            unit: 'EMU'
                                        },
                                        size: {
                                            width: { magnitude: tableWidth, unit: 'EMU' },
                                            height: { magnitude: tableHeight, unit: 'EMU' }
                                        }
                                    },
                                    rows: tableRows,
                                    columns: tableCols
                                }
                            })

                            // Populate table cells
                            for (let rIndex = 0; rIndex < tableRows; rIndex++) {
                                for (let cIndex = 0; cIndex < tableCols; cIndex++) {
                                    const cellData = table.data[rIndex][cIndex]
                                    const cellText = typeof cellData === 'string' ? cellData : cellData.text || ''
                                    const isHeaderRow = rIndex === 0 && table.headerRowBold !== false

                                    // Insert text
                                    updateRequests.push({
                                        insertText: {
                                            objectId: tableObjectId,
                                            cellLocation: {
                                                rowIndex: rIndex,
                                                columnIndex: cIndex
                                            },
                                            text: cellText
                                        }
                                    })

                                    // Apply formatting
                                    if (typeof cellData === 'object' && cellData.bold !== undefined) {
                                        updateRequests.push({
                                            updateTextStyle: {
                                                objectId: tableObjectId,
                                                cellLocation: {
                                                    rowIndex: rIndex,
                                                    columnIndex: cIndex
                                                },
                                                fields: 'bold',
                                                style: { bold: cellData.bold }
                                            }
                                        })
                                    } else if (isHeaderRow) {
                                        // Bold header row by default
                                        updateRequests.push({
                                            updateTextStyle: {
                                                objectId: tableObjectId,
                                                cellLocation: {
                                                    rowIndex: rIndex,
                                                    columnIndex: cIndex
                                                },
                                                fields: 'bold',
                                                style: { bold: true }
                                            }
                                        })
                                    }

                                    // Apply color if specified
                                    if (typeof cellData === 'object' && cellData.color) {
                                        updateRequests.push({
                                            updateTextStyle: {
                                                objectId: tableObjectId,
                                                cellLocation: {
                                                    rowIndex: rIndex,
                                                    columnIndex: cIndex
                                                },
                                                fields: 'foregroundColor',
                                                style: {
                                                    foregroundColor: hexToRgbColor(cellData.color)
                                                }
                                            }
                                        })
                                    }
                                }
                            }
                        }
                    }

                    // Add images
                    if (slideInfo.images && slideInfo.images.length > 0) {
                        for (const imageElement of slideInfo.images) {
                            const location = parseLocation(imageElement.location)
                            const imageObjectId = `Image_${i}_${updateRequests.length}`

                            updateRequests.push({
                                createImage: {
                                    objectId: imageObjectId,
                                    url: imageElement.images,
                                    elementProperties: {
                                        pageObjectId: slideObjectId,
                                        size: {
                                            width: { magnitude: 300, unit: 'PT' },
                                            height: { magnitude: 200, unit: 'PT' }
                                        },
                                        transform: {
                                            scaleX: 1,
                                            scaleY: 1,
                                            translateX: location.x - 150,
                                            translateY: location.y - 100,
                                            unit: 'PT'
                                        }
                                    }
                                }
                            })
                        }
                    }

                    // Assessment information display removed per user request

                    // Execute all update requests for this slide
                    if (updateRequests.length > 0) {
                        try {
                            await slidesClient.presentations.batchUpdate({
                                presentationId: presentationId,
                                requestBody: {
                                    requests: updateRequests
                                }
                            })
                        } catch (updateError: unknown) {
                            const errorMessage = updateError instanceof Error ? updateError.message : 'Unknown error'
                            console.warn(`Failed to update slide ${i + 1}:`, errorMessage)
                        }
                    }
                }
            }
        }

        const presentationUrl = `https://docs.google.com/presentation/d/${presentationId}/edit`

        return NextResponse.json({
            success: true,
            presentationId: presentationId,
            presentationUrl: presentationUrl,
            title: response.data.title || title
        })
    } catch (err: unknown) {
        console.error('Slides API error:', err)
        const errorObj = err as { message?: string; error?: { message?: string } }
        console.error('Error details:', JSON.stringify(errorObj, null, 2))

        // Extract more detailed error information
        const errorMessage = errorObj?.message || errorObj?.error?.message || 'Unknown error'
        const errorDetails = (err as { response?: { data?: unknown }; error?: unknown }).response?.data || (err as { error?: unknown }).error || err

        return NextResponse.json(
            {
                error: 'Failed to create presentation',
                details: errorMessage,
                fullError: process.env.NODE_ENV === 'development' ? errorDetails : undefined
            },
            { status: 500 }
        )
    }
}
