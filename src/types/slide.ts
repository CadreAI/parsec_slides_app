// Type definitions for slide creation

export interface SlideImage {
    images: string // URL or base64 image data
    location: string // e.g., "top-left", "center", "bottom-right", or coordinates
    content?: string // Optional description or alt text
}

export interface TextSegment {
    text: string
    bold?: boolean
    color?: string // Hex color code (e.g., "000000" for black, "1976D2" for blue)
    fontSize?: number // Font size in points
    italic?: boolean
}

export interface SlideText {
    location: string // e.g., "title", "body", "top-left", "center", or coordinates
    context: string // The text content (simple text)
    segments?: TextSegment[] // Formatted text segments (overrides context if provided)
    fontSize?: number // Default font size
    alignment?: 'LEFT' | 'CENTER' | 'RIGHT' | 'JUSTIFIED'
    bold?: boolean // Default bold
}

export interface TableCell {
    text: string
    bold?: boolean
    color?: string
}

export interface SlideTable {
    data: (string | TableCell)[][] // 2D array of table data
    location: string
    width?: number // Width in EMU
    height?: number // Height in EMU
    headerRowBold?: boolean // Make header row bold
}

export interface SlideData {
    slideType: string // e.g., "intro", "assessment", "graph", "content"
    information?: string // General information about the slide
    graph?: string // Graph data or description
    images?: SlideImage[] // Array of images
    text?: SlideText[] // Array of text elements
    table?: SlideTable // Table data
}
