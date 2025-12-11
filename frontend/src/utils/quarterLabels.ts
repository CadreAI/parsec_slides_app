/**
 * Utility functions for mapping quarter labels between display (BOY/MOY/EOY) and backend (Fall/Winter/Spring)
 */

const QUARTER_DISPLAY_MAP: Record<string, string> = {
    Fall: 'BOY',
    Winter: 'MOY',
    Spring: 'EOY'
}

const QUARTER_BACKEND_MAP: Record<string, string> = {
    BOY: 'Fall',
    MOY: 'Winter',
    EOY: 'Spring'
}

/**
 * Convert backend quarter values (Fall/Winter/Spring) to display labels (BOY/MOY/EOY)
 */
export function getQuarterDisplayLabel(backendValue: string): string {
    return QUARTER_DISPLAY_MAP[backendValue] || backendValue
}

/**
 * Convert display labels (BOY/MOY/EOY) back to backend values (Fall/Winter/Spring)
 */
export function getQuarterBackendValue(displayLabel: string): string {
    return QUARTER_BACKEND_MAP[displayLabel] || displayLabel
}

/**
 * Map an array of backend quarter values to display labels
 */
export function mapQuartersToDisplay(quarters: string[]): string[] {
    return quarters.map((q) => getQuarterDisplayLabel(q))
}

/**
 * Map an array of display labels back to backend values
 */
export function mapQuartersToBackend(displayLabels: string[]): string[] {
    return displayLabels.map((q) => getQuarterBackendValue(q))
}
