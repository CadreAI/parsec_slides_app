export function getDistrictOptions(
    availableDistricts: string[],
    partnerName: string,
    partnerConfig: Record<string, { districts: string[]; schools: Record<string, string[]> }>
): string[] {
    if (availableDistricts.length > 0) {
        return availableDistricts
    }
    if (!partnerName || !partnerConfig[partnerName]) {
        return []
    }
    return partnerConfig[partnerName].districts
}

export function getSchoolOptions(
    districtName: string,
    districtSchoolMap: Record<string, string[]>,
    availableSchools: string[],
    partnerName: string,
    partnerConfig: Record<string, { districts: string[]; schools: Record<string, string[]> }>
): string[] {
    if (districtName && Object.keys(districtSchoolMap).length > 0) {
        return districtSchoolMap[districtName] || []
    }
    if (availableSchools.length > 0) {
        return availableSchools
    }
    if (!partnerName || !partnerConfig[partnerName]) {
        return []
    }
    return partnerConfig[partnerName].schools[districtName] || []
}
