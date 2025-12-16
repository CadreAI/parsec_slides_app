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

export function getClusteredSchoolOptions(
    districtName: string,
    districtSchoolMap: Record<string, string[]>,
    clusteredSchools: string[],
    schoolClusters: Record<string, string[]>,
    partnerName: string,
    partnerConfig: Record<string, { districts: string[]; schools: Record<string, string[]> }>
): string[] {
    // If we have clustered schools, use those instead of raw schools
    if (clusteredSchools.length > 0) {
        // If we have a district-specific school map, filter clustered schools to only those in this district
        if (districtName && Object.keys(districtSchoolMap).length > 0) {
            const districtSchools = new Set(districtSchoolMap[districtName] || [])
            // Return clusters that contain at least one school from this district
            return clusteredSchools.filter(clusterName => {
                const schoolsInCluster = schoolClusters[clusterName] || []
                return schoolsInCluster.some(school => districtSchools.has(school))
            })
        }
        return clusteredSchools
    }
    
    // Fallback to regular school options if clustering not available
    return getSchoolOptions(districtName, districtSchoolMap, [], partnerName, partnerConfig)
}
