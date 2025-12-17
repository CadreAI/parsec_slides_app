import { useEffect, useState } from 'react'

export function useDistrictsAndSchools(
    partnerName: string,
    projectId: string,
    location: string,
    assessments?: string[],
    selectedTables?: Record<string, string>,
    districts?: string[] // Changed from districtName (string) to districts (array)
) {
    const [availableDistricts, setAvailableDistricts] = useState<string[]>([])
    const [availableSchools, setAvailableSchools] = useState<string[]>([])
    const [districtSchoolMap, setDistrictSchoolMap] = useState<Record<string, string[]>>({})
    const [isLoadingDistrictsSchools, setIsLoadingDistrictsSchools] = useState(false)

    // Clustering state
    const [schoolClusters, setSchoolClusters] = useState<Record<string, string[]>>({})
    const [clusteredSchools, setClusteredSchools] = useState<string[]>([])
    const [isLoadingClustering, setIsLoadingClustering] = useState(false)

    useEffect(() => {
        const abortController = new AbortController()

        const fetchDistrictsAndSchools = async () => {
            if (!partnerName || !projectId || partnerName.trim() === '') {
                setAvailableDistricts([])
                setAvailableSchools([])
                setDistrictSchoolMap({})
                setSchoolClusters({})
                setClusteredSchools([])
                return
            }

            // Don't fetch if assessments are required but not selected
            if (assessments && assessments.length === 0) {
                setAvailableDistricts([])
                setAvailableSchools([])
                setDistrictSchoolMap({})
                setSchoolClusters({})
                setClusteredSchools([])
                return
            }

            setIsLoadingDistrictsSchools(true)
            try {
                // Build query params
                const params = new URLSearchParams({
                    projectId: projectId,
                    datasetId: partnerName,
                    location: location
                })

                // Add assessments if provided
                if (assessments && assessments.length > 0) {
                    params.append('assessments', assessments.join(','))
                }

                // Add specific table paths if provided
                if (selectedTables && assessments) {
                    const relevantTables = assessments
                        .map((a) => selectedTables[a])
                        .filter(Boolean)
                        .map((path) => path.split('.').pop()) // Extract table name only
                    if (relevantTables.length > 0) {
                        params.append('tablePaths', relevantTables.join(','))
                    }
                }

                const res = await fetch(`/api/bigquery/districts-schools?${params.toString()}`, { signal: abortController.signal })
                const data = await res.json()

                console.log('[useDistrictsAndSchools] API Response:', {
                    success: data.success,
                    districtsCount: data.districts?.length || 0,
                    schoolsCount: data.schools?.length || 0,
                    districts: data.districts,
                    districtSchoolMapKeys: Object.keys(data.districtSchoolMap || {}),
                    source: data.source || 'unknown',
                    partnerName,
                    assessments
                })

                if (res.ok && data.success) {
                    setAvailableDistricts(data.districts || [])
                    setAvailableSchools(data.schools || [])
                    setDistrictSchoolMap(data.districtSchoolMap || {})
                    console.log(
                        `[useDistrictsAndSchools] ✓ Loaded ${data.districts?.length || 0} districts and ${data.schools?.length || 0} schools from ${data.source || 'unknown source'}`
                    )
                    // Don't cluster yet - wait for district selection
                } else {
                    console.warn('Failed to load districts/schools:', data.error || 'Unknown error')
                    setAvailableDistricts([])
                    setAvailableSchools([])
                    setDistrictSchoolMap({})
                    setSchoolClusters({})
                    setClusteredSchools([])
                }
            } catch (error) {
                // Ignore abort errors - these are intentional cancellations
                if (error instanceof Error && error.name === 'AbortError') {
                    console.log('[useDistrictsAndSchools] Request aborted (expected behavior)')
                    return
                }
                console.error('Error fetching districts and schools:', error)
                setAvailableDistricts([])
                setAvailableSchools([])
                setDistrictSchoolMap({})
                setSchoolClusters({})
                setClusteredSchools([])
            } finally {
                setIsLoadingDistrictsSchools(false)
            }
        }

        fetchDistrictsAndSchools()

        // Cleanup: abort the request if dependencies change or component unmounts
        return () => {
            abortController.abort()
        }
    }, [
        partnerName,
        projectId,
        location,
        assessments?.join(','),
        Object.values(selectedTables || {})
            .sort()
            .join(',')
    ])

    // Separate effect: Trigger clustering when districts are selected
    useEffect(() => {
        const abortController = new AbortController()

        console.log('[Clustering Effect] Triggered:', {
            districts,
            hasDistrictSchoolMap: Object.keys(districtSchoolMap).length > 0,
            districtSchoolMapKeys: Object.keys(districtSchoolMap)
        })

        const fetchSchoolClusters = async (schools: string[], districtNames: string[]) => {
            console.log('[Clustering] Starting for districts:', districtNames.join(', '), 'with', schools.length, 'schools')
            setIsLoadingClustering(true)
            try {
                const res = await fetch('/api/llm/cluster-schools', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        schools: schools,
                        district_name: districtNames.join(', ') // Pass comma-separated district names
                    }),
                    signal: abortController.signal
                })

                const data = await res.json()
                console.log('[Clustering] API Response:', data)

                if (res.ok && data.success) {
                    const clusters = data.clusters || {}
                    setSchoolClusters(clusters)
                    setClusteredSchools(Object.keys(clusters).sort())
                    console.log(`[Clustering] Success: ${schools.length} schools → ${Object.keys(clusters).length} clusters (source: ${data.source})`)
                    console.log('[Clustering] Cluster names:', Object.keys(clusters))
                } else {
                    console.warn('[Clustering] Failed, using identity mapping:', data.error || 'Unknown error')
                    // Fallback: use identity mapping (each school is its own cluster)
                    const identityClusters = Object.fromEntries(schools.map((s) => [s, [s]]))
                    setSchoolClusters(identityClusters)
                    setClusteredSchools(schools.sort())
                }
            } catch (error) {
                // Ignore abort errors - these are intentional cancellations
                if (error instanceof Error && error.name === 'AbortError') {
                    console.log('[Clustering] Request aborted (expected behavior)')
                    return
                }
                console.error('[Clustering] Error:', error)
                // Fallback: use identity mapping
                const identityClusters = Object.fromEntries(schools.map((s) => [s, [s]]))
                setSchoolClusters(identityClusters)
                setClusteredSchools(schools.sort())
            } finally {
                setIsLoadingClustering(false)
            }
        }

        // Only cluster when districts are selected and we have the districtSchoolMap
        if (districts && districts.length > 0 && Object.keys(districtSchoolMap).length > 0) {
            // Aggregate schools from all selected districts
            const allSchools = districts.flatMap((district) => districtSchoolMap[district] || [])
            const uniqueSchools = Array.from(new Set(allSchools))

            console.log('[Clustering] Schools across', districts.length, 'district(s):', uniqueSchools.length, uniqueSchools)
            if (uniqueSchools.length > 0) {
                fetchSchoolClusters(uniqueSchools, districts)
            } else {
                console.log('[Clustering] No schools in selected districts, clearing')
                setSchoolClusters({})
                setClusteredSchools([])
            }
        } else {
            console.log('[Clustering] No districts selected or no map, clearing')
            // Clear clustering when no districts selected
            setSchoolClusters({})
            setClusteredSchools([])
        }

        // Cleanup: abort the request if dependencies change or component unmounts
        return () => {
            abortController.abort()
        }
    }, [districts?.join(','), JSON.stringify(districtSchoolMap)])

    return {
        availableDistricts,
        availableSchools,
        clusteredSchools,
        schoolClusters,
        districtSchoolMap,
        isLoadingDistrictsSchools: isLoadingDistrictsSchools || isLoadingClustering
    }
}
