import { useEffect, useState } from 'react'

export function useDatasets(projectId: string, location: string) {
    const [partnerOptions, setPartnerOptions] = useState<Array<{ value: string; label: string }>>([
        { value: 'demodashboard', label: 'demodashboard (default)' }
    ])
    const [isLoadingDatasets, setIsLoadingDatasets] = useState(false)

    useEffect(() => {
        const fetchDatasets = async () => {
            if (!projectId || projectId.trim() === '') {
                return
            }

            setIsLoadingDatasets(true)
            try {
                const res = await fetch(`/api/bigquery/datasets?projectId=${encodeURIComponent(projectId)}&location=${encodeURIComponent(location)}`)
                const data = await res.json()

                if (res.ok && data.success && data.datasets) {
                    const datasetOptions = data.datasets.map((datasetId: string) => ({
                        value: datasetId,
                        label: datasetId
                    }))
                    setPartnerOptions(datasetOptions)
                    console.log(`Loaded ${datasetOptions.length} datasets from BigQuery`)
                } else {
                    console.warn('Failed to load datasets:', data.error || 'Unknown error')
                }
            } catch (error) {
                console.error('Error fetching datasets:', error)
            } finally {
                setIsLoadingDatasets(false)
            }
        }

        fetchDatasets()
    }, [projectId, location])

    return { partnerOptions, isLoadingDatasets }
}
