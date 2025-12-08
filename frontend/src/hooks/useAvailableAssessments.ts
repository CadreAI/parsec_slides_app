import { useEffect, useState } from 'react'

export interface AssessmentSource {
    id: string
    label: string
    defaultTable: string
}

// Default table patterns - can be overridden by backend
const DEFAULT_ASSESSMENT_CONFIG: Record<string, Omit<AssessmentSource, 'id'>> = {
    nwea: { label: 'NWEA Map Growth', defaultTable: 'parsecgo.demodashboard.Nwea_production_calpads_v4_2' },
    iready: { label: 'iReady', defaultTable: 'parsecgo.demodashboard.iready_production_calpads_v4_2' },
    star: { label: 'STAR', defaultTable: 'parsecgo.demodashboard.renaissance_production_calpads_v4_2' },
    cers: { label: 'CERS', defaultTable: 'parsecgo.demodashboard.cers_production' }
}

export function useAvailableAssessments() {
    const [assessments, setAssessments] = useState<AssessmentSource[]>([])
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        const fetchAssessments = async () => {
            setIsLoading(true)
            try {
                // Fetch all assessment filters (without specifying assessments to get all)
                const res = await fetch('/api/config/assessment-filters')
                if (res.ok) {
                    const data = await res.json()
                    if (data.success && data.assessment_details) {
                        // Build assessment sources from backend response
                        const assessmentSources: AssessmentSource[] = Object.keys(data.assessment_details).map((id) => {
                            const config = DEFAULT_ASSESSMENT_CONFIG[id] || {
                                label: id.charAt(0).toUpperCase() + id.slice(1),
                                defaultTable: `parsecgo.demodashboard.${id}_production`
                            }
                            return {
                                id,
                                ...config
                            }
                        })
                        setAssessments(assessmentSources)
                    } else {
                        // Fallback to defaults if backend doesn't return assessment_details
                        setAssessments(
                            Object.entries(DEFAULT_ASSESSMENT_CONFIG).map(([id, config]) => ({
                                id,
                                ...config
                            }))
                        )
                    }
                } else {
                    // Fallback to defaults on error
                    setAssessments(
                        Object.entries(DEFAULT_ASSESSMENT_CONFIG).map(([id, config]) => ({
                            id,
                            ...config
                        }))
                    )
                }
            } catch (error) {
                console.error('Error fetching available assessments:', error)
                // Fallback to defaults on error
                setAssessments(
                    Object.entries(DEFAULT_ASSESSMENT_CONFIG).map(([id, config]) => ({
                        id,
                        ...config
                    }))
                )
            } finally {
                setIsLoading(false)
            }
        }

        fetchAssessments()
    }, [])

    return { assessments, isLoading }
}
