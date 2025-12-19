import { createBigQueryClient } from '@/lib/bigquery'
import { auth } from '@clerk/nextjs/server'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/bigquery/assessment-filters
 *
 * Fetches available subjects and quarters from actual data tables
 *
 * Query params:
 *   - projectId: string (required) - GCP project ID
 *   - datasetId: string (required) - BigQuery dataset ID
 *   - assessments: string (required) - Comma-separated list of assessment IDs (e.g., "nwea,iready")
 *   - location?: string (optional) - BigQuery location (default: US)
 *
 * Returns:
 *   - success: boolean
 *   - filters: { subjects: string[], quarters: string[], supports_grades: boolean, supports_student_groups: boolean, supports_race: boolean }
 *   - assessment_details: Record<string, assessment filter config>
 */
export async function GET(req: NextRequest) {
    try {
        const { userId } = await auth()
        if (!userId) {
            return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
        }

        const projectId = req.nextUrl.searchParams.get('projectId')
        const datasetId = req.nextUrl.searchParams.get('datasetId')
        const assessments = req.nextUrl.searchParams.get('assessments') || ''
        const location = req.nextUrl.searchParams.get('location') || 'US'
        const tablePathsParam = req.nextUrl.searchParams.get('tablePaths')

        // Parse specific table paths if provided
        const specificTablePaths = tablePathsParam ? tablePathsParam.split(',').map((t) => t.trim()) : null

        if (!projectId || !datasetId) {
            return NextResponse.json(
                {
                    success: false,
                    error: 'projectId and datasetId query parameters are required'
                },
                { status: 400 }
            )
        }

        const client = createBigQueryClient(projectId, location)

        const dataset = client.dataset(datasetId)

        // Map assessment IDs to table name patterns
        const assessmentTableMap: Record<string, string[]> = {
            nwea: ['nwea_production_calpads_v4_2', 'Nwea_production_calpads_v4_2', 'nwea_production', 'nwea'],
            iready: ['iready_production_calpads_v4_2', 'iReady_production_calpads_v4_2', 'iready_production', 'iready'],
            star: ['renaissance_production_calpads_v4_2', 'Renaissance_production_calpads_v4_2', 'star_production', 'star', 'renaissance'],
            cers: ['cers_production', 'cers', 'CERS']
        }

        const requestedAssessments = assessments ? assessments.split(',').map((a) => a.trim()) : []
        const allSubjects = new Set<string>()
        const allQuarters = new Set<string>()
        interface AssessmentDetail {
            subjects: string[]
            quarters: string[]
            supports_grades: boolean
            supports_student_groups: boolean
            supports_race: boolean
        }
        const assessmentDetails: Record<string, AssessmentDetail> = {}

        // Query each requested assessment
        for (const assessmentId of requestedAssessments) {
            let tablePatterns = assessmentTableMap[assessmentId] || []

            // If specific table paths are provided, filter to relevant tables for this assessment
            if (specificTablePaths) {
                const relevantTables = specificTablePaths.filter((path) => {
                    const tableName = path.split('.').pop() || path
                    return tableName.toLowerCase().includes(assessmentId.toLowerCase())
                })
                if (relevantTables.length > 0) {
                    tablePatterns = relevantTables.map((path) => path.split('.').pop() || path)
                }
            }

            let foundTable: string | null = null

            // Find the table for this assessment
            for (const tableName of tablePatterns) {
                try {
                    const table = dataset.table(tableName)
                    const [exists] = await table.exists()
                    if (exists) {
                        foundTable = tableName
                        break
                    }
                } catch {
                    continue
                }
            }

            if (!foundTable) {
                console.warn(`[Assessment Filters] Table not found for ${assessmentId}`)
                continue
            }

            // Try to get subjects and quarters from the table
            const subjects = new Set<string>()
            const quarters = new Set<string>()

            const titleCase = (s: string) =>
                s
                    .toLowerCase()
                    .split(/\s+/)
                    .filter(Boolean)
                    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
                    .join(' ')

            const normalizeNweaSubject = (raw: string): string | null => {
                const cleaned = String(raw || '').trim()
                if (!cleaned) return null
                const s = cleaned.toLowerCase()
                if (s.includes('math')) return 'Mathematics'
                if (s.includes('reading') || s.includes('ela') || s.includes('language arts') || s.includes('language_arts')) return 'Reading'
                // Keep other MAP Growth subjects as distinct selectable sections (e.g., Science, Language Usage)
                const base = cleaned.split(' - ')[0].split('-')[0].split(':')[0].trim()
                return titleCase(base)
            }

            // iReady/CERS uses subject column, NWEA uses Course column, STAR uses activity_type column
            if (assessmentId === 'iready' || assessmentId === 'cers') {
                // For iReady/CERS: Prefer subject column if present; fallback to activity_type if subject isn't available.
                const subjectColumns = ['subject', 'Subject']
                for (const subjectCol of subjectColumns) {
                    try {
                        const query = `
                            SELECT DISTINCT \`${subjectCol}\` as subject
                            FROM \`${projectId}.${datasetId}.${foundTable}\`
                            WHERE \`${subjectCol}\` IS NOT NULL
                            LIMIT 100
                        `
                        const [rows] = await client.query({ query, location })

                        // Filter subjects based on subject column content (like iReady does)
                        rows.forEach((row: { subject?: string }) => {
                            if (row.subject) {
                                const subjectStr = String(row.subject).trim().toLowerCase()

                                // Check for Math: subject contains "math" (case-insensitive)
                                if (subjectStr.includes('math')) {
                                    subjects.add('Math')
                                }

                                // Check for ELA: subject contains "ela" (case-insensitive)
                                if (subjectStr.includes('ela')) {
                                    subjects.add('ELA')
                                }
                            }
                        })
                        if (subjects.size > 0) break
                    } catch {
                        continue
                    }
                }

                // Fallback (some iReady exports may not have subject, but have an activity_type-like column)
                if (assessmentId === 'iready' && subjects.size === 0) {
                    const activityTypeColumns = ['activity_type', 'Activity_Type', 'ActivityType']
                    for (const activityCol of activityTypeColumns) {
                        try {
                            const query = `
                                SELECT DISTINCT \`${activityCol}\` as activity_type
                                FROM \`${projectId}.${datasetId}.${foundTable}\`
                                WHERE \`${activityCol}\` IS NOT NULL
                                LIMIT 100
                            `
                            const [rows] = await client.query({ query, location })
                            rows.forEach((row: { activity_type?: string }) => {
                                if (!row.activity_type) return
                                const s = String(row.activity_type).trim().toLowerCase()
                                if (s.includes('math')) subjects.add('Math')
                                if (s.includes('ela') || s.includes('reading')) subjects.add('ELA')
                            })
                            if (subjects.size > 0) break
                        } catch {
                            continue
                        }
                    }
                }
            } else if (assessmentId === 'star') {
                // For STAR: Prefer subject column if present; fallback to activity_type.

                // 1) Try subject column(s) first (some STAR exports include this).
                const subjectColumns = ['subject', 'Subject']
                for (const subjectCol of subjectColumns) {
                    try {
                        const query = `
                            SELECT DISTINCT \`${subjectCol}\` as subject
                            FROM \`${projectId}.${datasetId}.${foundTable}\`
                            WHERE \`${subjectCol}\` IS NOT NULL
                            LIMIT 100
                        `
                        const [rows] = await client.query({ query, location })

                        rows.forEach((row: { subject?: string }) => {
                            if (!row.subject) return
                            const subjStr = String(row.subject).trim().toLowerCase()

                            if (subjStr.includes('math')) {
                                subjects.add('Mathematics')
                            }

                            if (subjStr.includes('read')) {
                                subjects.add('Reading')
                            }

                            // Spanish reading as separate option if present
                            if (subjStr.includes('read') && (subjStr.includes('spanish') || subjStr.includes('españ') || subjStr.includes('espanol'))) {
                                subjects.add('Reading (Spanish)')
                            }
                        })
                        if (subjects.size > 0) break // we found enough via subject col
                    } catch {
                        continue
                    }
                }

                // 2) Fallback to activity_type if subject wasn't found/usable
                if (subjects.size === 0) {
                    const activityTypeColumns = ['activity_type', 'Activity_Type', 'ActivityType']
                    for (const activityCol of activityTypeColumns) {
                        try {
                            const query = `
                                SELECT DISTINCT \`${activityCol}\` as activity_type
                                FROM \`${projectId}.${datasetId}.${foundTable}\`
                                WHERE \`${activityCol}\` IS NOT NULL
                                LIMIT 100
                            `
                            const [rows] = await client.query({ query, location })

                            // Filter subjects based on activity_type column content
                            rows.forEach((row: { activity_type?: string }) => {
                                if (row.activity_type) {
                                    const activityStr = String(row.activity_type).trim().toLowerCase()

                                    // Check for Math: activity_type contains "math" (case-insensitive)
                                    if (activityStr.includes('math')) {
                                        // Use "Mathematics" for UI consistency with other assessments
                                        subjects.add('Mathematics')
                                    }

                                    // Check for Reading: activity_type contains "read" (case-insensitive)
                                    if (activityStr.includes('read')) {
                                        subjects.add('Reading')
                                    }

                                    // Spanish reading (keep as a separate selectable option if present in data)
                                    if (
                                        activityStr.includes('read') &&
                                        (activityStr.includes('spanish') || activityStr.includes('españ') || activityStr.includes('espanol'))
                                    ) {
                                        subjects.add('Reading (Spanish)')
                                    }
                                }
                            })
                            if (subjects.size > 0) break
                        } catch {
                            continue
                        }
                    }
                }
            } else {
                // For NWEA: Prefer subject column if present; fallback to Course column.

                // 1) Try subject column(s) first
                const subjectColumns = ['subject', 'Subject']
                for (const subjectCol of subjectColumns) {
                    try {
                        const query = `
                            SELECT DISTINCT \`${subjectCol}\` as subject
                            FROM \`${projectId}.${datasetId}.${foundTable}\`
                            WHERE \`${subjectCol}\` IS NOT NULL
                            LIMIT 100
                        `
                        const [rows] = await client.query({ query, location })
                        rows.forEach((row: { subject?: string }) => {
                            if (!row.subject) return
                            const normalized = normalizeNweaSubject(row.subject)
                            if (normalized) subjects.add(normalized)
                        })
                        if (subjects.size > 0) break
                    } catch {
                        continue
                    }
                }

                // 2) Fallback to Course column if needed
                if (subjects.size === 0) {
                    const courseColumns = ['Course', 'course']
                    let foundCourseColumn: string | null = null

                    // Check if Course column exists
                    for (const courseCol of courseColumns) {
                        try {
                            const testQuery = `SELECT \`${courseCol}\` FROM \`${projectId}.${datasetId}.${foundTable}\` LIMIT 1`
                            await client.query({ query: testQuery, location })
                            foundCourseColumn = courseCol
                            break
                        } catch {
                            continue
                        }
                    }

                    if (foundCourseColumn) {
                        try {
                            // Query distinct Course values
                            const query = `
                                SELECT DISTINCT \`${foundCourseColumn}\` as course
                                FROM \`${projectId}.${datasetId}.${foundTable}\`
                                WHERE \`${foundCourseColumn}\` IS NOT NULL
                                LIMIT 100
                            `
                            const [rows] = await client.query({ query, location })

                            // Filter subjects based on Course column content (like NWEA does)
                            rows.forEach((row: { course?: string }) => {
                                if (row.course) {
                                    const normalized = normalizeNweaSubject(row.course)
                                    if (normalized) subjects.add(normalized)
                                }
                            })
                        } catch (error) {
                            console.warn(`[Assessment Filters] Could not query Course column:`, error)
                        }
                    }
                }

                // Final fallback to subject column if still empty (kept for backward compatibility)
                if (subjects.size === 0) {
                    const subjectColumns = ['subject', 'Subject']
                    for (const subjectCol of subjectColumns) {
                        try {
                            const query = `
                                SELECT DISTINCT \`${subjectCol}\` as subject
                                FROM \`${projectId}.${datasetId}.${foundTable}\`
                                WHERE \`${subjectCol}\` IS NOT NULL
                                LIMIT 50
                            `
                            const [rows] = await client.query({ query, location })
                            rows.forEach((row: { subject?: string }) => {
                                const normalized = row.subject ? normalizeNweaSubject(row.subject) : null
                                if (normalized) subjects.add(normalized)
                            })
                            if (subjects.size > 0) break
                        } catch {
                            continue
                        }
                    }
                }
            }

            // Try different column name patterns for quarters/test windows
            const quarterColumns = ['testwindow', 'TestWindow', 'test_window', 'window', 'termname', 'TermName']
            for (const quarterCol of quarterColumns) {
                try {
                    const query = `
                        SELECT DISTINCT \`${quarterCol}\` as quarter
                        FROM \`${projectId}.${datasetId}.${foundTable}\`
                        WHERE \`${quarterCol}\` IS NOT NULL
                        LIMIT 50
                    `
                    const [rows] = await client.query({ query, location })
                    rows.forEach((row: { quarter?: string }) => {
                        if (row.quarter) {
                            const quarterStr = String(row.quarter).trim()
                            // Normalize quarter names
                            const quarterUpper = quarterStr.toUpperCase()
                            if (quarterUpper.includes('FALL')) {
                                quarters.add('Fall')
                            } else if (quarterUpper.includes('WINTER')) {
                                quarters.add('Winter')
                            } else if (quarterUpper.includes('SPRING')) {
                                quarters.add('Spring')
                            }
                        }
                    })
                    if (quarters.size > 0) break
                } catch {
                    continue
                }
            }

            // Merge into overall sets
            subjects.forEach((s) => allSubjects.add(s))
            quarters.forEach((q) => allQuarters.add(q))

            // Store assessment details
            assessmentDetails[assessmentId] = {
                subjects: Array.from(subjects).sort(),
                quarters: Array.from(quarters).sort(),
                supports_grades: true,
                supports_student_groups: true,
                supports_race: true
            }
        }

        // Fallback to defaults if nothing found
        if (allSubjects.size === 0) {
            allSubjects.add('Reading')
            allSubjects.add('Mathematics')
            allSubjects.add('ELA')
            allSubjects.add('Math')
        }
        if (allQuarters.size === 0) {
            allQuarters.add('Fall')
            allQuarters.add('Winter')
            allQuarters.add('Spring')
        }

        // If no assessment details found, use fallback
        if (Object.keys(assessmentDetails).length === 0 && requestedAssessments.length > 0) {
            const fallbackFilters: Record<string, AssessmentDetail> = {
                nwea: {
                    subjects: ['Reading', 'Mathematics'],
                    quarters: ['Fall', 'Winter', 'Spring'],
                    supports_grades: true,
                    supports_student_groups: true,
                    supports_race: true
                },
                iready: {
                    subjects: ['ELA', 'Math'],
                    quarters: ['Fall', 'Winter', 'Spring'],
                    supports_grades: true,
                    supports_student_groups: true,
                    supports_race: true
                },
                star: {
                    subjects: ['Reading', 'Mathematics'],
                    quarters: ['Fall', 'Winter', 'Spring'],
                    supports_grades: true,
                    supports_student_groups: true,
                    supports_race: true
                },
                cers: { subjects: ['ELA', 'Math'], quarters: [], supports_grades: true, supports_student_groups: true, supports_race: true }
            }
            requestedAssessments.forEach((assessmentId) => {
                if (fallbackFilters[assessmentId]) {
                    assessmentDetails[assessmentId] = fallbackFilters[assessmentId]
                    assessmentDetails[assessmentId].supports_grades = true
                    assessmentDetails[assessmentId].supports_student_groups = true
                    assessmentDetails[assessmentId].supports_race = true
                }
            })
        }

        return NextResponse.json({
            success: true,
            filters: {
                subjects: Array.from(allSubjects).sort(),
                quarters: Array.from(allQuarters).sort(),
                supports_grades: true,
                supports_student_groups: true,
                supports_race: true
            },
            assessment_details: assessmentDetails
        })
    } catch (error: unknown) {
        console.error('Error fetching assessment filters from BigQuery:', error)

        // Fallback to defaults
        const assessments = req.nextUrl.searchParams.get('assessments') || ''
        const requestedAssessments = assessments ? assessments.split(',').map((a) => a.trim()) : []
        interface AssessmentDetail {
            subjects: string[]
            quarters: string[]
            supports_grades: boolean
            supports_student_groups: boolean
            supports_race: boolean
        }
        const fallbackFilters: Record<string, AssessmentDetail> = {
            nwea: {
                subjects: ['Reading', 'Mathematics'],
                quarters: ['Fall', 'Winter', 'Spring'],
                supports_grades: true,
                supports_student_groups: true,
                supports_race: true
            },
            iready: {
                subjects: ['ELA', 'Math'],
                quarters: ['Fall', 'Winter', 'Spring'],
                supports_grades: true,
                supports_student_groups: true,
                supports_race: true
            },
            star: {
                subjects: ['Reading', 'Mathematics'],
                quarters: ['Fall', 'Winter', 'Spring'],
                supports_grades: true,
                supports_student_groups: true,
                supports_race: true
            },
            cers: { subjects: ['ELA', 'Math'], quarters: [], supports_grades: true, supports_student_groups: true, supports_race: true }
        }

        const allSubjects = new Set<string>()
        const allQuarters = new Set<string>()
        const assessmentDetails: Record<string, AssessmentDetail> = {}

        requestedAssessments.forEach((assessmentId) => {
            const config = fallbackFilters[assessmentId]
            if (config) {
                config.subjects?.forEach((s: string) => allSubjects.add(s))
                config.quarters?.forEach((q: string) => allQuarters.add(q))
                assessmentDetails[assessmentId] = {
                    ...config,
                    supports_grades: true,
                    supports_student_groups: true,
                    supports_race: true
                }
            }
        })

        return NextResponse.json({
            success: true,
            filters: {
                subjects: Array.from(allSubjects).sort(),
                quarters: Array.from(allQuarters).sort(),
                supports_grades: true,
                supports_student_groups: true,
                supports_race: true
            },
            assessment_details: assessmentDetails
        })
    }
}
