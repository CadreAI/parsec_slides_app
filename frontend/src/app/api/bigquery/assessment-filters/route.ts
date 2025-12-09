import { BigQuery } from '@google-cloud/bigquery'
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
        const projectId = req.nextUrl.searchParams.get('projectId')
        const datasetId = req.nextUrl.searchParams.get('datasetId')
        const assessments = req.nextUrl.searchParams.get('assessments') || ''
        const location = req.nextUrl.searchParams.get('location') || 'US'

        if (!projectId || !datasetId) {
            return NextResponse.json(
                {
                    success: false,
                    error: 'projectId and datasetId query parameters are required'
                },
                { status: 400 }
            )
        }

        // Set up BigQuery client
        process.env.GOOGLE_CLOUD_PROJECT = projectId

        // Try to resolve service account credentials
        try {
            const { resolveServiceAccountCredentialsPath } = await import('@/lib/credentials')
            const serviceAccountPath = resolveServiceAccountCredentialsPath()
            if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
                process.env.GOOGLE_APPLICATION_CREDENTIALS = serviceAccountPath
                console.log(`[Assessment Filters] Using service account credentials: ${serviceAccountPath}`)
            }
        } catch (credError) {
            console.warn('[Assessment Filters] Could not resolve credentials, will try Application Default Credentials:', credError)
        }

        const client = new BigQuery({
            projectId: projectId,
            location: location
        })

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
            const tablePatterns = assessmentTableMap[assessmentId] || []
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

            // iReady/CERS uses subject column, NWEA uses Course column, STAR uses activity_type column
            if (assessmentId === 'iready' || assessmentId === 'cers') {
                // For iReady/CERS: Use subject column, check for "math" and "ela" (case-insensitive)
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
            } else if (assessmentId === 'star') {
                // For STAR: Use activity_type column, check for "math" and "read" (case-insensitive)
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
                                    subjects.add('Math')
                                }

                                // Check for Reading: activity_type contains "read" (case-insensitive)
                                if (activityStr.includes('read')) {
                                    subjects.add('Reading')
                                }
                            }
                        })
                        if (subjects.size > 0) break
                    } catch {
                        continue
                    }
                }
            } else {
                // For NWEA: Use Course column, check for "reading" and "math" (case-insensitive)
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
                                const courseStr = String(row.course).trim().toLowerCase()

                                // Check for Reading: course contains "reading" (case-insensitive)
                                if (courseStr.includes('reading')) {
                                    subjects.add('Reading')
                                }

                                // Check for Mathematics: course contains "math" (case-insensitive)
                                if (courseStr.includes('math')) {
                                    subjects.add('Mathematics')
                                }
                            }
                        })
                    } catch (error) {
                        console.warn(`[Assessment Filters] Could not query Course column:`, error)
                    }
                }

                // Fallback to subject column if Course column not found or no subjects found
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
                                if (row.subject) {
                                    const subjectStr = String(row.subject).trim().toLowerCase()
                                    if (subjectStr.includes('math')) {
                                        subjects.add('Mathematics')
                                    } else if (subjectStr.includes('reading')) {
                                        subjects.add('Reading')
                                    }
                                }
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

        console.log(`[Assessment Filters] Found ${allSubjects.size} subjects and ${allQuarters.size} quarters from data`)

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
