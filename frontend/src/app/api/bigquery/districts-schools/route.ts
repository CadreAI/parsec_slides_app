import { resolveServiceAccountCredentialsPath } from '@/lib/credentials'
import { BigQuery } from '@google-cloud/bigquery'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/bigquery/districts-schools
 *
 * Fetches unique district names and school names from specified assessment tables
 *
 * Query params:
 *   - projectId: string (required) - GCP project ID
 *   - datasetId: string (required) - Dataset ID (partner name)
 *   - location?: string (optional) - BigQuery location (default: US)
 *   - assessments?: string (optional) - Comma-separated list of assessment IDs (nwea, star, iready)
 *                                       If not provided, queries all available tables
 *
 * Returns:
 *   - success: boolean
 *   - districts: string[] - Array of unique district names
 *   - schools: string[] - Array of unique school names
 */
export async function GET(req: NextRequest) {
    try {
        const projectId = req.nextUrl.searchParams.get('projectId')
        const datasetId = req.nextUrl.searchParams.get('datasetId')
        const location = req.nextUrl.searchParams.get('location') || 'US'
        const assessmentsParam = req.nextUrl.searchParams.get('assessments')

        // Parse assessments if provided
        const requestedAssessments = assessmentsParam ? assessmentsParam.split(',').map((a) => a.trim().toLowerCase()) : null

        if (!projectId || !datasetId) {
            return NextResponse.json(
                {
                    success: false,
                    error: 'projectId and datasetId query parameters are required'
                },
                { status: 400 }
            )
        }

        // Set up credentials
        process.env.GOOGLE_CLOUD_PROJECT = projectId
        try {
            const serviceAccountPath = resolveServiceAccountCredentialsPath()
            if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
                process.env.GOOGLE_APPLICATION_CREDENTIALS = serviceAccountPath
            }
        } catch {
            // Credentials will use ADC if not found
        }

        const client = new BigQuery({
            projectId: projectId,
            location: location
        })

        // Determine which tables to query based on requested assessments
        const shouldQueryNwea = !requestedAssessments || requestedAssessments.includes('nwea')
        const shouldQueryStar = !requestedAssessments || requestedAssessments.includes('star')
        const shouldQueryIready = !requestedAssessments || requestedAssessments.includes('iready')
        const shouldQueryCersIab = !requestedAssessments || requestedAssessments.includes('cers_iab') || requestedAssessments.includes('cers')

        // Try to find NWEA table
        let nweaTableId: string | null = null
        if (shouldQueryNwea) {
            const nweaTableNames = ['Nwea_production_calpads', 'Nwea_production_calpads_v4_2', 'nwea_production_calpads', 'nwea_production_calpads_v4_2']
            for (const testTableName of nweaTableNames) {
                try {
                    const dataset = client.dataset(datasetId)
                    const table = dataset.table(testTableName)
                    const [exists] = await table.exists()
                    if (exists) {
                        nweaTableId = `${projectId}.${datasetId}.${testTableName}`
                        break
                    }
                } catch {
                    continue
                }
            }
        }

        // Try to find STAR table
        let starTableId: string | null = null
        if (shouldQueryStar) {
            const starTableNames = [
                'renaissance_production_calpads_v4_2',
                'Renaissance_production_calpads_v4_2',
                'star_production_calpads_v4_2',
                'star_production_calpads',
                'star_production',
                'star',
                'STAR',
                'renaissance'
            ]
            for (const testTableName of starTableNames) {
                try {
                    const dataset = client.dataset(datasetId)
                    const table = dataset.table(testTableName)
                    const [exists] = await table.exists()
                    if (exists) {
                        starTableId = `${projectId}.${datasetId}.${testTableName}`
                        break
                    }
                } catch {
                    continue
                }
            }
        }

        // Try to find iReady table
        let ireadyTableId: string | null = null
        if (shouldQueryIready) {
            const ireadyTableNames = [
                'iready_production_calpads_v4_2',
                'iReady_production_calpads_v4_2',
                'iready_production_calpads',
                'iready_production',
                'iready',
                'iReady',
                'IREADY'
            ]
            for (const testTableName of ireadyTableNames) {
                try {
                    const dataset = client.dataset(datasetId)
                    const table = dataset.table(testTableName)
                    const [exists] = await table.exists()
                    if (exists) {
                        ireadyTableId = `${projectId}.${datasetId}.${testTableName}`
                        break
                    }
                } catch {
                    continue
                }
            }
        }

        // Try to find cers_iab table
        let cersIabTableId: string | null = null
        if (shouldQueryCersIab) {
            const cersIabTableNames = ['cers_iab', 'CERS_IAB', 'cers_iab_production']
            for (const testTableName of cersIabTableNames) {
                try {
                    const dataset = client.dataset(datasetId)
                    const table = dataset.table(testTableName)
                    const [exists] = await table.exists()
                    if (exists) {
                        cersIabTableId = `${projectId}.${datasetId}.${testTableName}`
                        break
                    }
                } catch {
                    continue
                }
            }
        }

        if (!nweaTableId && !cersIabTableId && !starTableId && !ireadyTableId) {
            const requestedList = requestedAssessments ? requestedAssessments.join(', ') : 'any'
            return NextResponse.json(
                {
                    success: false,
                    error: `No assessment tables found in dataset ${datasetId} for requested assessments: ${requestedList}`
                },
                { status: 404 }
            )
        }

        // Normalize district/school names for deduplication
        // This function normalizes names to handle case differences and variations
        function normalizeName(name: string): string {
            if (!name) return ''
            // Trim whitespace and convert to lowercase for comparison
            return name.trim().toLowerCase()
        }

        // Map normalized name -> canonical name (prefer title case)
        const normalizedToCanonical: Record<string, string> = {}

        // Helper to get or create canonical name
        function getCanonicalName(name: string): string {
            if (!name) return ''
            const normalized = normalizeName(name)
            if (!normalized) return ''

            // If we've seen this normalized name before, return the canonical version
            if (normalizedToCanonical[normalized]) {
                const existingCanonical = normalizedToCanonical[normalized]
                const trimmedName = name.trim()

                // Prefer title case over all caps or all lowercase
                const existingIsAllCaps = existingCanonical === existingCanonical.toUpperCase() && existingCanonical !== existingCanonical.toLowerCase()
                const newIsTitleCase = trimmedName !== trimmedName.toUpperCase() && trimmedName !== trimmedName.toLowerCase()

                // If existing is all caps and new is title case, update canonical
                if (existingIsAllCaps && newIsTitleCase) {
                    normalizedToCanonical[normalized] = trimmedName
                    return trimmedName
                }

                return existingCanonical
            }

            // Create canonical name (use trimmed original)
            const canonical = name.trim()
            normalizedToCanonical[normalized] = canonical
            return canonical
        }

        // Extract unique districts and schools from all tables
        // Use normalized names as keys, but store canonical names
        const districtSet = new Set<string>()
        const schoolSet = new Set<string>()
        const districtSchoolMap: Record<string, string[]> = {}

        // Query NWEA table if it exists
        if (nweaTableId) {
            // Check which columns exist in the table
            let hasLearningCenter = false
            let hasDistrictName = false
            try {
                const tableName = nweaTableId.split('.').pop()!
                const [table] = await client.dataset(datasetId).table(tableName).getMetadata()
                const fields = table.schema?.fields || []
                hasLearningCenter = fields.some(
                    (field: { name?: string }) => field.name?.toLowerCase() === 'learning_center' || field.name?.toLowerCase() === 'learningcenter'
                )
                hasDistrictName = fields.some(
                    (field: { name?: string }) =>
                        field.name?.toLowerCase() === 'districtname' ||
                        field.name?.toLowerCase() === 'district_name' ||
                        field.name?.toLowerCase() === 'district'
                )
            } catch (error) {
                console.warn(`Could not check table schema: ${error}`)
            }

            // Use learning_center if it exists, otherwise fall back to SchoolName
            // BigQuery column names: learning_center (lowercase with underscore), SchoolName (PascalCase)
            const schoolColumn = hasLearningCenter ? `COALESCE(learning_center, SchoolName)` : `SchoolName`

            // Build WHERE clause - only require school column to have values
            // Pull all DistrictName values (including NULL) to get all possible districts
            const whereClause = `WHERE ${schoolColumn} IS NOT NULL`

            const nweaSql = `
                SELECT DISTINCT
                    DistrictName,
                    ${schoolColumn} AS SchoolName
                FROM \`${nweaTableId}\`
                ${whereClause}
            `

            const nweaOptions = {
                query: nweaSql,
                useQueryCache: true,
                useLegacySql: false,
                priority: 'INTERACTIVE' as const
            }

            try {
                const [nweaJob] = await client.createQueryJob(nweaOptions)
                const [nweaRows] = await nweaJob.getQueryResults()

                for (const row of nweaRows) {
                    // NWEA table uses DistrictName column - get all possible values including NULL
                    const districtName = row.DistrictName != null ? String(row.DistrictName).trim() : null
                    // Check for learning_center first (lowercase with underscore), then fall back to SchoolName
                    const schoolName = String(row.learning_center || row.SchoolName || '').trim()

                    if (schoolName) {
                        // Add district name if it exists, otherwise use dataset name for charter schools
                        const effectiveDistrictName = districtName || (hasLearningCenter ? datasetId : null)

                        // Only add district if we have a valid name
                        if (effectiveDistrictName) {
                            // Normalize and get canonical district name
                            const canonicalDistrict = getCanonicalName(effectiveDistrictName)
                            districtSet.add(canonicalDistrict)

                            // Normalize school name for deduplication
                            const normalizedSchool = normalizeName(schoolName)
                            const canonicalSchool = schoolName.trim() // Use original as canonical

                            if (!districtSchoolMap[canonicalDistrict]) {
                                districtSchoolMap[canonicalDistrict] = []
                            }

                            // Check if school already exists (case-insensitive)
                            const existingSchools = districtSchoolMap[canonicalDistrict].map((s) => normalizeName(s))
                            if (!existingSchools.includes(normalizedSchool)) {
                                districtSchoolMap[canonicalDistrict].push(canonicalSchool)
                            }
                        }
                        schoolSet.add(schoolName.trim())
                    }
                }
            } catch (error) {
                console.warn(`Error querying NWEA table: ${error}`)
            }
        }

        // Query iReady table if it exists
        if (ireadyTableId) {
            try {
                // Check which columns exist in the iReady table
                const tableName = ireadyTableId.split('.').pop()!
                const [table] = await client.dataset(datasetId).table(tableName).getMetadata()
                const fields = table.schema?.fields || []

                const hasDistrictName = fields.some(
                    (field: { name?: string }) =>
                        field.name?.toLowerCase() === 'districtname' ||
                        field.name?.toLowerCase() === 'district_name' ||
                        field.name?.toLowerCase() === 'district'
                )
                const hasSchool = fields.some(
                    (field: { name?: string }) =>
                        field.name?.toLowerCase() === 'school' || field.name?.toLowerCase() === 'schoolname' || field.name?.toLowerCase() === 'school_name'
                )
                const hasLearningCenter = fields.some(
                    (field: { name?: string }) => field.name?.toLowerCase() === 'learning_center' || field.name?.toLowerCase() === 'learningcenter'
                )

                // Build SQL based on available columns
                let districtColumn: string
                let schoolColumn: string
                let whereClause: string

                if (hasDistrictName) {
                    // Standard case: use DistrictName
                    districtColumn = 'DistrictName'
                    // For schools, prioritize learning_center if available
                    if (hasLearningCenter) {
                        schoolColumn = 'COALESCE(learning_center, SchoolName, School_Name, school)'
                    } else {
                        schoolColumn = 'COALESCE(SchoolName, School_Name, school)'
                    }
                    whereClause = 'WHERE DistrictName IS NOT NULL'
                } else if (hasSchool) {
                    // Charter school case: use School as district identifier
                    districtColumn = 'School'
                    // Use learning_center for schools if available
                    if (hasLearningCenter) {
                        schoolColumn = 'learning_center'
                        whereClause = 'WHERE School IS NOT NULL AND learning_center IS NOT NULL'
                    } else {
                        // Fallback: use School as both district and school (distinct values)
                        schoolColumn = 'School'
                        whereClause = 'WHERE School IS NOT NULL'
                    }
                } else {
                    // Fallback: try common column names
                    districtColumn = 'COALESCE(DistrictName, District_Name, districtname, district_name)'
                    schoolColumn = 'COALESCE(SchoolName, School_Name, schoolname, school_name, learning_center)'
                    whereClause = 'WHERE ' + districtColumn + ' IS NOT NULL'
                }

                const ireadySql = `
                    SELECT DISTINCT
                        ${districtColumn} AS DistrictName,
                        ${schoolColumn} AS SchoolName
                    FROM \`${ireadyTableId}\`
                    ${whereClause}
                `

                const ireadyOptions = {
                    query: ireadySql,
                    useQueryCache: true,
                    useLegacySql: false,
                    priority: 'INTERACTIVE' as const
                }

                const [ireadyJob] = await client.createQueryJob(ireadyOptions)
                const [ireadyRows] = await ireadyJob.getQueryResults()

                for (const row of ireadyRows) {
                    const districtName = String(row.DistrictName || '').trim()
                    const schoolName = String(row.SchoolName || '').trim()

                    if (districtName) {
                        // Normalize and get canonical district name
                        const canonicalDistrict = getCanonicalName(districtName)
                        districtSet.add(canonicalDistrict)

                        if (schoolName && schoolName !== districtName) {
                            // Only add school if it's different from district (for charter schools using School as district)
                            // Normalize school name for deduplication
                            const normalizedSchool = normalizeName(schoolName)
                            const canonicalSchool = schoolName.trim() // Use original as canonical

                            if (!districtSchoolMap[canonicalDistrict]) {
                                districtSchoolMap[canonicalDistrict] = []
                            }

                            // Check if school already exists (case-insensitive)
                            const existingSchools = districtSchoolMap[canonicalDistrict].map((s) => normalizeName(s))
                            if (!existingSchools.includes(normalizedSchool)) {
                                districtSchoolMap[canonicalDistrict].push(canonicalSchool)
                            }
                            schoolSet.add(canonicalSchool)
                        }
                    }
                }
            } catch (error) {
                console.warn(`Error querying iReady table: ${error}`)
            }
        }

        // Query STAR table if it exists
        if (starTableId) {
            // STAR uses District_Name and School_Name (with underscores)
            const starSql = `
                SELECT DISTINCT
                    District_Name AS DistrictName,
                    School_Name AS SchoolName
                FROM \`${starTableId}\`
                WHERE District_Name IS NOT NULL
            `

            const starOptions = {
                query: starSql,
                useQueryCache: true,
                useLegacySql: false,
                priority: 'INTERACTIVE' as const
            }

            try {
                const [starJob] = await client.createQueryJob(starOptions)
                const [starRows] = await starJob.getQueryResults()

                for (const row of starRows) {
                    // STAR uses District_Name and School_Name, but we alias them as DistrictName/SchoolName in SQL
                    const districtName = String(row.DistrictName || row.District_Name || '').trim()
                    const schoolName = String(row.SchoolName || row.School_Name || '').trim()

                    if (districtName) {
                        // Normalize and get canonical district name
                        const canonicalDistrict = getCanonicalName(districtName)
                        districtSet.add(canonicalDistrict)

                        if (schoolName) {
                            // Normalize school name for deduplication
                            const normalizedSchool = normalizeName(schoolName)
                            const canonicalSchool = schoolName.trim() // Use original as canonical

                            if (!districtSchoolMap[canonicalDistrict]) {
                                districtSchoolMap[canonicalDistrict] = []
                            }

                            // Check if school already exists (case-insensitive)
                            const existingSchools = districtSchoolMap[canonicalDistrict].map((s) => normalizeName(s))
                            if (!existingSchools.includes(normalizedSchool)) {
                                districtSchoolMap[canonicalDistrict].push(canonicalSchool)
                            }
                            schoolSet.add(canonicalSchool)
                        }
                    }
                }
            } catch (error) {
                console.warn(`Error querying STAR table: ${error}`)
            }
        }

        // Query cers_iab table if it exists
        if (cersIabTableId) {
            // cers_iab may use lowercase column names (districtname, schoolname) or PascalCase
            // Use COALESCE to handle both cases
            const cersIabSql = `
                SELECT DISTINCT
                    COALESCE(DistrictName, districtname) AS DistrictName,
                    COALESCE(SchoolName, schoolname) AS SchoolName
                FROM \`${cersIabTableId}\`
                WHERE COALESCE(DistrictName, districtname) IS NOT NULL
            `

            const cersIabOptions = {
                query: cersIabSql,
                useQueryCache: true,
                useLegacySql: false,
                priority: 'INTERACTIVE' as const
            }

            try {
                const [cersIabJob] = await client.createQueryJob(cersIabOptions)
                const [cersIabRows] = await cersIabJob.getQueryResults()

                for (const row of cersIabRows) {
                    const districtName = String(row.DistrictName || row.districtname || row.District_Name || row.district_name || '').trim()
                    const schoolName = String(row.SchoolName || row.schoolname || row.School_Name || row.school_name || '').trim()

                    if (districtName) {
                        // Normalize and get canonical district name
                        const canonicalDistrict = getCanonicalName(districtName)
                        districtSet.add(canonicalDistrict)

                        if (schoolName) {
                            // Normalize school name for deduplication
                            const normalizedSchool = normalizeName(schoolName)
                            const canonicalSchool = schoolName.trim() // Use original as canonical

                            if (!districtSchoolMap[canonicalDistrict]) {
                                districtSchoolMap[canonicalDistrict] = []
                            }

                            // Check if school already exists (case-insensitive)
                            const existingSchools = districtSchoolMap[canonicalDistrict].map((s) => normalizeName(s))
                            if (!existingSchools.includes(normalizedSchool)) {
                                districtSchoolMap[canonicalDistrict].push(canonicalSchool)
                            }
                            schoolSet.add(canonicalSchool)
                        }
                    }
                }
            } catch (error) {
                console.warn(`Error querying cers_iab table: ${error}`)
            }
        }

        const districts = Array.from(districtSet).sort()
        const schools = Array.from(schoolSet).sort()

        const sourceTables = []
        if (nweaTableId) sourceTables.push('NWEA')
        if (starTableId) sourceTables.push('STAR')
        if (ireadyTableId) sourceTables.push('iReady')
        if (cersIabTableId) sourceTables.push('cers_iab')

        return NextResponse.json({
            success: true,
            districts: districts,
            schools: schools,
            districtSchoolMap: districtSchoolMap // Map of district -> schools for filtering
        })
    } catch (error: unknown) {
        console.error('Error fetching districts and schools:', error)
        const errorMessage = error instanceof Error ? error.message : 'Failed to fetch districts and schools'
        return NextResponse.json(
            {
                success: false,
                error: errorMessage
            },
            { status: 500 }
        )
    }
}
