import { resolveServiceAccountCredentialsPath } from '@/lib/credentials'
import { BigQuery } from '@google-cloud/bigquery'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/bigquery/districts-schools
 *
 * Fetches unique district names and school names from NWEA and cers_iab tables
 *
 * Query params:
 *   - projectId: string (required) - GCP project ID
 *   - datasetId: string (required) - Dataset ID (partner name)
 *   - location?: string (optional) - BigQuery location (default: US)
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

        // Try to find NWEA table
        const nweaTableNames = ['Nwea_production_calpads', 'Nwea_production_calpads_v4_2', 'nwea_production_calpads', 'nwea_production_calpads_v4_2']
        let nweaTableId: string | null = null

        for (const testTableName of nweaTableNames) {
            try {
                const dataset = client.dataset(datasetId)
                const table = dataset.table(testTableName)
                const [exists] = await table.exists()
                if (exists) {
                    nweaTableId = `${projectId}.${datasetId}.${testTableName}`
                    console.log(`Found NWEA table: ${nweaTableId}`)
                    break
                }
            } catch {
                continue
            }
        }

        // Try to find cers_iab table
        const cersIabTableNames = ['cers_iab', 'CERS_IAB', 'cers_iab_production']
        let cersIabTableId: string | null = null

        for (const testTableName of cersIabTableNames) {
            try {
                const dataset = client.dataset(datasetId)
                const table = dataset.table(testTableName)
                const [exists] = await table.exists()
                if (exists) {
                    cersIabTableId = `${projectId}.${datasetId}.${testTableName}`
                    console.log(`Found cers_iab table: ${cersIabTableId}`)
                    break
                }
            } catch {
                continue
            }
        }

        // Try to find STAR table if NWEA is not found
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
        let starTableId: string | null = null

        // Only search for STAR if NWEA is not found
        if (!nweaTableId) {
            for (const testTableName of starTableNames) {
                try {
                    const dataset = client.dataset(datasetId)
                    const table = dataset.table(testTableName)
                    const [exists] = await table.exists()
                    if (exists) {
                        starTableId = `${projectId}.${datasetId}.${testTableName}`
                        console.log(`Found STAR table: ${starTableId}`)
                        break
                    }
                } catch {
                    continue
                }
            }
        }

        if (!nweaTableId && !cersIabTableId && !starTableId) {
            return NextResponse.json(
                {
                    success: false,
                    error: `Neither NWEA, STAR, nor cers_iab table found in dataset ${datasetId}`
                },
                { status: 404 }
            )
        }

        // Extract unique districts and schools from all tables
        const districtSet = new Set<string>()
        const schoolSet = new Set<string>()
        const districtSchoolMap: Record<string, string[]> = {}

        // Query NWEA table if it exists
        if (nweaTableId) {
            const nweaSql = `
                SELECT DISTINCT
                    DistrictName,
                    SchoolName
                FROM \`${nweaTableId}\`
                WHERE DistrictName IS NOT NULL
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
                    const districtName = String(row.DistrictName || row.districtname || row.District_Name || row.district_name || '').trim()
                    const schoolName = String(row.SchoolName || row.schoolname || row.School_Name || row.school_name || '').trim()

                    if (districtName) {
                        districtSet.add(districtName)
                        if (schoolName) {
                            if (!districtSchoolMap[districtName]) {
                                districtSchoolMap[districtName] = []
                            }
                            if (!districtSchoolMap[districtName].includes(schoolName)) {
                                districtSchoolMap[districtName].push(schoolName)
                            }
                            schoolSet.add(schoolName)
                        }
                    }
                }
                console.log(`Found ${nweaRows.length} rows from NWEA table`)
            } catch (error) {
                console.warn(`Error querying NWEA table: ${error}`)
            }
        }

        // Query STAR table if it exists (and NWEA was not found)
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
                        districtSet.add(districtName)
                        if (schoolName) {
                            if (!districtSchoolMap[districtName]) {
                                districtSchoolMap[districtName] = []
                            }
                            if (!districtSchoolMap[districtName].includes(schoolName)) {
                                districtSchoolMap[districtName].push(schoolName)
                            }
                            schoolSet.add(schoolName)
                        }
                    }
                }
                console.log(`Found ${starRows.length} rows from STAR table`)
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
                        districtSet.add(districtName)
                        if (schoolName) {
                            if (!districtSchoolMap[districtName]) {
                                districtSchoolMap[districtName] = []
                            }
                            if (!districtSchoolMap[districtName].includes(schoolName)) {
                                districtSchoolMap[districtName].push(schoolName)
                            }
                            schoolSet.add(schoolName)
                        }
                    }
                }
                console.log(`Found ${cersIabRows.length} rows from cers_iab table`)
            } catch (error) {
                console.warn(`Error querying cers_iab table: ${error}`)
            }
        }

        const districts = Array.from(districtSet).sort()
        const schools = Array.from(schoolSet).sort()

        const sourceTables = []
        if (nweaTableId) sourceTables.push('NWEA')
        if (starTableId) sourceTables.push('STAR')
        if (cersIabTableId) sourceTables.push('cers_iab')
        console.log(`Found ${districts.length} districts and ${schools.length} schools total (from ${sourceTables.join(', ')} tables)`)

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
