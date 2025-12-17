'use client'

import { Label } from '@/components/ui/label'
import { MultiSelect } from '@/components/ui/multi-select'
import { Checkbox } from '@/components/ui/checkbox'
import { useDistrictsAndSchools } from '@/hooks/useDistrictsAndSchools'
import { getDistrictOptions, getClusteredSchoolOptions } from '@/utils/formHelpers'
import { useState, useEffect } from 'react'

interface AssessmentScopeSelectorProps {
    assessmentId: string
    partnerName: string
    projectId: string
    location: string
    customDataSource?: string
    scope: { districts: string[]; schools: string[]; districtOnly?: boolean }
    onScopeChange: (assessmentId: string, scope: { districts: string[]; schools: string[]; districtOnly?: boolean }) => void
    partnerConfig: Record<string, { districts: string[]; schools: Record<string, string[]> }>
}

const PARTNER_CONFIG: Record<string, { districts: string[]; schools: Record<string, string[]> }> = {
    demodashboard: {
        districts: ['Parsec Academy'],
        schools: {
            'Parsec Academy': ['Parsec Academy']
        }
    }
}

export function AssessmentScopeSelector({
    assessmentId,
    partnerName,
    projectId,
    location,
    customDataSource,
    scope,
    onScopeChange,
    partnerConfig
}: AssessmentScopeSelectorProps) {
    // Fetch districts and schools for this specific assessment
    const { availableDistricts, availableSchools, clusteredSchools, schoolClusters, districtSchoolMap, isLoadingDistrictsSchools } = useDistrictsAndSchools(
        partnerName,
        projectId,
        location,
        [assessmentId], // Only fetch for this assessment
        customDataSource ? { [assessmentId]: customDataSource } : {},
        scope.districts
    )

    // Detect if we actually have districts or if "districts" are really schools (charter school case)
    const hasActualDistricts = Object.keys(districtSchoolMap).length > 0

    // Get district options
    const districtOptions = getDistrictOptions(availableDistricts, partnerName, partnerConfig)

    // Aggregate schools from all selected districts
    const aggregatedSchools = scope.districts.flatMap((district) => districtSchoolMap[district] || [])
    const uniqueSchools = Array.from(new Set(aggregatedSchools)).sort()

    // Determine school options
    const effectiveSchoolOptions = hasActualDistricts ? availableSchools : availableDistricts
    const schoolOptions = uniqueSchools.length > 0 ? uniqueSchools : effectiveSchoolOptions

    const clusteredSchoolOptions = getClusteredSchoolOptions(
        scope.districts.join(','),
        districtSchoolMap,
        clusteredSchools,
        schoolClusters,
        partnerName,
        partnerConfig
    )

    const finalSchoolOptions = clusteredSchools.length > 0 ? clusteredSchoolOptions : schoolOptions

    return (
        <div className="ml-6 mt-3 space-y-3 rounded-lg border-l-2 border-blue-200 bg-blue-50/30 p-3">
            <div className="text-xs font-semibold uppercase tracking-wide text-gray-600">📍 Scope for {assessmentId}</div>

            {/* District Selection - Only show if we have actual districts */}
            {hasActualDistricts && (
                <div className="space-y-2">
                    <Label className="text-xs">District(s)</Label>
                    <MultiSelect
                        options={districtOptions}
                        selected={scope.districts}
                        onChange={(selected) => {
                            onScopeChange(assessmentId, {
                                ...scope,
                                districts: selected,
                                schools: [] // Reset schools when districts change
                            })
                        }}
                        placeholder={isLoadingDistrictsSchools ? 'Loading districts...' : 'Select district(s)...'}
                        disabled={isLoadingDistrictsSchools || !partnerName}
                    />
                </div>
            )}

            {/* District Only Mode */}
            {hasActualDistricts && scope.districts.length > 0 && (
                <div className="flex items-start space-x-2">
                    <Checkbox
                        id={`${assessmentId}-districtOnly`}
                        checked={scope.districtOnly || false}
                        onChange={(e) => {
                            onScopeChange(assessmentId, {
                                ...scope,
                                districtOnly: e.target.checked,
                                schools: e.target.checked ? [] : scope.schools
                            })
                        }}
                        disabled={scope.districts.length === 0 || isLoadingDistrictsSchools}
                    />
                    <div className="flex-1">
                        <Label htmlFor={`${assessmentId}-districtOnly`} className="cursor-pointer text-xs font-medium">
                            District Only Mode
                        </Label>
                        <p className="text-xs text-gray-600">Generate charts for district-level data only (no school-specific charts)</p>
                    </div>
                </div>
            )}

            {/* School Selection */}
            {((hasActualDistricts && !scope.districtOnly && scope.districts.length > 0) || !hasActualDistricts) && (
                <div className="space-y-2">
                    <Label className="text-xs">School(s)</Label>
                    <MultiSelect
                        options={finalSchoolOptions}
                        selected={scope.schools}
                        onChange={(selected) => {
                            onScopeChange(assessmentId, {
                                ...scope,
                                schools: selected
                            })
                        }}
                        placeholder={
                            isLoadingDistrictsSchools
                                ? 'Loading schools...'
                                : hasActualDistricts && scope.districts.length === 0
                                  ? 'Select district(s) first...'
                                  : 'Select school(s)...'
                        }
                        disabled={isLoadingDistrictsSchools || !partnerName || (hasActualDistricts && scope.districts.length === 0)}
                    />
                </div>
            )}

            {isLoadingDistrictsSchools && <p className="text-xs text-gray-600">Fetching data from {assessmentId} table...</p>}
        </div>
    )
}
