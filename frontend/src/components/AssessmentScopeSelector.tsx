'use client'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { MultiSelect } from '@/components/ui/multi-select'
import { useDistrictsAndSchools } from '@/hooks/useDistrictsAndSchools'
import { getClusteredSchoolOptions, getDistrictOptions } from '@/utils/formHelpers'
import { useEffect } from 'react'
import { toast } from 'sonner'

interface AssessmentScopeSelectorProps {
    assessmentId: string
    partnerName: string
    projectId: string
    location: string
    customDataSource?: string
    scope: {
        districts: string[]
        schools: string[]
        resolvedSchools?: string[]
        includeDistrictwide?: boolean
        includeSchools?: boolean
    }
    onScopeChangeAction: (
        assessmentId: string,
        scope: {
            districts: string[]
            schools: string[]
            resolvedSchools?: string[]
            includeDistrictwide?: boolean
            includeSchools?: boolean
        }
    ) => void
    partnerConfig: Record<string, { districts: string[]; schools: Record<string, string[]> }>
}

export function AssessmentScopeSelector({
    assessmentId,
    partnerName,
    projectId,
    location,
    customDataSource,
    scope,
    onScopeChangeAction,
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
    const scopeLabel = hasActualDistricts ? 'District' : 'Charter'
    const aggregateLabel = hasActualDistricts ? 'Districtwide' : 'Charterwide'

    // Get district options
    const districtOptions = getDistrictOptions(availableDistricts, partnerName, partnerConfig)

    // UX: if there is exactly one district available, auto-select it.
    // This prevents confusing states like: districtSchoolMap has keys but the selected districts array is empty.
    useEffect(() => {
        if (!hasActualDistricts) return
        if (isLoadingDistrictsSchools) return
        if (!partnerName) return
        if (Array.isArray(scope.districts) && scope.districts.length > 0) return
        if (districtOptions.length !== 1) return

        const onlyDistrict = districtOptions[0]
        onScopeChangeAction(assessmentId, {
            ...scope,
            districts: [onlyDistrict],
            schools: [],
            resolvedSchools: []
        })
        // Intentionally omit `scope` object from deps to avoid re-triggering due to identity changes.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
        assessmentId,
        districtOptions.join(','),
        hasActualDistricts,
        isLoadingDistrictsSchools,
        partnerName,
        (scope.districts || []).join(','),
        onScopeChangeAction
    ])

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

    const includeDistrictwide = scope.includeDistrictwide !== false // default true
    const includeSchools = scope.includeSchools === true // default false

    // If clustering is active, user selects cluster labels. Expand them to raw school names for backend use.
    const resolveSelectedSchools = (selected: string[]): string[] => {
        const hasClusters = clusteredSchools.length > 0 && Object.keys(schoolClusters || {}).length > 0
        if (!hasClusters) return selected
        const out: string[] = []
        for (const item of selected) {
            const expanded = schoolClusters?.[item]
            if (Array.isArray(expanded) && expanded.length > 0) {
                out.push(...expanded)
            } else {
                out.push(item)
            }
        }
        return Array.from(new Set(out)).sort()
    }

    return (
        <div className="ml-6 mt-3 space-y-3 rounded-lg border-l-2 border-blue-200 bg-blue-50/30 p-3">
            <div className="text-xs font-semibold uppercase tracking-wide text-gray-600">📍 Scope for {assessmentId}</div>

            {/* Quick action */}
            <div className="flex flex-wrap items-center gap-2">
                <Button
                    type="button"
                    variant={includeDistrictwide && !includeSchools ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => {
                        onScopeChangeAction(assessmentId, {
                            ...scope,
                            includeDistrictwide: true,
                            includeSchools: false,
                            schools: [],
                            resolvedSchools: []
                        })
                    }}
                    disabled={isLoadingDistrictsSchools}
                >
                    {scopeLabel} only
                </Button>
                <p className="text-xs text-gray-600">Skips all school charts for this assessment</p>
            </div>

            {/* District Selection - Only show if we have actual districts */}
            {hasActualDistricts && (
                <div className="space-y-2">
                    <Label className="text-xs">District / Charter</Label>
                    <MultiSelect
                        options={districtOptions}
                        selected={scope.districts}
                        onChange={(selected) => {
                            onScopeChangeAction(assessmentId, {
                                ...scope,
                                districts: selected,
                                schools: [], // Reset schools when districts change
                                resolvedSchools: []
                            })
                        }}
                        placeholder={isLoadingDistrictsSchools ? 'Loading districts...' : 'Select district(s)...'}
                        disabled={isLoadingDistrictsSchools || !partnerName}
                    />
                </div>
            )}

            {/* Districtwide aggregate toggle (works even when dataset has no districts) */}
            <div className="flex items-start space-x-2">
                <Checkbox
                    id={`${assessmentId}-includeDistrictwide`}
                    checked={includeDistrictwide}
                    onChange={(e) => {
                        onScopeChangeAction(assessmentId, {
                            ...scope,
                            includeDistrictwide: e.target.checked
                        })
                    }}
                    disabled={isLoadingDistrictsSchools}
                />
                <div className="flex-1">
                    <Label htmlFor={`${assessmentId}-includeDistrictwide`} className="cursor-pointer text-xs font-medium">
                        Include {aggregateLabel} Aggregate
                    </Label>
                    <p className="text-xs text-gray-600">Generate {aggregateLabel.toLowerCase()} aggregated charts alongside any school charts</p>
                </div>
            </div>

            {/* School charts toggle */}
            <div className="flex items-start space-x-2">
                <Checkbox
                    id={`${assessmentId}-includeSchools`}
                    checked={includeSchools}
                    onChange={(e) => {
                        onScopeChangeAction(assessmentId, {
                            ...scope,
                            includeSchools: e.target.checked,
                            schools: e.target.checked ? scope.schools : [],
                            resolvedSchools: e.target.checked ? resolveSelectedSchools(scope.schools) : []
                        })
                    }}
                    disabled={isLoadingDistrictsSchools || (hasActualDistricts && scope.districts.length === 0)}
                />
                <div className="flex-1">
                    <Label htmlFor={`${assessmentId}-includeSchools`} className="cursor-pointer text-xs font-medium">
                        Include School Charts
                    </Label>
                    <p className="text-xs text-gray-600">Generate charts for selected schools</p>
                </div>
            </div>

            {/* School Selection */}
            {includeSchools && ((hasActualDistricts && scope.districts.length > 0) || !hasActualDistricts) && (
                <div className="space-y-2">
                    <Label className="text-xs">School(s)</Label>
                    <MultiSelect
                        options={finalSchoolOptions}
                        selected={scope.schools}
                        onChange={(selected) => {
                            const limited = selected.slice(0, 5)
                            onScopeChangeAction(assessmentId, {
                                ...scope,
                                schools: limited,
                                resolvedSchools: resolveSelectedSchools(limited)
                            })
                        }}
                        maxSelected={5}
                        onMaxSelected={() => toast.warning('You can select up to 5 schools per assessment.')}
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
