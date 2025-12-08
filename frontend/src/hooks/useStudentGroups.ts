import { useState } from 'react'

const DEFAULT_STUDENT_GROUPS = ['All Students', 'English Learners', 'Students with Disabilities', 'Socioeconomically Disadvantaged', 'Foster', 'Homeless']

const DEFAULT_RACE_GROUPS = [
    'Hispanic or Latino',
    'White',
    'Black or African American',
    'Asian',
    'Filipino',
    'American Indian or Alaska Native',
    'Native Hawaiian or Pacific Islander',
    'Two or More Races',
    'Not Stated'
]

const DEFAULT_MAPPINGS: Record<string, { type?: string; column?: string; in?: (string | number)[] }> = {
    'All Students': { type: 'all' },
    'English Learners': { column: 'englishlearner', in: ['Y', 'Yes', 'True', 1] },
    'Students with Disabilities': { column: 'studentswithdisabilities', in: ['Y', 'Yes', 'True', 1] },
    'Socioeconomically Disadvantaged': { column: 'socioeconomicallydisadvantaged', in: ['Y', 'Yes', 'True', 1] },
    'Hispanic or Latino': { column: 'ethnicityrace', in: ['Hispanic', 'Hispanic or Latino'] },
    White: { column: 'ethnicityrace', in: ['White'] },
    'Black or African American': { column: 'ethnicityrace', in: ['Black', 'African American', 'Black or African American'] },
    Asian: { column: 'ethnicityrace', in: ['Asian'] },
    Filipino: { column: 'ethnicityrace', in: ['Filipino'] },
    'American Indian or Alaska Native': { column: 'ethnicityrace', in: ['American Indian', 'Alaska Native', 'American Indian or Alaska Native'] },
    'Native Hawaiian or Pacific Islander': {
        column: 'ethnicityrace',
        in: ['Pacific Islander', 'Native Hawaiian', 'Native Hawaiian or Other Pacific Islander']
    },
    'Two or More Races': { column: 'ethnicityrace', in: ['Two or More Races', 'Multiracial', 'Multiple Races'] },
    'Not Stated': { column: 'ethnicityrace', in: ['Not Stated', 'Unknown', ''] },
    Foster: { column: 'foster', in: ['Y', 'Yes', 'True', 1] },
    Homeless: { column: 'homeless', in: ['Y', 'Yes', 'True', 1] }
}

const DEFAULT_ORDER: Record<string, number> = {
    'All Students': 1,
    'English Learners': 2,
    'Students with Disabilities': 3,
    'Socioeconomically Disadvantaged': 4,
    'Hispanic or Latino': 5,
    White: 6,
    'Black or African American': 7,
    Asian: 8,
    Filipino: 9,
    'American Indian or Alaska Native': 10,
    'Native Hawaiian or Pacific Islander': 11,
    'Two or More Races': 12,
    'Not Stated': 13,
    Foster: 14,
    Homeless: 15
}

export function useStudentGroups() {
    const [studentGroupOptions] = useState<string[]>(DEFAULT_STUDENT_GROUPS)
    const [raceOptions] = useState<string[]>(DEFAULT_RACE_GROUPS)
    const [studentGroupMappings] = useState<Record<string, { type?: string; column?: string; in?: (string | number)[] }>>(DEFAULT_MAPPINGS)
    const [studentGroupOrder] = useState<Record<string, number>>(DEFAULT_ORDER)

    return {
        studentGroupOptions,
        raceOptions,
        studentGroupMappings,
        studentGroupOrder
    }
}
