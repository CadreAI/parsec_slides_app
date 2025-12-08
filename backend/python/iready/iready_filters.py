"""
iReady chart filtering and validation utilities
"""

import pandas as pd


def apply_chart_filters(df, filters):
    """Apply chart generation filters to dataframe"""
    if not filters:
        return df
    
    d = df.copy()
    
    # Filter by grades
    if filters.get("grades") is not None and len(filters["grades"]) > 0:
        # Convert grade column to numeric, handling "K" as 0
        def normalize_grade(grade_val):
            if pd.isna(grade_val):
                return None
            grade_str = str(grade_val).strip().upper()
            if grade_str == "K" or grade_str == "KINDERGARTEN":
                return 0
            try:
                return int(float(grade_str))
            except:
                return None
        
        # iReady uses student_grade column
        grade_col = "student_grade" if "student_grade" in d.columns else "grade"
        if grade_col in d.columns:
            d["grade_normalized"] = d[grade_col].apply(normalize_grade)
            d = d[d["grade_normalized"].isin(filters["grades"])].copy()
            d = d.drop(columns=["grade_normalized"], errors="ignore")
    
    # Filter by years (iReady uses academicyear)
    if filters.get("years") is not None and len(filters["years"]) > 0:
        year_col = "academicyear" if "academicyear" in d.columns else "year"
        if year_col in d.columns:
            d[year_col] = pd.to_numeric(d[year_col].astype(str).str[:4], errors="coerce")
            d = d[d[year_col].isin(filters["years"])].copy()
    
    # Note: Quarters filtering is NOT done here because we need to iterate over quarters
    # during chart generation. The quarters filter is applied per-chart via window_filter parameter.
    
    # Filter by race/ethnicity (if provided as separate filter)
    if filters.get("race") is not None and len(filters["race"]) > 0:
        race_col = None
        for col in ["ethnicityrace", "race", "ethnicity"]:
            if col in d.columns:
                race_col = col
                break
        
        if race_col:
            race_values = []
            for race_filter in filters["race"]:
                # Map UI race values to possible database values
                race_map = {
                    "Hispanic or Latino": ["Hispanic", "Hispanic or Latino"],
                    "White": ["White"],
                    "Black or African American": ["Black", "African American", "Black or African American"],
                    "Asian": ["Asian"],
                    "Filipino": ["Filipino"],
                    "American Indian or Alaska Native": ["American Indian", "Alaska Native", "American Indian or Alaska Native"],
                    "Native Hawaiian or Pacific Islander": ["Pacific Islander", "Native Hawaiian", "Native Hawaiian or Other Pacific Islander"],
                    "Two or More Races": ["Two or More Races", "Multiracial", "Multiple Races"],
                    "Not Stated": ["Not Stated", "Unknown", ""]
                }
                race_values.extend(race_map.get(race_filter, [race_filter]))
            d = d[d[race_col].astype(str).str.strip().isin([str(v).strip() for v in race_values])].copy()
    
    return d


def should_generate_subject(subject, filters):
    """Check if a subject should be generated based on filters"""
    if not filters or filters.get("subjects") is None or len(filters.get("subjects", [])) == 0:
        return True  # Generate all if no filter
    
    # iReady uses "ELA" and "Math" instead of "Reading" and "Mathematics"
    subject_map = {"Math": ["Math", "Mathematics"], "ELA": ["ELA", "Reading"]}
    # Normalize filter values and subject name
    filter_subjects = [str(s).strip() for s in filters["subjects"]]
    subject_normalized = str(subject).strip()
    
    # Check if subject matches directly or through mapping
    if subject_normalized in filter_subjects:
        return True
    if subject_normalized in subject_map:
        for mapped_name in subject_map[subject_normalized]:
            if mapped_name in filter_subjects:
                return True
    return False


def should_generate_student_group(group_name, filters):
    """Check if a student group chart should be generated"""
    if not filters:
        return True  # Generate all if no filter
    
    # Check if student groups filter is specified
    student_groups_filter = filters.get("student_groups")
    race_filter = filters.get("race")
    
    # If both filters are empty/None, generate all
    if (student_groups_filter is None or len(student_groups_filter) == 0) and \
       (race_filter is None or len(race_filter) == 0):
        return True
    
    # Check if this group is in student_groups filter
    if student_groups_filter and group_name in student_groups_filter:
        return True
    
    # Check if this group is in race filter (race groups are also in student_groups config)
    if race_filter and group_name in race_filter:
        return True
    
    # If filters are specified but this group is not in either, don't generate
    return False


def should_generate_grade(grade, filters):
    """Check if a grade-specific chart should be generated"""
    if not filters or filters.get("grades") is None or len(filters.get("grades", [])) == 0:
        return True  # Generate all if no filter
    
    # Normalize grade value (handle "K" as 0)
    try:
        if pd.isna(grade):
            return False
        grade_str = str(grade).strip().upper()
        if grade_str == "K" or grade_str == "KINDERGARTEN":
            grade_val = 0
        else:
            grade_val = int(float(grade_str))
        return grade_val in filters["grades"]
    except:
        return False

