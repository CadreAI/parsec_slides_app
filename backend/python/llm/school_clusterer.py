"""
School clustering module for grouping school name variants using LLM
Uses GPT-3.5-turbo for fast, cost-effective clustering of school names
"""

import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Load environment variables
load_dotenv()


def _validate_clustering_result(result_data: Dict, input_schools: List[str]) -> None:
    """
    Validate that clustering result matches input requirements:
    1. All output schools must exist in input list
    2. Total count of schools in clusters must equal input length

    Args:
        result_data: The clustering result dictionary
        input_schools: Original list of school names

    Raises:
        ValueError: If validation fails
    """
    clusters = result_data.get("clusters", {})
    input_set = set(input_schools)

    # Collect all schools from clusters
    all_output_schools = set()
    for school_list in clusters.values():
        all_output_schools.update(school_list)

    # Validation 1: Check that all output schools exist in input
    invalid_schools = all_output_schools - input_set
    if invalid_schools:
        raise ValueError(f"Output contains schools not in input: {invalid_schools}")

    # Validation 2: Check that total count matches input length
    total_output_count = len(all_output_schools)
    input_count = len(input_schools)
    if total_output_count != input_count:
        raise ValueError(
            f"Output count ({total_output_count}) does not match input count ({input_count}). "
            f"Missing schools: {input_set - all_output_schools}"
        )


def _perform_clustering_llm_call(
    client: OpenAI, schools: List[str], district_name: Optional[str] = None
) -> Dict:
    """
    Perform a single LLM clustering call and process the result

    Args:
        client: OpenAI client instance
        schools: List of school names to cluster
        district_name: Optional district name for context

    Returns:
        Dict with clusters and school_to_cluster mapping (without source field)
    """
    # Build clustering prompt
    district_context = f" in {district_name} district" if district_name else ""
    schools_list = "\n".join([f"- {school}" for school in schools])

    clustering_prompt = f"""You are a school name clustering assistant. Your job is to group school names ONLY when they represent the EXACT SAME physical school with different grade level suffixes.

Schools{district_context}:
{schools_list}

ONLY cluster when ALL of these conditions are met:
1. The school names are IDENTICAL except for grade level indicators at the END
2. Grade level indicators are: "MS", "HS", "Middle School", "High School", "Elementary", "Elementary School", "K-8", "TK-8", "6-8", "9-12"
3. Everything before the grade level indicator must be EXACTLY the same

DO NOT cluster (keep as separate schools):
- Schools with ANY city/location names (Chico, Paradise, etc.)
- Schools with "Charter" vs "Charter School" (different names)
- Schools with ANY descriptive words after the base name (even if they look similar)
- Schools with directional indicators (North, South, East, West)
- Schools with "(Historic Data)" suffix
- When in doubt, DO NOT cluster

Example 1 (SHOULD cluster - only grade levels differ):
Input: ["Lincoln Elementary School", "Lincoln Middle School", "Lincoln High School"]
Output:
{{
    "clusters": {{
        "Lincoln": ["Lincoln Elementary School", "Lincoln Middle School", "Lincoln High School"]
    }}
}}

Example 2 (SHOULD NOT cluster - different names beyond grade levels):
Input: ["Achieve Charter Chico", "Achieve Charter Paradise", "Achieve Charter High School"]
Output:
{{
    "clusters": {{
        "Achieve Charter Chico": ["Achieve Charter Chico"],
        "Achieve Charter Paradise": ["Achieve Charter Paradise"],
        "Achieve Charter High School": ["Achieve Charter High School"]
    }}
}}

Example 3 (SHOULD NOT cluster - "Charter" vs "Charter School"):
Input: ["Achieve Charter", "Achieve Charter School"]
Output:
{{
    "clusters": {{
        "Achieve Charter": ["Achieve Charter"],
        "Achieve Charter School": ["Achieve Charter School"]
    }}
}}

Be VERY conservative. If there's ANY difference beyond simple grade level suffixes, keep schools separate.

Respond with a JSON object:
{{
    "clusters": {{
        "Cluster Name": ["School 1", "School 2"],
        ...
    }}
}}"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a school name clustering assistant. Always respond with valid JSON only.",
            },
            {"role": "user", "content": clustering_prompt},
        ],
        temperature=0.2,  # Low temperature for consistent clustering
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    clusters = result.get("clusters", {})

    # Validate and build school_to_cluster mapping
    school_to_cluster = {}
    validated_clusters = {}
    all_clustered_schools = set()

    for cluster_name, school_list in clusters.items():
        if not isinstance(school_list, list):
            continue

        # Validate all schools in cluster exist in original list
        valid_schools = [s for s in school_list if s in schools]
        if valid_schools:
            validated_clusters[cluster_name] = valid_schools
            for school in valid_schools:
                school_to_cluster[school] = cluster_name
                all_clustered_schools.add(school)

    # Add any missing schools as their own clusters (identity mapping)
    for school in schools:
        if school not in all_clustered_schools:
            validated_clusters[school] = [school]
            school_to_cluster[school] = school

    return {
        "clusters": validated_clusters,
        "school_to_cluster": school_to_cluster,
    }


def cluster_schools(schools: List[str], district_name: Optional[str] = None) -> Dict:
    """
    Return schools as individual clusters (no combining/clustering)
    Just formats them nicely for output

    Args:
        schools: List of school names to format
        district_name: Optional district name for context (informational only)

    Returns:
        Dict with:
            - clusters: Dict[str, List[str]] - each school is its own cluster
            - school_to_cluster: Dict[str, str] - identity mapping
            - source: str - 'identity'

    Example:
        Input: ["Classical Academy MS", "Classical Academy HS", "Other School"]
        Output: {
            "clusters": {
                "Classical Academy MS": ["Classical Academy MS"],
                "Classical Academy HS": ["Classical Academy HS"],
                "Other School": ["Other School"]
            },
            "school_to_cluster": {
                "Classical Academy MS": "Classical Academy MS",
                "Classical Academy HS": "Classical Academy HS",
                "Other School": "Other School"
            },
            "source": "identity"
        }
    """
    # Handle empty case
    if not schools or len(schools) == 0:
        return {"clusters": {}, "school_to_cluster": {}, "source": "identity"}

    # Create identity mapping (each school is its own cluster)
    result = _create_identity_mapping(schools, "identity")

    # Output all schools in a nice format
    print(f"\n[School Clusterer] === Schools List ===")
    if district_name:
        print(f"  District: {district_name}")
    print(f"  Total Schools: {len(schools)}\n")
    
    for school_name in sorted(schools):
        print(f"  • {school_name}")
    
    print(f"\n[School Clusterer] ========================\n")
    
    return result


def _create_identity_mapping(schools: List[str], source: str = "identity") -> Dict:
    """Create identity mapping where each school is its own cluster"""
    clusters = {school: [school] for school in schools}
    school_to_cluster = {school: school for school in schools}
    return {
        "clusters": clusters,
        "school_to_cluster": school_to_cluster,
        "source": source,
    }
