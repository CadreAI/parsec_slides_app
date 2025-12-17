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

    clustering_prompt = f"""You are a school name clustering assistant. Your job is to group school names that represent the same physical school but have different suffixes or variants.

Schools{district_context}:
{schools_list}

Common patterns to cluster:
1. Same base name with different grade levels (e.g., "Classical Academy MS", "Classical Academy HS", "Classical Academy Elementary")
2. Same base name with different campuses (e.g., "Lincoln School North", "Lincoln School South")
3. Same base name with different programs (e.g., "Washington Charter TK-8", "Washington Charter High School")

Rules:
- Group schools ONLY if they clearly represent the same school with different variants
- Keep the base/common name as the cluster name (remove grade level suffixes)
- Do NOT cluster schools that are actually different schools
- If a school has no variants, it becomes its own cluster
- Preserve original school names exactly in the arrays

Respond with a JSON object:
{{
    "clusters": {{
        "Base School Name 1": ["Original School Name 1", "Original School Name 2"],
        "Base School Name 2": ["Original School Name 3"],
        ...
    }}
}}

Example:
Input schools: ["Classical Academy Middle School", "Classical Academy High School", "Other School"]
Output:
{{
    "clusters": {{
        "Classical Academy": ["Classical Academy Middle School", "Classical Academy High School"],
        "Other School": ["Other School"]
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
    Cluster school names that represent the same school into groups

    Args:
        schools: List of school names to cluster
        district_name: Optional district name for context

    Returns:
        Dict with:
            - clusters: Dict[str, List[str]] - cluster name mapped to original school names
            - school_to_cluster: Dict[str, str] - original school name mapped to cluster name
            - source: str - 'llm' or 'identity' (fallback)

    Example:
        Input: ["Classical Academy MS", "Classical Academy HS", "Other School"]
        Output: {
            "clusters": {
                "Classical Academy": ["Classical Academy MS", "Classical Academy HS"],
                "Other School": ["Other School"]
            },
            "school_to_cluster": {
                "Classical Academy MS": "Classical Academy",
                "Classical Academy HS": "Classical Academy",
                "Other School": "Other School"
            },
            "source": "llm"
        }
    """
    # Handle empty or single school case
    if not schools or len(schools) == 0:
        return {"clusters": {}, "school_to_cluster": {}, "source": "identity"}

    if len(schools) == 1:
        school = schools[0]
        return {
            "clusters": {school: [school]},
            "school_to_cluster": {school: school},
            "source": "identity",
        }

    # If no OpenAI client available, return identity mapping
    if OpenAI is None:
        print("[School Clusterer] OpenAI not available, using identity mapping")
        return _create_identity_mapping(schools, "identity")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[School Clusterer] OpenAI API key not found, using identity mapping")
        return _create_identity_mapping(schools, "identity")

    client = OpenAI(api_key=api_key)

    # Try clustering with retry on validation failure
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            result_data = _perform_clustering_llm_call(client, schools, district_name)
            result_data["source"] = "llm"

            # Validate the result
            try:
                _validate_clustering_result(result_data, schools)

                # Validation passed
                cluster_count = len(result_data["clusters"])
                reduction = len(schools) - cluster_count
                attempt_msg = f" (attempt {attempt + 1})" if attempt > 0 else ""
                print(
                    f"[School Clusterer] Clustered {len(schools)} schools into {cluster_count} groups (reduced by {reduction}){attempt_msg}"
                )
                return result_data

            except ValueError as validation_error:
                if attempt < max_attempts - 1:
                    print(
                        f"[School Clusterer] Validation failed on attempt {attempt + 1}: {validation_error}, retrying..."
                    )
                    continue
                else:
                    print(
                        f"[School Clusterer] Validation failed after {max_attempts} attempts: {validation_error}, using identity mapping"
                    )
                    return _create_identity_mapping(schools, "identity")

        except Exception as e:
            if attempt < max_attempts - 1:
                print(
                    f"[School Clusterer] Error on attempt {attempt + 1}: {e}, retrying..."
                )
                continue
            else:
                print(
                    f"[School Clusterer] Error clustering schools after {max_attempts} attempts: {e}, using identity mapping"
                )
                return _create_identity_mapping(schools, "identity")


def _create_identity_mapping(schools: List[str], source: str = "identity") -> Dict:
    """Create identity mapping where each school is its own cluster"""
    clusters = {school: [school] for school in schools}
    school_to_cluster = {school: school for school in schools}
    return {
        "clusters": clusters,
        "school_to_cluster": school_to_cluster,
        "source": source,
    }
