"""
Decision LLM module for determining whether to use AI insights and chart selection/ordering
Uses lightweight GPT-3.5-turbo for fast, cost-effective decisions
"""
import os
import re
from typing import Dict, Optional, List
from pathlib import Path
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Load environment variables
load_dotenv()

# Import layout learner
try:
    from .layout_learner import get_layout_context
except ImportError:
    # Fallback if layout_learner not available
    def get_layout_context(reference_deck_dir: Optional[str] = None) -> str:
        return ""


def should_use_ai_insights(
    user_prompt: Optional[str] = None,
    chart_count: int = 0,
    default_enable: bool = True
) -> Dict[str, any]:
    """
    Determine if AI insights should be used based on user prompt and context
    
    Args:
        user_prompt: Optional user-provided prompt describing their preferences
        chart_count: Number of charts to analyze
        default_enable: Default value if no prompt provided
    
    Returns:
        Dict with:
            - use_ai: bool (whether to use AI insights)
            - reasoning: str (explanation of decision)
            - confidence: float (0.0-1.0 confidence in decision)
            - analysis_focus: Optional[str] (specific focus areas if use_ai=True)
    """
    # If no OpenAI client available, return default
    if OpenAI is None:
        return {
            'use_ai': default_enable,
            'reasoning': 'OpenAI not available, using default',
            'confidence': 0.5,
            'analysis_focus': None
        }
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return {
            'use_ai': default_enable,
            'reasoning': 'OpenAI API key not found, using default',
            'confidence': 0.5,
            'analysis_focus': None
        }
    
    client = OpenAI(api_key=api_key)
    
    # If no user prompt, use default
    if not user_prompt or not user_prompt.strip():
        return {
            'use_ai': default_enable,
            'reasoning': 'No user prompt provided, using default setting',
            'confidence': 0.7,
            'analysis_focus': None
        }
    
    # Build decision prompt
    decision_prompt = f"""You are a decision assistant for a slide presentation system. Your job is to determine if AI-generated insights should be used for chart analysis.

Context:
- Number of charts: {chart_count}
- User request: "{user_prompt}"

Considerations:
1. If user explicitly requests AI insights, analysis, or detailed explanations → USE AI
2. If user wants "quick", "simple", "basic", or "no analysis" → DON'T USE AI
3. If user mentions specific focus areas (trends, comparisons, recommendations) → USE AI
4. If user wants "standard" or "default" slides → MAYBE (consider chart count)
5. For many charts (>10), AI might be too expensive/time-consuming → CONSIDER COST
6. For few charts (<5), AI is usually worth it → USE AI

Respond with a JSON object:
{{
    "use_ai": true/false,
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0,
    "analysis_focus": "optional focus areas like 'trends', 'comparisons', 'actionable insights', or null"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a decision assistant. Always respond with valid JSON only."},
                {"role": "user", "content": decision_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent decisions
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        import json
        decision = json.loads(response.choices[0].message.content)
        
        # Validate response structure
        result = {
            'use_ai': decision.get('use_ai', default_enable),
            'reasoning': decision.get('reasoning', 'Decision made by LLM'),
            'confidence': float(decision.get('confidence', 0.7)),
            'analysis_focus': decision.get('analysis_focus')
        }
        
        # Ensure confidence is in valid range
        result['confidence'] = max(0.0, min(1.0, result['confidence']))
        
        print(f"[Decision LLM] Decision: use_ai={result['use_ai']}, confidence={result['confidence']:.2f}, reasoning={result['reasoning']}")
        
        return result
        
    except Exception as e:
        print(f"[Decision LLM] Error making decision: {e}, falling back to default")
        return {
            'use_ai': default_enable,
            'reasoning': f'Error in decision LLM: {str(e)}, using default',
            'confidence': 0.5,
            'analysis_focus': None
        }


def parse_chart_instructions(
    user_prompt: Optional[str] = None,
    chart_paths: Optional[List[str]] = None,
    deck_type: Optional[str] = None
) -> Dict[str, any]:
    """
    Parse user prompt to extract chart selection and ordering instructions
    
    Args:
        user_prompt: User-provided prompt describing which charts to include and order
        chart_paths: List of available chart file paths
    
    Returns:
        Dict with:
            - chart_selection: List[str] (filtered chart paths in desired order)
            - instructions: Dict with selection criteria (grades, subjects, sections, etc.)
    """
    if not user_prompt or not user_prompt.strip() or not chart_paths:
        return {
            'chart_selection': chart_paths or [],
            'instructions': None
        }
    
    # Check if user explicitly wants ALL charts
    user_prompt_lower = user_prompt.lower().strip()
    all_keywords = ['all graphs', 'all charts', 'all of them', 'everything', 'include all', 'show all', 'output all']
    if any(keyword in user_prompt_lower for keyword in all_keywords):
        print(f"[Chart Selection] User requested all charts - skipping filtering")
        return {
            'chart_selection': chart_paths,
            'instructions': None,
            'reasoning': 'User requested all charts/graphs'
        }
    
    if OpenAI is None:
        return {
            'chart_selection': chart_paths,
            'instructions': None
        }
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return {
            'chart_selection': chart_paths,
            'instructions': None
        }
    
    client = OpenAI(api_key=api_key)
    
    # Extract chart names for context
    chart_names = [Path(p).name for p in chart_paths]
    
    # Get layout context from reference decks (filtered by deck_type)
    layout_context = get_layout_context(deck_type=deck_type)
    
    # Extract max slides limit from user prompt
    import re
    max_slides = None
    max_slides_patterns = [
        r'limit.*?(\d+).*?slide',
        r'max.*?(\d+).*?slide',
        r'(\d+).*?slide',
        r'up to (\d+)',
        r'maximum of (\d+)',
        r'no more than (\d+)'
    ]
    for pattern in max_slides_patterns:
        match = re.search(pattern, user_prompt.lower())
        if match:
            max_slides = int(match.group(1))
            print(f"[Chart Selection] Detected max slides limit: {max_slides}")
            break
    
    # Build prompt for chart selection/ordering
    max_slides_instruction = f"\n**IMPORTANT**: User requested a maximum of {max_slides} slides. Prioritize the MOST SPECIFIC and RELEVANT charts that match their criteria. If more than {max_slides} charts match, select the {max_slides} most important/relevant ones." if max_slides else ""
    
    selection_prompt = f"""You are a chart selection assistant. Parse the user's instructions to determine which charts to include and in what order.

User request: "{user_prompt}"

Available charts (by filename):
{chr(10).join(f"- {name}" for name in chart_names)}

{layout_context if layout_context else ""}
{max_slides_instruction}

**CHART SELECTION GUIDELINES (CRITICAL):**
1. If the user says "all graphs", "all charts", "all of them", "everything", "include all", "show all", or "output all" → return ALL charts in chart_selection list (unless a max slides limit is specified).

2. Otherwise, use reference deck patterns to INTELLIGENTLY SELECT charts:
   - **REQUIRED CHARTS**: Charts listed as "required charts" appear in 80%+ of reference decks - prioritize including these if they match user's request
   - **OPTIONAL CHARTS**: Charts listed as "optional charts" appear inconsistently - only include if explicitly relevant to user's request
   - **OMIT UNNECESSARY CHARTS**: If a chart type doesn't appear in reference decks AND isn't explicitly requested by the user, OMIT IT
   - **FOLLOW GROUPINGS**: If charts typically appear together in reference decks, include them together when relevant
   - **RESPECT FLOW**: Follow the presentation flow patterns from reference decks when ordering

3. **FILTERING LOGIC**:
   - Start with charts that match user's explicit request
   - Then add required charts from reference decks that are relevant
   - Only add optional charts if they enhance the presentation
   - EXCLUDE charts that don't match reference deck patterns unless explicitly requested

4. **EXAMPLE**: If reference decks show that "section3" charts are common but "section5" charts never appear, and user requests "grade 1-4 trends", include section3 charts but don't include section5 charts unless user explicitly asks for them.

Instructions:
1. If user wants ALL charts → list ALL chart filenames in chart_selection and set exclude_others=false
2. Otherwise, identify which charts the user wants (by keywords like: grade, subject, section, trend, etc.)
3. Determine the order they want them in
4. If user says "that's it" or "only", exclude charts not explicitly mentioned
5. IMPORTANT: If user mentions "grades 1-4" or "grade 1-4", include ALL grades in that range (1, 2, 3, 4)
6. IMPORTANT: If user mentions demographic groups (Hispanic, Latino, Black, African American, White, etc.) or student groups → include section2 charts
7. Common patterns:
   - "all graphs" or "all charts" → return ALL charts
   - "grade 1 math and reading" → section3 charts for grade 1, math and reading
   - "grades 1-4" → ALL section3 charts for grades 1, 2, 3, and 4 (both math and reading)
   - "fall year to year trend" → section1 fall trends charts
   - "Hispanic", "Latino", "Black", "White", "demographic", "student group" → section2 charts
   - "first show X, then show Y" → order matters
   - "only X" → exclude everything else

Respond with JSON:
{{
    "chart_selection": ["list of chart filenames in desired order - ONLY include charts that match user's request AND reference deck patterns.{" If user specified a max slides limit, prioritize the MOST SPECIFIC charts up to that limit." if max_slides else " If user wants ALL charts, list ALL filenames here"}],
    "instructions": {{
        "grades": ["list of ALL grade numbers mentioned (e.g., ['1', '2', '3', '4'] for 'grades 1-4'), empty array if all grades"],
        "subjects": ["math", "reading", or both, empty array if all subjects],
        "sections": ["section1", "section2", "section3", etc., empty array if all sections],
        "order_matters": true/false,
        "exclude_others": false if user wants all charts, true only if user says "only" or "that's it",
        "filter_by_reference_patterns": true{"" + (f',\n        "max_slides": {max_slides}' if max_slides else "")}
    }},
    "reasoning": "brief explanation of selection, including which charts were included/excluded based on reference deck patterns{" and how the max slides limit was applied" if max_slides else ""}"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a chart selection assistant. Always respond with valid JSON only."},
                {"role": "user", "content": selection_prompt}
            ],
            temperature=0.2,  # Lower temperature for more consistent parsing
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        selected_filenames = result.get('chart_selection', [])
        instructions = result.get('instructions', {})
        reasoning = result.get('reasoning', 'No reasoning provided')
        
        # Map filenames back to full paths
        filename_to_path = {Path(p).name: p for p in chart_paths}
        selected_paths = []
        seen_paths = set()
        
        # First, try to match by filenames from LLM
        for filename in selected_filenames:
            # Try exact match first
            if filename in filename_to_path:
                path = filename_to_path[filename]
                if path not in seen_paths:
                    selected_paths.append(path)
                    seen_paths.add(path)
            else:
                # Try partial match (chart name keywords in filename)
                filename_lower = filename.lower()
                for path in chart_paths:
                    path_name_lower = Path(path).name.lower()
                    # Check if all key words from filename are in path name
                    keywords = [kw.strip() for kw in filename_lower.replace('_', ' ').replace('-', ' ').split() if len(kw) > 2]
                    if keywords and all(kw in path_name_lower for kw in keywords):
                        if path not in seen_paths:
                            selected_paths.append(path)
                            seen_paths.add(path)
                            break
        
        # Get exclude_others flag early
        exclude_others = instructions.get('exclude_others', False) if instructions else False
        
        # Apply reference deck filtering if enabled
        filter_by_reference = instructions.get('filter_by_reference_patterns', True) if instructions else True
        
        # If filtering is enabled, check against reference deck patterns
        # Only filter if user didn't explicitly request all charts
        if filter_by_reference and not exclude_others and len(selected_paths) > 0:
            try:
                from .layout_learner import learn_layout_patterns
                script_dir = Path(__file__).parent
                reference_deck_dir = script_dir / 'reference_decks'
                patterns = learn_layout_patterns(str(reference_deck_dir))
                selection_patterns = patterns.get('chart_selection_patterns', {})
                required_charts = selection_patterns.get('required_charts', [])
                optional_charts = selection_patterns.get('optional_charts', [])
                
                # Build set of chart types that match reference patterns
                reference_chart_types = set(required_charts + optional_charts)
                
                # Filter selected paths to only include those matching reference patterns
                # unless user explicitly requested them
                if reference_chart_types:
                    filtered_paths = []
                    for path in selected_paths:
                        path_name_lower = Path(path).name.lower()
                        # Check if chart matches any reference pattern
                        matches_reference = False
                        for ref_type in reference_chart_types:
                            # Check if chart filename contains elements of reference type
                            ref_parts = ref_type.lower().split('_')
                            if all(part in path_name_lower for part in ref_parts if len(part) > 3):
                                matches_reference = True
                                break
                        
                        # Include if matches reference OR if user explicitly requested (check if in user prompt keywords)
                        if matches_reference or any(keyword in user_prompt_lower for keyword in Path(path).stem.lower().split('_')):
                            filtered_paths.append(path)
                    
                    if filtered_paths:
                        print(f"[Chart Selection] Filtered by reference patterns: {len(selected_paths)} → {len(filtered_paths)} charts")
                        selected_paths = filtered_paths
            except Exception as e:
                print(f"[Chart Selection] Error applying reference filtering: {e}")
        
        # Fallback: If instructions specify grades/subjects/sections, match charts by criteria
        # This handles cases where LLM didn't list all filenames but gave criteria
        # BUT: If user wanted all charts, skip criteria filtering
        if instructions and len(selected_paths) < len(chart_paths) and not exclude_others:
            grades = instructions.get('grades', [])
            subjects = instructions.get('subjects', [])
            sections = instructions.get('sections', [])
            
            # If user wants all charts (empty arrays or all sections mentioned), return all
            all_sections_mentioned = sections and len(sections) >= 3  # If 3+ sections mentioned, probably wants all
            if all_sections_mentioned and not grades and not subjects:
                print(f"[Chart Selection] Detected 'all charts' request from instructions - returning all charts")
                selected_paths = chart_paths
            elif grades or subjects or sections:
                print(f"[Chart Selection] Using criteria-based matching: grades={grades}, subjects={subjects}, sections={sections}")
                for path in chart_paths:
                    if path in seen_paths:
                        continue
                    
                    path_name_lower = Path(path).name.lower()
                    matches = True
                    
                    # Check grade match
                    if grades:
                        grade_match = False
                        for grade in grades:
                            # Match "grade1", "grade_1", "grade-1", etc.
                            if f'grade{grade}' in path_name_lower or f'grade_{grade}' in path_name_lower or f'grade-{grade}' in path_name_lower:
                                grade_match = True
                                break
                        if not grade_match:
                            matches = False
                    
                    # Check subject match
                    if matches and subjects:
                        subject_match = False
                        for subject in subjects:
                            if subject.lower() in path_name_lower:
                                subject_match = True
                                break
                        if not subject_match:
                            matches = False
                    
                    # Check section match
                    if matches and sections:
                        section_match = False
                        for section in sections:
                            if section.lower() in path_name_lower:
                                section_match = True
                                break
                        if not section_match:
                            matches = False
                    
                    if matches:
                        selected_paths.append(path)
                        seen_paths.add(path)
        
        # Sort charts by order specified in user prompt
        # If order_matters is True, prioritize by section using learned patterns from reference decks
        # Within sections, sort by grade number
        if instructions and instructions.get('order_matters', False):
            # Get learned section order from reference decks (filtered by deck_type)
            try:
                from .layout_learner import learn_layout_patterns
                script_dir = Path(__file__).parent
                base_reference_dir = script_dir / 'reference_decks'
                
                # Map deck_type to specific folder
                if deck_type:
                    deck_type_upper = deck_type.upper()
                    if deck_type_upper == 'BOY':
                        reference_deck_dir = base_reference_dir / 'BOY-DECKS'
                    elif deck_type_upper == 'MOY':
                        reference_deck_dir = base_reference_dir / 'MOY-DECKS'
                    elif deck_type_upper == 'EOY':
                        reference_deck_dir = base_reference_dir / 'EOY-DECKS'
                    else:
                        reference_deck_dir = base_reference_dir
                else:
                    reference_deck_dir = base_reference_dir
                
                patterns = learn_layout_patterns(str(reference_deck_dir))
                learned_section_order = patterns.get('section_order', [])
            except Exception as e:
                print(f"[Chart Selection] Could not load layout patterns: {e}, using default order")
                learned_section_order = []
            
            # Build section priority map from learned order, fallback to default
            section_priority_map = {}
            if learned_section_order:
                for idx, section in enumerate(learned_section_order):
                    section_priority_map[section.lower()] = idx + 1
            else:
                # Default fallback order
                section_priority_map = {
                    'section3': 1,
                    'section1': 2,
                    'section4': 3,
                    'section2': 4,
                    'section0': 5,
                    'section6': 6
                }
            
            def chart_sort_key(path: str) -> tuple:
                path_name_lower = Path(path).name.lower()
                # Section priority: use learned order, fallback to default
                section_priority = 999
                for section_name, priority in section_priority_map.items():
                    if section_name in path_name_lower:
                        section_priority = priority
                        break
                
                # Scope priority: district before school (so district charts come first for same grade/section/subject)
                scope_priority = 1 if path_name_lower.startswith('district_') else 2 if path_name_lower.startswith('school_') else 3
                
                # Extract grade number for sorting within section
                grade_num = 999
                grade_match = re.search(r'grade[_\s-]?(\d+)', path_name_lower)
                if grade_match:
                    try:
                        grade_num = int(grade_match.group(1))
                    except:
                        pass
                
                # Subject priority: math before reading (for same grade/section/scope)
                subject_priority = 0
                if 'math' in path_name_lower:
                    subject_priority = 1
                elif 'reading' in path_name_lower:
                    subject_priority = 2
                
                return (section_priority, scope_priority, grade_num, subject_priority)
            
            selected_paths.sort(key=chart_sort_key)
            print(f"[Chart Selection] Sorted {len(selected_paths)} charts by order priority")
        
        # If user said "that's it" or "only", use strict selection
        # Otherwise, if no charts selected, return all (user might have been vague)
        # Also, if selected_paths is much smaller than total and exclude_others is false, user probably wants all
        if instructions:
            exclude_others = instructions.get('exclude_others', False)
            # If we selected very few charts but user didn't say "only", they probably want all
            if len(selected_paths) < len(chart_paths) * 0.1 and not exclude_others:
                print(f"[Chart Selection] Selected only {len(selected_paths)}/{len(chart_paths)} charts but exclude_others=false - returning all charts")
                selected_paths = chart_paths
        else:
            exclude_others = False
        
        if not selected_paths and not exclude_others:
            selected_paths = chart_paths
        elif not selected_paths and exclude_others:
            # User wanted specific charts but we couldn't match - return empty to signal issue
            print(f"[Chart Selection] WARNING: Could not match any charts from selection: {selected_filenames}")
            selected_paths = chart_paths  # Fallback to all charts
        
        # Apply max slides limit if specified
        if max_slides and len(selected_paths) > max_slides:
            print(f"[Chart Selection] Limiting to {max_slides} most specific charts (had {len(selected_paths)})")
            # Prioritize: section1 > section2 > section3 > section4 > section5
            # Within same section: district > school, lower grades first, reading before math
            def chart_priority(path):
                path_name_lower = Path(path).name.lower()
                # Section priority (lower number = higher priority)
                section_priority = 999
                for i in range(1, 6):
                    if f'section{i}' in path_name_lower:
                        section_priority = i
                        break
                
                # Scope priority (district = 0, school = 1)
                scope_priority = 1 if 'school' in path_name_lower or 'SCHOOL_' in Path(path).name else 0
                
                # Grade priority (lower grade = higher priority)
                grade_num = 999
                for g in range(0, 13):
                    if f'grade{g}' in path_name_lower or f'grade_{g}' in path_name_lower:
                        grade_num = g
                        break
                
                # Subject priority (reading = 0, math = 1)
                subject_priority = 1 if 'math' in path_name_lower else 0
                
                return (section_priority, scope_priority, grade_num, subject_priority)
            
            # Sort by priority and take top max_slides
            selected_paths.sort(key=chart_priority)
            selected_paths = selected_paths[:max_slides]
            print(f"[Chart Selection] Limited to {len(selected_paths)} charts after applying max slides limit")
        
        print(f"[Chart Selection] Selected {len(selected_paths)}/{len(chart_paths)} charts")
        print(f"[Chart Selection] Reasoning: {reasoning}")
        if instructions:
            print(f"[Chart Selection] Instructions: {instructions}")
        
        return {
            'chart_selection': selected_paths,
            'instructions': instructions,
            'reasoning': reasoning
        }
        
    except Exception as e:
        print(f"[Chart Selection] Error parsing instructions: {e}, using all charts")
        return {
            'chart_selection': chart_paths,
            'instructions': None,
            'reasoning': f'Error parsing: {str(e)}'
        }

