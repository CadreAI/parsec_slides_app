"""
Decision LLM module for determining whether to use AI insights and chart selection/ordering
Uses lightweight GPT-3.5-turbo for fast, cost-effective decisions
"""
import os
import re
from typing import Dict, Optional, List
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Load environment variables
load_dotenv()


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
    chart_paths: Optional[List[str]] = None
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
    from pathlib import Path
    chart_names = [Path(p).name for p in chart_paths]
    
    # Build prompt for chart selection/ordering
    selection_prompt = f"""You are a chart selection assistant. Parse the user's instructions to determine which charts to include and in what order.

User request: "{user_prompt}"

Available charts (by filename):
{chr(10).join(f"- {name}" for name in chart_names)}

Instructions:
1. Identify which charts the user wants (by keywords like: grade, subject, section, trend, etc.)
2. Determine the order they want them in
3. If user says "that's it" or "only", exclude charts not explicitly mentioned
4. IMPORTANT: If user mentions "grades 1-4" or "grade 1-4", include ALL grades in that range (1, 2, 3, 4)
5. IMPORTANT: If user mentions demographic groups (Hispanic, Latino, Black, African American, White, etc.) or student groups → include section2 charts
6. Common patterns:
   - "grade 1 math and reading" → section3 charts for grade 1, math and reading
   - "grades 1-4" → ALL section3 charts for grades 1, 2, 3, and 4 (both math and reading)
   - "fall year to year trend" → section1 fall trends charts
   - "Hispanic", "Latino", "Black", "White", "demographic", "student group" → section2 charts
   - "first show X, then show Y" → order matters
   - "only X" → exclude everything else

Respond with JSON:
{{
    "chart_selection": ["list of chart filenames in desired order - include ALL charts matching criteria"],
    "instructions": {{
        "grades": ["list of ALL grade numbers mentioned (e.g., ['1', '2', '3', '4'] for 'grades 1-4')"],
        "subjects": ["math", "reading", or both],
        "sections": ["section1", "section2", "section3", etc. - include section2 if demographics/student groups mentioned],
        "order_matters": true/false,
        "exclude_others": true/false
    }},
    "reasoning": "brief explanation of selection"
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
        
        # Fallback: If instructions specify grades/subjects/sections, match charts by criteria
        # This handles cases where LLM didn't list all filenames but gave criteria
        if instructions and len(selected_paths) < len(chart_paths):
            grades = instructions.get('grades', [])
            subjects = instructions.get('subjects', [])
            sections = instructions.get('sections', [])
            
            if grades or subjects or sections:
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
        # If order_matters is True, prioritize by section (section3 first, then section1, then others)
        # Within sections, sort by grade number
        if instructions and instructions.get('order_matters', False):
            def chart_sort_key(path: str) -> tuple:
                path_name_lower = Path(path).name.lower()
                # Section priority: section3 (grade trends) first, then section1 (fall trends), then others
                section_priority = 999
                if 'section3' in path_name_lower:
                    section_priority = 1
                elif 'section1' in path_name_lower:
                    section_priority = 2
                elif 'section4' in path_name_lower:
                    section_priority = 3
                elif 'section2' in path_name_lower:
                    section_priority = 4
                elif 'section0' in path_name_lower:
                    section_priority = 5
                elif 'section6' in path_name_lower:
                    section_priority = 6
                
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
        exclude_others = instructions.get('exclude_others', False) if instructions else False
        if not selected_paths and not exclude_others:
            selected_paths = chart_paths
        elif not selected_paths and exclude_others:
            # User wanted specific charts but we couldn't match - return empty to signal issue
            print(f"[Chart Selection] WARNING: Could not match any charts from selection: {selected_filenames}")
            selected_paths = chart_paths  # Fallback to all charts
        
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

