"""
Enhanced chart analysis service using OpenAI GPT-4 (text-only) with chart data JSON files.
Uses actual chart data instead of image analysis for more accurate insights.
"""
import os
import base64
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Token estimation: roughly 1 token = 4 characters
def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
    return len(text) // 4

def summarize_school_data(school_data: Dict) -> str:
    """Summarize school_data to reduce token count"""
    if not school_data:
        return "No school data available"
    
    total_schools = len(school_data)
    summary = f"**School Data Summary ({total_schools} schools):**\n"
    
    # Limit to first 8 schools to keep summary concise
    schools_to_show = min(8, total_schools)
    
    # Extract key metrics from each school
    for i, (school_name, school_info) in enumerate(list(school_data.items())[:schools_to_show]):
        summary += f"\n{school_name}:\n"
        
        if isinstance(school_info, dict):
            # Extract key metrics
            if 'metrics' in school_info:
                metrics = school_info['metrics']
                if isinstance(metrics, dict):
                    key_metrics = []
                    for key in ['score_now', 'score_delta', 'hi_now', 'hi_delta', 'lo_now', 'lo_delta', 'sgp_now', 'sgp_delta']:
                        if key in metrics and metrics[key] is not None:
                            key_metrics.append(f"{key}={metrics[key]}")
                    if key_metrics:
                        summary += "  Metrics: " + ", ".join(key_metrics) + "\n"
            
            # Summarize time periods (limit to 3 most recent)
            if 'time_order' in school_info:
                time_order = school_info['time_order']
                if isinstance(time_order, list) and len(time_order) > 0:
                    recent_periods = time_order[-3:] if len(time_order) > 3 else time_order
                    summary += f"  Time Periods: {', '.join(str(t) for t in recent_periods)}"
                    if len(time_order) > 3:
                        summary += f" (showing last 3 of {len(time_order)} total)"
                    summary += "\n"
            
            # Count data points instead of listing all
            if 'pct_data' in school_info:
                pct_data = school_info['pct_data']
                if isinstance(pct_data, list) and len(pct_data) > 0:
                    summary += f"  Percentage Data: {len(pct_data)} data points\n"
                    # Show summary of latest period only
                    if len(pct_data) > 0:
                        latest = pct_data[-1] if isinstance(pct_data[-1], dict) else {}
                        if 'time_label' in latest:
                            summary += f"    Latest: {latest.get('time_label', 'N/A')}\n"
            
            if 'score_data' in school_info:
                score_data = school_info['score_data']
                if isinstance(score_data, list) and len(score_data) > 0:
                    summary += f"  Score Data: {len(score_data)} data points\n"
                    # Show latest score
                    if len(score_data) > 0:
                        latest_score = score_data[-1] if isinstance(score_data[-1], dict) else {}
                        if 'avg_score' in latest_score:
                            summary += f"    Latest Avg Score: {latest_score.get('avg_score', 'N/A')}\n"
    
    if total_schools > schools_to_show:
        summary += f"\n... ({total_schools - schools_to_show} more schools with similar data structure)\n"
    
    # Add overall summary statistics
    summary += f"\n**Overall Summary:**\n"
    summary += f"  Total Schools: {total_schools}\n"
    
    return summary


def encode_image(image_path: str) -> str:
    """Encode image to base64 string for OpenAI Vision API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_reference_deck(deck_path: str) -> Optional[str]:
    """
    Load a reference insight deck (PDF or text) to provide context
    
    Args:
        deck_path: Path to reference deck file
    
    Returns:
        String with deck content or None if not found
    """
    deck_path_obj = Path(deck_path)
    
    if not deck_path_obj.exists():
        print(f"Warning: Reference deck not found: {deck_path}")
        return None
    
    try:
        # For text files, read directly
        if deck_path_obj.suffix.lower() in ['.txt', '.md']:
            with open(deck_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # For JSON files (e.g., extracted deck data)
        elif deck_path_obj.suffix.lower() == '.json':
            with open(deck_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert to readable format
                return json.dumps(data, indent=2)
        
        # For PDF files, you would need PyPDF2 or similar
        elif deck_path_obj.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                with open(deck_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                print("PyPDF2 not installed. Install with: pip install PyPDF2")
                return None
        
        print(f"Warning: Unsupported file type for reference deck: {deck_path_obj.suffix}")
        return None
        
    except Exception as e:
        print(f"Warning: Failed to load reference deck from {deck_path}: {e}")
        return None


def load_chart_data(chart_path: str) -> Optional[Dict]:
    """Load chart data JSON file if it exists"""
    chart_path_obj = Path(chart_path)
    data_path = chart_path_obj.parent / f"{chart_path_obj.stem}_data.json"
    
    if data_path.exists():
        try:
            with open(data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load chart data from {data_path}: {e}")
    
    return None


def build_emergent_learning_prompt(
    context: str,
    data_context: str,
    reference_deck_content: Optional[str] = None,
    analysis_focus: Optional[str] = None,
    framework_level: str = "full"  # "full", "insights", "hypotheses", "opportunities"
) -> str:
    """
    Build analysis prompt using Emergent Learning framework
    
    Args:
        context: Basic chart context
        data_context: Actual chart data context
        reference_deck_content: Content from reference insight deck
        analysis_focus: Optional focus area
        framework_level: Which EL framework level to emphasize
    
    Returns:
        Formatted prompt string
    """
    
    # Build reference deck section
    reference_section = ""
    if reference_deck_content:
        # Truncate if too long (keep first 2000 chars)
        truncated_content = reference_deck_content[:2000]
        if len(reference_deck_content) > 2000:
            truncated_content += "\n... [content truncated]"
        
        reference_section = f"""

**REFERENCE INSIGHT DECK CONTEXT:**
The following is an example of how similar educational data has been analyzed using the Emergent Learning framework. Use this as a reference for the style, depth, and structure of insights:

{truncated_content}

When generating your analysis, mirror the analytical approach shown in this reference deck while focusing on the specific data in the current chart.
"""
    
    # Build framework guidance based on level
    framework_guidance = {
        "full": """
**EMERGENT LEARNING FRAMEWORK - 4 STEP ANALYSIS:**

Your analysis should support the Emergent Learning framework which has 4 quadrants:

1. **Ground Truths** (Facts & Data):
   - What the data objectively shows
   - Specific numbers, percentages, trends
   - No interpretation yet, just observable facts

2. **Insights** (Meaning from Data):
   - What patterns emerge and WHY they matter
   - The significance for student learning
   - What questions educators should consider based on each insight
   - Format: Finding → Implication → Question

3. **Hypotheses** (Predictions & Implications):
   - What the data suggests for future student performance
   - Implications for teachers, principals, and administrators
   - Forward-looking predictions based on current trends

4. **Opportunities** (Guiding Questions):
   - Questions to explore at classroom-level
   - Questions for grade-level collaboration
   - Questions for school-level investigation
   - Questions for system-level consideration

Your insights should bridge between Ground Truths and Hypotheses, helping educators move from "what happened" to "what does this mean" to "what questions should we explore."
""",
        "insights": """
**FOCUS: DEVELOPING INSIGHTS FROM GROUND TRUTHS**

Transform the observable data (ground truths) into meaningful insights with guiding questions:

For each insight, provide:
1. **Finding**: What pattern or meaning emerges from the data?
2. **Implication**: Why does this matter for student learning and outcomes?
3. **Question**: What question should educators consider to deepen understanding or guide next steps?

Consider these guiding questions:
- What patterns emerge across different time periods or groups?
- What's surprising or unexpected in the data?
- Why is this significant for teaching and learning?
- What questions should educators explore based on this finding?
- What deeper investigation does this data prompt?
- What should we be asking to understand this better?
""",
        "hypotheses": """
**FOCUS: DEVELOPING HYPOTHESES FROM INSIGHTS**

Based on the insights from the data, develop forward-looking hypotheses:
- What does this data suggest about student performance in coming weeks/months?
- What are the implications for instructional practice?
- How might this impact different student groups differently?
- What might happen if current trends continue?
- What intervention points does the data suggest?
""",
        "opportunities": """
**FOCUS: IDENTIFYING GUIDING QUESTIONS**

Translate insights and hypotheses into questions that guide exploration:
- **Classroom-Level**: Questions about teaching strategies, differentiation approaches
- **Grade-Level**: Questions for collaborative planning and shared investigation
- **School-Level**: Questions about programs, policies, resource allocation
- **System-Level**: Questions about district initiatives, professional development, systemic considerations
"""
    }
    
    selected_guidance = framework_guidance.get(framework_level, framework_guidance["full"])
    
    # Build focus instruction
    focus_instruction = ""
    if analysis_focus:
        focus_instruction = f"\n\n**ANALYSIS FOCUS:** {analysis_focus}\n"
    
    # Construct full prompt
    prompt = f"""Analyze this educational assessment data using the Emergent Learning framework principles to generate deep, actionable insights.

The data below represents assessment results. Use the actual numerical data, metrics, and distributions provided - do not reference any chart images.

{context}
{data_context}
{reference_section}
{selected_guidance}
{focus_instruction}

**OUTPUT REQUIREMENTS:**

Provide your response as a JSON object with this structure:
{{
    "title": "Clear, concise chart title (max 80 characters)",
    "description": "2-3 sentence summary connecting to larger context",
    "groundTruths": [
        "Observable fact 1 with specific numbers",
        "Observable fact 2 with specific numbers"
    ],
    "insights": [
        {{
            "finding": "Pattern or meaning derived from ground truths - what does this MEAN?",
            "implication": "What this means for educators or students",
            "question": "What question should educators explore to deepen understanding or guide next steps?"
        }},
        {{
            "finding": "Second insight connecting multiple data points or revealing trends",
            "implication": "What this means for educators or students",
            "question": "What question should educators explore to deepen understanding or guide next steps?"
        }}
    ],
    "hypotheses": [
        "Forward-looking prediction based on insights - what might happen next?",
        "Implication for instruction or student outcomes"
    ],
    "opportunities": {{
        "classroom": "Question about classroom-level teaching strategies or approaches",
        "grade": "Question for grade-level collaborative investigation",
        "school": "Question about school-level programs or policies",
        "system": "Question about district/system-level considerations"
    }},
    "subject": "math" or "reading" or null,
    "grade": "grade level if visible" or null,
    "keyMetrics": ["metric1", "metric2"],
    "confidenceLevel": "high/medium/low - based on data completeness and clarity",
    "questionsRaised": [
        "Question 1 that this data prompts us to investigate",
        "Question 2 for deeper understanding"
    ]
}}

**CRITICAL GUIDELINES:**
- Ground truths should be objective facts with specific numbers
- Insights should explain MEANING and PATTERNS, not just restate facts
- Hypotheses should be forward-looking and prompt investigation
- Opportunities should be QUESTIONS that guide exploration, not directives
- Use actual numbers from the chart data provided
- Connect insights to the Emergent Learning approach shown in reference materials
- Focus on questions that help educators understand and investigate student outcomes
"""
    
    return prompt


def analyze_chart_with_gpt(
    chart_path: str,
    chart_metadata: Optional[Dict] = None,
    api_key: Optional[str] = None,
    analysis_focus: Optional[str] = None,
    reference_deck_path: Optional[str] = None,
    framework_level: str = "full"
) -> Dict:
    """
    Analyze a single chart using OpenAI GPT-4 (text-only) with chart data JSON files.
    Uses actual chart data instead of image analysis for more accurate insights.
    
    Args:
        chart_path: Path to the chart image file (used to locate corresponding _data.json file)
        chart_metadata: Optional metadata about the chart
        api_key: OpenAI API key
        analysis_focus: Optional focus area for analysis
        reference_deck_path: Path to reference insight deck for context
        framework_level: Which EL framework level to emphasize
    
    Returns:
        Dictionary with comprehensive analysis results
    
    Note:
        Requires chart data JSON file ({chart_name}_data.json) to exist alongside chart image.
        This file is automatically created during chart generation.
    """
    if OpenAI is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required.")
    
    client = OpenAI(api_key=api_key)
    
    # Load chart data and reference deck
    chart_data = load_chart_data(chart_path)
    
    if not chart_data:
        print(f"Warning: No chart data JSON found for {chart_path}. Chart data is required for analysis.")
        raise ValueError(f"Chart data JSON file not found for {chart_path}. Please ensure chart generation saves data files.")
    
    reference_deck_content = None
    if reference_deck_path:
        reference_deck_content = load_reference_deck(reference_deck_path)
    
    # Build context from metadata and chart data
    context_parts = []
    if chart_metadata:
        if chart_metadata.get('chart_name'):
            context_parts.append(f"Chart name: {chart_metadata['chart_name']}")
        if chart_metadata.get('scope'):
            context_parts.append(f"Scope: {chart_metadata['scope']}")
        if chart_metadata.get('section'):
            context_parts.append(f"Section: {chart_metadata['section']}")
    
    context = "\n".join(context_parts) if context_parts else "This is an assessment data visualization chart."
    
    # Build comprehensive data context from JSON data
    data_context = "\n\n**CHART DATA (Use this data for analysis, NOT the image):**\n"
    
    # Add scope and window info
    if 'scope' in chart_data:
        data_context += f"Scope: {chart_data['scope']}\n"
    if 'window_filter' in chart_data:
        data_context += f"Assessment Window: {chart_data['window_filter']}\n"
    if 'subjects' in chart_data:
        data_context += f"Subjects: {', '.join(chart_data['subjects'])}\n"
    
    # Add metrics
    if 'metrics' in chart_data:
        metrics = chart_data['metrics']
        if isinstance(metrics, list) and len(metrics) > 0:
            data_context += "\n**Key Metrics:**\n"
            for i, metric in enumerate(metrics):
                if metric:
                    subject = chart_data.get('subjects', [f'Subject {i+1}'])[i] if i < len(chart_data.get('subjects', [])) else f'Subject {i+1}'
                    data_context += f"\n{subject} Metrics:\n"
                    for key, value in metric.items():
                        if value is not None:
                            data_context += f"  - {key}: {value}\n"
        elif isinstance(metrics, dict):
            data_context += "\n**Key Metrics:**\n"
            for key, value in metrics.items():
                if value is not None:
                    data_context += f"  - {key}: {value}\n"
    
    # Add time orders if available
    if 'time_orders' in chart_data:
        time_orders = chart_data['time_orders']
        if isinstance(time_orders, list) and len(time_orders) > 0:
            data_context += f"\n**Time Periods:** {', '.join(str(t) for t in time_orders)}\n"
    
    # Add percentage distribution data
    if 'pct_data' in chart_data:
        for pct_info in chart_data['pct_data']:
            if pct_info.get('data'):
                subject = pct_info.get('subject', 'Unknown')
                data_context += f"\n**{subject} Percentage Distribution by Achievement Quintile:**\n"
                # Group by time period
                time_periods = {}
                for record in pct_info['data']:
                    time_label = record.get('time_label', '')
                    quintile = record.get('achievementquintile', '')
                    pct = record.get('pct', 0)
                    if time_label not in time_periods:
                        time_periods[time_label] = {}
                    time_periods[time_label][quintile] = pct
                
                # Show all time periods, not just latest
                for time_label in sorted(time_periods.keys()):
                    data_context += f"\n  {time_label}:\n"
                    for quintile in sorted(time_periods[time_label].keys()):
                        pct = time_periods[time_label][quintile]
                        data_context += f"    - {quintile}: {pct:.1f}%\n"
    
    # Add score data
    if 'score_data' in chart_data:
        for score_info in chart_data['score_data']:
            if score_info.get('data'):
                subject = score_info.get('subject', 'Unknown')
                data_context += f"\n**{subject} Average Scores Over Time:**\n"
                for record in score_info['data']:
                    time_label = record.get('time_label', '')
                    avg_score = record.get('avg_score', 0)
                    data_context += f"  - {time_label}: {avg_score:.1f}\n"
    
    # Add any additional data fields with intelligent summarization for large datasets
    for key in ['grade_data', 'school_data', 'cohort_data', 'sgp_data']:
        if key in chart_data and chart_data[key]:
            if key == 'school_data' and isinstance(chart_data[key], dict):
                # Summarize school_data to avoid token limit issues
                data_context += summarize_school_data(chart_data[key])
            else:
                # For other data types, check size before including
                data_str = json.dumps(chart_data[key], indent=2, default=str)
                if estimate_tokens(data_str) > 2000:  # If data is too large, summarize
                    if isinstance(chart_data[key], dict):
                        data_context += f"\n**{key.replace('_', ' ').title()}:**\n"
                        data_context += f"  Contains {len(chart_data[key])} entries\n"
                        # Include summary of first few entries
                        for i, (k, v) in enumerate(list(chart_data[key].items())[:3]):
                            data_context += f"  {k}: {str(v)[:100]}...\n"
                        if len(chart_data[key]) > 3:
                            data_context += f"  ... ({len(chart_data[key]) - 3} more entries)\n"
                    elif isinstance(chart_data[key], list):
                        data_context += f"\n**{key.replace('_', ' ').title()}:**\n"
                        data_context += f"  Contains {len(chart_data[key])} records\n"
                        # Include first few records
                        for i, record in enumerate(chart_data[key][:3]):
                            data_context += f"  Record {i+1}: {str(record)[:100]}...\n"
                        if len(chart_data[key]) > 3:
                            data_context += f"  ... ({len(chart_data[key]) - 3} more records)\n"
                else:
                    data_context += f"\n**{key.replace('_', ' ').title()}:**\n"
                    data_context += data_str + "\n"
    
    # Build the enhanced prompt
    prompt = build_emergent_learning_prompt(
        context,
        data_context,
        reference_deck_content,
        analysis_focus,
        framework_level
    )
    
    # Add instruction to use data only, not image
    prompt += "\n\n**IMPORTANT:** Analyze ONLY the chart data provided above. Do not reference the chart image - use the actual numerical data, metrics, and distributions provided in the data section."
    
    # Estimate token count and choose appropriate model
    estimated_tokens = estimate_tokens(prompt)
    model = "gpt-4"
    max_context = 8192
    
    # If prompt is too large for gpt-4, use gpt-4o which has 128k context window
    if estimated_tokens > 7000:  # Leave room for response
        # Use gpt-4o which has 128k context window (much larger than gpt-4's 8k)
        model = "gpt-4o"
        max_context = 128000
        print(f"[Chart Analyzer] Prompt is large ({estimated_tokens} estimated tokens), using {model} for larger context window")
    elif estimated_tokens > 6000:
        print(f"[Chart Analyzer] Warning: Large prompt ({estimated_tokens} estimated tokens), may exceed gpt-4 context limit")
    
    try:
        # Use text-only model with appropriate context window
        # Calculate max_tokens leaving buffer for response
        max_response_tokens = min(2000, max_context - estimated_tokens - 1000)  # Leave 1k token buffer
        if max_response_tokens < 500:
            max_response_tokens = 500  # Minimum response size
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_response_tokens,
            temperature=0.4  # Slightly higher for more creative insights
        )
        
        # Extract and clean JSON
        content = response.choices[0].message.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Clean trailing commas
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        content = re.sub(r',\s*\n\s*[}\]]', lambda m: m.group(0).replace(',', ''), content)
        
        try:
            analysis = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Content preview: {content[:500]}")
            # Create fallback structure
            analysis = {
                "title": Path(chart_path).stem.replace('_', ' ').title(),
                "description": "Analysis error - see error field",
                "groundTruths": [],
                "insights": [],
                "hypotheses": [],
                "opportunities": {},
                "subject": None,
                "grade": None,
                "keyMetrics": [],
                "confidenceLevel": "low",
                "questionsRaised": [],
                "error": str(e)
            }
        
        # Add metadata
        analysis['chart_path'] = chart_path
        analysis['chart_name'] = Path(chart_path).stem
        analysis['framework_level'] = framework_level
        analysis['used_reference_deck'] = reference_deck_path is not None
        
        return analysis
        
    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__
        
        # Check for context length errors
        if "context_length" in error_str.lower() or "maximum context length" in error_str.lower():
            print(f"[Chart Analyzer] Context length exceeded for {chart_path}")
            print(f"[Chart Analyzer] Estimated tokens: {estimated_tokens}, Model: {model}, Max context: {max_context}")
            
            # Try with gpt-4o if we haven't already
            if model != "gpt-4o":
                print(f"[Chart Analyzer] Retrying with gpt-4o (larger context window)...")
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        max_tokens=2000,
                        temperature=0.4
                    )
                    # Process response same as above
                    content = response.choices[0].message.content.strip()
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    content = re.sub(r',(\s*[}\]])', r'\1', content)
                    content = re.sub(r',\s*\n\s*[}\]]', lambda m: m.group(0).replace(',', ''), content)
                    analysis = json.loads(content)
                    analysis['chart_path'] = chart_path
                    analysis['chart_name'] = Path(chart_path).stem
                    analysis['framework_level'] = framework_level
                    analysis['used_reference_deck'] = reference_deck_path is not None
                    print(f"[Chart Analyzer] ✓ Successfully analyzed with gpt-4o")
                    return analysis
                except Exception as retry_error:
                    print(f"[Chart Analyzer] Retry with gpt-4o also failed: {retry_error}")
            
            # If retry failed or already using gpt-4o, return error analysis
            return {
                "title": Path(chart_path).stem.replace('_', ' ').title(),
                "description": "Chart data too large for analysis - context length exceeded",
                "groundTruths": [],
                "insights": ["This chart contains too much data to analyze automatically. Please review manually."],
                "hypotheses": [],
                "opportunities": {},
                "subject": None,
                "grade": None,
                "keyMetrics": [],
                "confidenceLevel": "low",
                "questionsRaised": [],
                "error": "context_length_exceeded",
                "chart_path": chart_path,
                "chart_name": Path(chart_path).stem
            }
        
        print(f"Error analyzing chart {chart_path}: {e}")
        raise


def analyze_multiple_charts_with_gpt(
    chart_batch: List[Tuple[str, Dict]],
    api_key: Optional[str] = None,
    analysis_focus: Optional[str] = None,
    reference_deck_path: Optional[str] = None,
    framework_level: str = "full"
) -> List[Dict]:
    """
    Analyze multiple charts in a single API call to reduce API usage.
    
    Args:
        chart_batch: List of (chart_path, metadata) tuples
        api_key: OpenAI API key
        analysis_focus: Optional focus area for analysis
        reference_deck_path: Path to reference insight deck
        framework_level: EL framework level to emphasize
    
    Returns:
        List of analysis dictionaries, one per chart
    """
    if OpenAI is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    if not chart_batch:
        return []
    
    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required.")
    
    client = OpenAI(api_key=api_key)
    
    # Load all chart data
    charts_info = []
    for chart_item in chart_batch:
        # Handle both tuple (path, metadata) and ensure metadata is a dict
        if isinstance(chart_item, tuple) and len(chart_item) == 2:
            chart_path, metadata = chart_item
            # Ensure metadata is a dict, not a string
            if not isinstance(metadata, dict):
                metadata = {'chart_name': Path(chart_path).stem, 'chart_path': str(chart_path)}
        else:
            print(f"Warning: Unexpected chart item format: {type(chart_item)}. Skipping.")
            continue
        
        chart_data = load_chart_data(chart_path)
        if not chart_data:
            print(f"Warning: No chart data JSON found for {chart_path}. Skipping.")
            continue
        
        charts_info.append({
            'path': chart_path,
            'metadata': metadata,
            'data': chart_data
        })
    
    if not charts_info:
        return []
    
    # Load reference deck once
    reference_deck_content = None
    if reference_deck_path:
        reference_deck_content = load_reference_deck(reference_deck_path)
    
    # Build multi-chart prompt
    prompt_parts = [
        f"You are analyzing {len(charts_info)} educational assessment charts. ",
        "For each chart, provide a comprehensive analysis using the Emergent Learning framework.\n\n"
    ]
    
    # Add each chart's data
    for idx, chart_info in enumerate(charts_info, 1):
        chart_path = chart_info['path']
        metadata = chart_info['metadata']
        chart_data = chart_info['data']
        
        chart_name = metadata.get('chart_name', Path(chart_path).stem)
        prompt_parts.append(f"{'='*60}\n")
        prompt_parts.append(f"CHART {idx}: {chart_name}\n")
        prompt_parts.append(f"{'='*60}\n\n")
        
        # Build context
        context_parts = []
        if metadata.get('chart_name'):
            context_parts.append(f"Chart name: {metadata['chart_name']}")
        if metadata.get('scope'):
            context_parts.append(f"Scope: {metadata['scope']}")
        if metadata.get('section'):
            context_parts.append(f"Section: {metadata['section']}")
        
        context = "\n".join(context_parts) if context_parts else "This is an assessment data visualization chart."
        
        # Build data context (similar to single chart analysis)
        data_context = "\n**CHART DATA (Use this data for analysis, NOT the image):**\n"
        
        if 'scope' in chart_data:
            data_context += f"Scope: {chart_data['scope']}\n"
        if 'window_filter' in chart_data:
            data_context += f"Assessment Window: {chart_data['window_filter']}\n"
        if 'subjects' in chart_data:
            data_context += f"Subjects: {', '.join(chart_data['subjects'])}\n"
        
        # Add metrics (summarized)
        if 'metrics' in chart_data:
            metrics = chart_data['metrics']
            if isinstance(metrics, list) and len(metrics) > 0:
                data_context += "\n**Key Metrics:**\n"
                for i, metric in enumerate(metrics[:2]):  # Limit to 2 subjects
                    if metric:
                        subject = chart_data.get('subjects', [f'Subject {i+1}'])[i] if i < len(chart_data.get('subjects', [])) else f'Subject {i+1}'
                        data_context += f"\n{subject} Metrics:\n"
                        for key, value in list(metric.items())[:5]:  # Limit to 5 metrics
                            if value is not None:
                                data_context += f"  - {key}: {value}\n"
            elif isinstance(metrics, dict):
                data_context += "\n**Key Metrics:**\n"
                for key, value in list(metrics.items())[:5]:  # Limit to 5 metrics
                    if value is not None:
                        data_context += f"  - {key}: {value}\n"
        
        # Add time orders
        if 'time_orders' in chart_data:
            time_orders = chart_data['time_orders']
            if isinstance(time_orders, list) and len(time_orders) > 0:
                data_context += f"\n**Time Periods:** {', '.join(str(t) for t in time_orders[:5])}\n"
        
        # Add percentage data (summarized - latest period only)
        if 'pct_data' in chart_data:
            pct_data = chart_data['pct_data']
            # Handle both list and dict formats
            if isinstance(pct_data, list):
                pct_items = pct_data[:2]  # Limit to 2 subjects
            elif isinstance(pct_data, dict):
                # Convert dict to list format for processing
                pct_items = []
                for key, value in list(pct_data.items())[:2]:
                    if isinstance(value, list):
                        pct_items.append({'subject': key, 'data': value})
                    else:
                        pct_items.append({'subject': key, 'data': []})
            else:
                pct_items = []
            
            for pct_info in pct_items:
                if isinstance(pct_info, dict) and pct_info.get('data'):
                    subject = pct_info.get('subject', 'Unknown')
                    data_context += f"\n**{subject} Latest Percentage Distribution:**\n"
                    # Show only latest time period
                    records = pct_info['data']
                    if records and isinstance(records, list):
                        latest = records[-1]
                        if isinstance(latest, dict):
                            time_label = latest.get('time_label', '')
                            data_context += f"  {time_label}:\n"
                            for key in ['achievementquintile', 'pct']:
                                if key in latest:
                                    data_context += f"    {key}: {latest[key]}\n"
        
        # Add score data (latest only)
        if 'score_data' in chart_data:
            score_data = chart_data['score_data']
            # Handle both list and dict formats
            if isinstance(score_data, list):
                score_items = score_data[:2]  # Limit to 2 subjects
            elif isinstance(score_data, dict):
                # Convert dict to list format for processing
                score_items = []
                for key, value in list(score_data.items())[:2]:
                    if isinstance(value, list):
                        score_items.append({'subject': key, 'data': value})
                    else:
                        score_items.append({'subject': key, 'data': []})
            else:
                score_items = []
            
            for score_info in score_items:
                if isinstance(score_info, dict) and score_info.get('data'):
                    subject = score_info.get('subject', 'Unknown')
                    data_context += f"\n**{subject} Latest Average Score:**\n"
                    records = score_info['data']
                    if records and isinstance(records, list):
                        latest = records[-1]
                        if isinstance(latest, dict):
                            time_label = latest.get('time_label', '')
                            avg_score = latest.get('avg_score', 0)
                            data_context += f"  {time_label}: {avg_score:.1f}\n"
        
        # Summarize large data fields
        for key in ['school_data', 'grade_data', 'cohort_data', 'sgp_data']:
            if key in chart_data and chart_data[key]:
                if key == 'school_data' and isinstance(chart_data[key], dict):
                    data_context += summarize_school_data(chart_data[key])
                else:
                    data_str = json.dumps(chart_data[key], indent=2, default=str)
                    if estimate_tokens(data_str) > 1000:  # More aggressive truncation for multi-chart
                        if isinstance(chart_data[key], dict):
                            data_context += f"\n**{key.replace('_', ' ').title()}:** {len(chart_data[key])} entries\n"
                        elif isinstance(chart_data[key], list):
                            data_context += f"\n**{key.replace('_', ' ').title()}:** {len(chart_data[key])} records\n"
        
        prompt_parts.append(context)
        prompt_parts.append(data_context)
        prompt_parts.append("\n")
    
    # Build the analysis prompt
    analysis_prompt = build_emergent_learning_prompt(
        f"Multiple charts analysis ({len(charts_info)} charts)",
        "\n".join(prompt_parts),
        reference_deck_content,
        analysis_focus,
        framework_level
    )
    
    # Add instruction for multiple charts
    analysis_prompt += f"\n\n**IMPORTANT:** You are analyzing {len(charts_info)} charts. "
    analysis_prompt += "Return a JSON array with one analysis object per chart, in the same order as the charts were presented. "
    analysis_prompt += "Each analysis should follow the same structure as a single chart analysis. "
    analysis_prompt += "Use the chart data provided, NOT the images.\n\n"
    analysis_prompt += "Return format:\n"
    analysis_prompt += "[\n"
    analysis_prompt += "  {{\n"
    analysis_prompt += "    \"chart_name\": \"name of chart\",\n"
    analysis_prompt += "    \"title\": \"...\",\n"
    analysis_prompt += "    \"description\": \"...\",\n"
    analysis_prompt += "    ... (rest of analysis structure)\n"
    analysis_prompt += "  }},\n"
    analysis_prompt += "  ... (one object per chart)\n"
    analysis_prompt += "]\n"
    
    # Estimate tokens and choose model
    estimated_tokens = estimate_tokens(analysis_prompt)
    model = "gpt-4o"  # Always use gpt-4o for multi-chart to handle larger prompts
    max_context = 128000
    
    if estimated_tokens > 100000:
        print(f"[Chart Analyzer] Warning: Very large multi-chart prompt ({estimated_tokens} tokens), may need further optimization")
    
    try:
        max_response_tokens = min(4000, max_context - estimated_tokens - 2000)  # Larger response for multiple charts
        if max_response_tokens < 1000:
            max_response_tokens = 1000
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            max_tokens=max_response_tokens,
            temperature=0.4
        )
        
        # Extract and parse JSON array
        content = response.choices[0].message.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Clean trailing commas
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        content = re.sub(r',\s*\n\s*[}\]]', lambda m: m.group(0).replace(',', ''), content)
        
        try:
            analyses_list = json.loads(content)
            if not isinstance(analyses_list, list):
                # If single object returned, wrap it
                analyses_list = [analyses_list]
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Content preview: {content[:500]}")
            # Fallback: return error analyses
            analyses_list = []
            for chart_path, metadata in chart_batch:
                analyses_list.append({
                    "title": Path(chart_path).stem.replace('_', ' ').title(),
                    "description": "Analysis error - JSON parse failed",
                    "groundTruths": [],
                    "insights": [],
                    "hypotheses": [],
                    "opportunities": {},
                    "chart_path": chart_path,
                    "chart_name": Path(chart_path).stem,
                    "error": str(e)
                })
        
        # Ensure each analysis has chart_path and chart_name
        for i, analysis in enumerate(analyses_list):
            if i < len(charts_info):
                chart_path = charts_info[i]['path']
                metadata = charts_info[i]['metadata']
                analysis['chart_path'] = chart_path
                analysis['chart_name'] = metadata.get('chart_name', Path(chart_path).stem)
                analysis['framework_level'] = framework_level
                analysis['used_reference_deck'] = reference_deck_path is not None
        
        return analyses_list
        
    except Exception as e:
        error_str = str(e)
        print(f"[Chart Analyzer] Error in multi-chart analysis: {e}")
        
        # Return error analyses for all charts
        error_analyses = []
        for chart_path, metadata in chart_batch:
            error_analyses.append({
                "title": Path(chart_path).stem.replace('_', ' ').title(),
                "description": f"Analysis error: {str(e)}",
                "groundTruths": [],
                "insights": [],
                "hypotheses": [],
                "opportunities": {},
                "chart_path": chart_path,
                "chart_name": Path(chart_path).stem,
                "error": str(e)
            })
        return error_analyses


def analyze_charts_batch(
    chart_batch: List[Tuple[str, Dict]],
    api_key: Optional[str] = None,
    batch_num: int = 1,
    analysis_focus: Optional[str] = None,
    reference_deck_path: Optional[str] = None,
    framework_level: str = "full",
    charts_per_api_call: int = 8  # Default optimized for single-worker environments
) -> List[Dict]:
    """
    Analyze a batch of charts with EL framework.
    Groups charts into API calls to reduce total API usage.
    
    Args:
        chart_batch: List of (chart_path, metadata) tuples
        api_key: OpenAI API key
        batch_num: Batch number for logging
        analysis_focus: Optional focus area for analysis
        reference_deck_path: Path to reference insight deck
        framework_level: EL framework level to emphasize
        charts_per_api_call: Number of charts to analyze per API call (default: 3)
    
    Returns:
        List of analysis dictionaries
    """
    analyses = []
    
    print(f"[Batch {batch_num}] Analyzing {len(chart_batch)} charts ({charts_per_api_call} per API call)...")
    
    # Group charts into sub-batches for API calls
    sub_batches = []
    for i in range(0, len(chart_batch), charts_per_api_call):
        sub_batch = chart_batch[i:i + charts_per_api_call]
        sub_batch_num = (i // charts_per_api_call) + 1
        total_sub_batches = (len(chart_batch) + charts_per_api_call - 1) // charts_per_api_call
        sub_batches.append((sub_batch, sub_batch_num, total_sub_batches))
    
    # Process sub-batches in parallel (optimized for single-worker environments)
    # For I/O-bound API calls, 2-3 concurrent workers is optimal on single-worker systems
    # This prevents overwhelming the worker while still providing parallelism
    max_workers = min(3, len(sub_batches))
    print(f"  Using {max_workers} concurrent API call(s) for {len(sub_batches)} sub-batch(es)")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sub_batch = {}
        for sub_batch, sub_batch_num, total_sub_batches in sub_batches:
            print(f"  [Sub-batch {sub_batch_num}/{total_sub_batches}] Queuing {len(sub_batch)} chart(s)...")
            future = executor.submit(
                analyze_multiple_charts_with_gpt,
                sub_batch,
                api_key,
                analysis_focus,
                reference_deck_path,
                framework_level
            )
            future_to_sub_batch[future] = (sub_batch, sub_batch_num, total_sub_batches)
        
        # Collect results as they complete
        for future in as_completed(future_to_sub_batch):
            sub_batch, sub_batch_num, total_sub_batches = future_to_sub_batch[future]
            try:
                sub_analyses = future.result()
                for analysis in sub_analyses:
                    analyses.append(analysis)
                    chart_name = analysis.get('chart_name', 'Unknown')
                    print(f"    ✓ {chart_name}")
            except Exception as e:
                print(f"  ✗ Error in sub-batch {sub_batch_num}: {e}")
                # Fallback to individual analysis for this sub-batch
                print(f"  Falling back to individual analysis for {len(sub_batch)} chart(s)...")
                for chart_path, metadata in sub_batch:
                    try:
                        analysis = analyze_chart_with_gpt(
                            chart_path,
                            metadata,
                            api_key,
                            analysis_focus,
                            reference_deck_path,
                            framework_level
                        )
                        analyses.append(analysis)
                        chart_name = metadata.get('chart_name', Path(chart_path).stem)
                        print(f"    ✓ {chart_name} (individual)")
                    except Exception as fallback_error:
                        print(f"    ✗ {chart_path}: {fallback_error}")
                        analyses.append({
                            "title": Path(chart_path).stem.replace('_', ' ').title(),
                            "description": f"Error: {str(fallback_error)}",
                            "groundTruths": [],
                            "insights": [],
                            "hypotheses": [],
                            "opportunities": {},
                            "chart_path": chart_path,
                            "chart_name": Path(chart_path).stem,
                            "error": str(fallback_error)
                        })
    
    return analyses


def analyze_charts_batch_paths(
    chart_paths: List[str],
    api_key: Optional[str] = None,
    batch_size: int = 10,
    analysis_focus: Optional[str] = None,
    reference_deck_path: Optional[str] = None,
    framework_level: str = "full",
    charts_per_api_call: int = 8  # Default optimized for single-worker environments
) -> List[Dict]:
    """
    Analyze charts from a list of file paths with Emergent Learning framework.
    Uses batched API calls to reduce total API usage.
    
    Args:
        chart_paths: List of chart file paths
        api_key: OpenAI API key
        batch_size: Number of charts to process per batch (for organization)
        analysis_focus: Optional focus area for analysis
        reference_deck_path: Optional path to reference insight deck
        framework_level: EL framework level to emphasize
        charts_per_api_call: Number of charts to analyze per API call (default: 3)
    
    Returns:
        List of analysis dictionaries
    """
    if not chart_paths:
        return []
    
    # Convert paths to (path, metadata) tuples expected by analyze_charts_batch
    chart_list = []
    for chart_path in chart_paths:
        chart_path_obj = Path(chart_path)
        metadata = {
            'chart_name': chart_path_obj.stem,
            'chart_path': str(chart_path)
        }
        chart_list.append((str(chart_path), metadata))
    
    print(f"Analyzing {len(chart_list)} charts with EL framework (level: {framework_level})")
    print(f"Using batched API calls: {charts_per_api_call} charts per API call")
    if reference_deck_path:
        print(f"Using reference deck: {reference_deck_path}")
    
    all_analyses = []
    total_batches = (len(chart_list) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(chart_list), batch_size):
        batch = chart_list[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        batch_analyses = analyze_charts_batch(
            batch, 
            api_key, 
            batch_num,
            analysis_focus=analysis_focus,
            reference_deck_path=reference_deck_path,
            framework_level=framework_level,
            charts_per_api_call=charts_per_api_call
        )
        all_analyses.extend(batch_analyses)
    
    total_api_calls = (len(chart_list) + charts_per_api_call - 1) // charts_per_api_call
    print(f"\n✅ Completed {len(all_analyses)} analyses in ~{total_api_calls} API calls (vs {len(chart_list)} individual calls)")
    return all_analyses


def analyze_charts_from_index(
    chart_index_path: str,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    max_charts: Optional[int] = None,
    batch_size: int = 10,
    reference_deck_path: Optional[str] = None,
    framework_level: str = "full"
) -> List[Dict]:
    """
    Analyze charts from index with Emergent Learning framework
    
    Args:
        chart_index_path: Path to chart_index.csv
        output_dir: Base directory for chart paths
        api_key: OpenAI API key
        max_charts: Max number of charts to analyze
        batch_size: Charts per batch
        reference_deck_path: Path to reference insight deck
        framework_level: EL framework level to emphasize
    """
    chart_index_path = Path(chart_index_path)
    df = pd.read_csv(chart_index_path)
    
    base_dir = Path(output_dir) if output_dir else chart_index_path.parent
    
    if max_charts:
        df = df.head(max_charts)
    
    chart_list = []
    for idx, row in df.iterrows():
        chart_path = base_dir / row['file_path']
        if not chart_path.exists():
            print(f"Warning: Chart not found: {chart_path}")
            continue
        
        metadata = {
            'chart_name': row.get('chart_name', ''),
            'scope': row.get('scope', ''),
            'section': row.get('section', '')
        }
        chart_list.append((str(chart_path), metadata))
    
    print(f"Analyzing {len(chart_list)} charts with EL framework (level: {framework_level})")
    if reference_deck_path:
        print(f"Using reference deck: {reference_deck_path}")
    
    all_analyses = []
    total_batches = (len(chart_list) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(chart_list), batch_size):
        batch = chart_list[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        batch_analyses = analyze_charts_batch(
            batch, 
            api_key, 
            batch_num,
            reference_deck_path=reference_deck_path,
            framework_level=framework_level
        )
        all_analyses.extend(batch_analyses)
    
    print(f"\n✅ Completed {len(all_analyses)} analyses")
    return all_analyses


def save_analyses_to_json(analyses: List[Dict], output_path: str):
    """Save analyses to JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(analyses, f, indent=2)
    
    print(f"Saved {len(analyses)} analyses to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze charts with Emergent Learning framework")
    parser.add_argument("--chart-index", required=True, help="Path to chart_index.csv")
    parser.add_argument("--output-dir", help="Base directory for chart paths")
    parser.add_argument("--output-json", help="Output JSON file")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--max-charts", type=int, help="Max charts to analyze")
    parser.add_argument("--reference-deck", help="Path to reference insight deck")
    parser.add_argument("--framework-level", 
                       choices=["full", "insights", "hypotheses", "opportunities"],
                       default="full",
                       help="EL framework level to emphasize")
    
    args = parser.parse_args()
    
    analyses = analyze_charts_from_index(
        args.chart_index,
        args.output_dir,
        args.api_key,
        args.max_charts,
        reference_deck_path=args.reference_deck,
        framework_level=args.framework_level
    )
    
    if args.output_json:
        save_analyses_to_json(analyses, args.output_json)
    else:
        print(json.dumps(analyses, indent=2))
