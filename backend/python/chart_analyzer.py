"""
Enhanced chart analysis service using OpenAI Vision API with Emergent Learning framework
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
   - What educators should consider doing based on each insight
   - Format: Finding → Implication → Recommendation

3. **Hypotheses** (Predictions & Implications):
   - What the data suggests for future student performance
   - Implications for teachers, principals, and administrators
   - Forward-looking predictions based on current trends

4. **Opportunities** (Actionable Ideas):
   - Specific actions at classroom-level
   - Grade-level interventions
   - School-level initiatives
   - System-level changes

Your insights should bridge between Ground Truths and Hypotheses, helping educators move from "what happened" to "what does this mean" to "what should we do."
""",
        "insights": """
**FOCUS: DEVELOPING INSIGHTS FROM GROUND TRUTHS**

Transform the observable data (ground truths) into meaningful insights with actionable guidance:

For each insight, provide:
1. **Finding**: What pattern or meaning emerges from the data?
2. **Implication**: Why does this matter for student learning and outcomes?
3. **Recommendation**: What should educators consider doing in response?

Consider these guiding questions:
- What patterns emerge across different time periods or groups?
- What's surprising or unexpected in the data?
- Why is this significant for teaching and learning?
- What should educators pay attention to based on this finding?
- What initial steps or considerations does this suggest?
- How confident can we be in acting on this insight?
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
**FOCUS: IDENTIFYING ACTIONABLE OPPORTUNITIES**

Translate insights and hypotheses into concrete opportunities:
- **Classroom-Level**: Specific teaching strategies, differentiation approaches
- **Grade-Level**: Collaborative planning, shared interventions
- **School-Level**: Programs, policies, resource allocation
- **System-Level**: District initiatives, professional development, systemic changes
"""
    }
    
    selected_guidance = framework_guidance.get(framework_level, framework_guidance["full"])
    
    # Build focus instruction
    focus_instruction = ""
    if analysis_focus:
        focus_instruction = f"\n\n**ANALYSIS FOCUS:** {analysis_focus}\n"
    
    # Construct full prompt
    prompt = f"""Analyze this NWEA assessment data visualization chart using the Emergent Learning framework principles to generate deep, actionable insights.

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
            "recommendation": "Specific action or consideration based on this insight"
        }},
        {{
            "finding": "Second insight connecting multiple data points or revealing trends",
            "implication": "What this means for educators or students",
            "recommendation": "Specific action or consideration based on this insight"
        }}
    ],
    "hypotheses": [
        "Forward-looking prediction based on insights - what might happen next?",
        "Implication for instruction or student outcomes"
    ],
    "opportunities": {{
        "classroom": "Specific classroom-level action teachers could take",
        "grade": "Grade-level collaborative opportunity",
        "school": "School-level initiative or program",
        "system": "District/system-level recommendation"
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
- Hypotheses should be forward-looking and actionable
- Opportunities should be specific and tied to the data
- Use actual numbers from the chart data provided
- Connect insights to the Emergent Learning approach shown in reference materials
- Focus on what educators need to know to improve student outcomes
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
    Analyze a single chart using OpenAI Vision API with Emergent Learning framework
    
    Args:
        chart_path: Path to the chart image file
        chart_metadata: Optional metadata about the chart
        api_key: OpenAI API key
        analysis_focus: Optional focus area for analysis
        reference_deck_path: Path to reference insight deck for context
        framework_level: Which EL framework level to emphasize
    
    Returns:
        Dictionary with comprehensive analysis results
    """
    if OpenAI is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required.")
    
    client = OpenAI(api_key=api_key)
    
    # Encode image
    base64_image = encode_image(chart_path)
    
    # Load chart data and reference deck
    chart_data = load_chart_data(chart_path)
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
    
    context = "\n".join(context_parts) if context_parts else "This is an NWEA assessment data visualization chart."
    
    # Build data context
    data_context = ""
    if chart_data:
        data_context = "\n\n**ACTUAL CHART DATA:**\n"
        
        if 'metrics' in chart_data:
            metrics = chart_data['metrics']
            if isinstance(metrics, list) and len(metrics) > 0:
                for i, metric in enumerate(metrics):
                    if metric:
                        data_context += f"\nSubject {i+1} Metrics:\n"
                        for key, value in metric.items():
                            if value is not None:
                                data_context += f"  - {key}: {value}\n"
        
        if 'pct_data' in chart_data:
            for pct_info in chart_data['pct_data']:
                if pct_info.get('data'):
                    subject = pct_info.get('subject', 'Unknown')
                    data_context += f"\n{subject} Percentage Distribution:\n"
                    latest_periods = {}
                    for record in pct_info['data']:
                        time_label = record.get('time_label', '')
                        quintile = record.get('achievementquintile', '')
                        pct = record.get('pct', 0)
                        if time_label not in latest_periods:
                            latest_periods[time_label] = {}
                        latest_periods[time_label][quintile] = pct
                    
                    if latest_periods:
                        latest_time = max(latest_periods.keys())
                        data_context += f"  Latest period ({latest_time}):\n"
                        for quintile, pct in latest_periods[latest_time].items():
                            data_context += f"    - {quintile}: {pct:.1f}%\n"
        
        if 'score_data' in chart_data:
            for score_info in chart_data['score_data']:
                if score_info.get('data'):
                    subject = score_info.get('subject', 'Unknown')
                    data_context += f"\n{subject} Average Scores:\n"
                    for record in score_info['data']:
                        time_label = record.get('time_label', '')
                        avg_score = record.get('avg_score', 0)
                        data_context += f"  - {time_label}: {avg_score:.1f}\n"
    
    # Build the enhanced prompt
    prompt = build_emergent_learning_prompt(
        context,
        data_context,
        reference_deck_content,
        analysis_focus,
        framework_level
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=2000,  # Increased for more comprehensive analysis
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
        print(f"Error analyzing chart {chart_path}: {e}")
        raise


def analyze_charts_batch(
    chart_batch: List[Tuple[str, Dict]],
    api_key: Optional[str] = None,
    batch_num: int = 1,
    analysis_focus: Optional[str] = None,
    reference_deck_path: Optional[str] = None,
    framework_level: str = "full"
) -> List[Dict]:
    """Analyze a batch of charts with EL framework"""
    analyses = []
    
    def analyze_single(chart_path: str, metadata: Dict) -> Dict:
        try:
            return analyze_chart_with_gpt(
                chart_path, 
                metadata, 
                api_key, 
                analysis_focus,
                reference_deck_path,
                framework_level
            )
        except Exception as e:
            print(f"Error analyzing chart {chart_path}: {e}")
            return {
                "title": Path(chart_path).stem.replace('_', ' ').title(),
                "description": f"Error: {str(e)}",
                "groundTruths": [],
                "insights": [],
                "hypotheses": [],
                "opportunities": {},
                "chart_path": chart_path,
                "chart_name": Path(chart_path).stem,
                "error": str(e)
            }
    
    print(f"[Batch {batch_num}] Analyzing {len(chart_batch)} charts with EL framework...")
    with ThreadPoolExecutor(max_workers=min(10, len(chart_batch))) as executor:
        future_to_chart = {
            executor.submit(analyze_single, chart_path, metadata): (chart_path, metadata)
            for chart_path, metadata in chart_batch
        }
        
        for future in as_completed(future_to_chart):
            chart_path, metadata = future_to_chart[future]
            try:
                analysis = future.result()
                analyses.append(analysis)
                chart_name = metadata.get('chart_name', Path(chart_path).stem)
                print(f"  ✓ {chart_name}")
            except Exception as e:
                print(f"  ✗ {chart_path}: {e}")
    
    return analyses


def analyze_charts_batch_paths(
    chart_paths: List[str],
    api_key: Optional[str] = None,
    batch_size: int = 10,
    analysis_focus: Optional[str] = None,
    reference_deck_path: Optional[str] = None,
    framework_level: str = "full"
) -> List[Dict]:
    """
    Analyze charts from a list of file paths with Emergent Learning framework
    
    Args:
        chart_paths: List of chart file paths
        api_key: OpenAI API key
        batch_size: Number of charts to analyze per batch
        analysis_focus: Optional focus area for analysis
        reference_deck_path: Optional path to reference insight deck
        framework_level: EL framework level to emphasize
    
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
            framework_level=framework_level
        )
        all_analyses.extend(batch_analyses)
    
    print(f"\n✅ Completed {len(all_analyses)} analyses")
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
