"""
Chart analysis service using OpenAI Vision API to generate insights from charts
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


def load_chart_data(chart_path: str) -> Optional[Dict]:
    """
    Load chart data JSON file if it exists
    
    Args:
        chart_path: Path to the chart image file
    
    Returns:
        Dictionary with chart data or None if not found
    """
    chart_path_obj = Path(chart_path)
    data_path = chart_path_obj.parent / f"{chart_path_obj.stem}_data.json"
    
    if data_path.exists():
        try:
            with open(data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load chart data from {data_path}: {e}")
    
    return None


def analyze_chart_with_gpt(
    chart_path: str,
    chart_metadata: Optional[Dict] = None,
    api_key: Optional[str] = None
) -> Dict:
    """
    Analyze a single chart using OpenAI Vision API
    
    Args:
        chart_path: Path to the chart image file
        chart_metadata: Optional metadata about the chart (name, scope, section, etc.)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
    
    Returns:
        Dictionary with analysis results including title, description, and insights
    """
    if OpenAI is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    # Encode image
    base64_image = encode_image(chart_path)
    
    # Try to load chart data JSON if available
    chart_data = load_chart_data(chart_path)
    
    # Build context from metadata and chart data
    context_parts = []
    if chart_metadata:
        if chart_metadata.get('chart_name'):
            context_parts.append(f"Chart name: {chart_metadata['chart_name']}")
        if chart_metadata.get('scope'):
            context_parts.append(f"Scope: {chart_metadata['scope']}")
        if chart_metadata.get('section'):
            context_parts.append(f"Section: {chart_metadata['section']}")
    
    # Add chart data to context if available
    data_context = ""
    if chart_data:
        data_context = "\n\n**Actual Chart Data:**\n"
        
        # Extract key metrics
        if 'metrics' in chart_data:
            metrics = chart_data['metrics']
            if isinstance(metrics, list) and len(metrics) > 0:
                for i, metric in enumerate(metrics):
                    if metric:
                        data_context += f"\nSubject {i+1} Metrics:\n"
                        for key, value in metric.items():
                            if value is not None:
                                data_context += f"  - {key}: {value}\n"
        
        # Add percentage data summary
        if 'pct_data' in chart_data:
            for pct_info in chart_data['pct_data']:
                if pct_info.get('data'):
                    subject = pct_info.get('subject', 'Unknown')
                    data_context += f"\n{subject} Percentage Distribution:\n"
                    # Summarize latest time period
                    latest_periods = {}
                    for record in pct_info['data']:
                        time_label = record.get('time_label', '')
                        quintile = record.get('achievementquintile', '')
                        pct = record.get('pct', 0)
                        if time_label not in latest_periods:
                            latest_periods[time_label] = {}
                        latest_periods[time_label][quintile] = pct
                    
                    # Show most recent period
                    if latest_periods:
                        latest_time = max(latest_periods.keys())
                        data_context += f"  Latest period ({latest_time}):\n"
                        for quintile, pct in latest_periods[latest_time].items():
                            data_context += f"    - {quintile}: {pct:.1f}%\n"
        
        # Add score data summary
        if 'score_data' in chart_data:
            for score_info in chart_data['score_data']:
                if score_info.get('data'):
                    subject = score_info.get('subject', 'Unknown')
                    data_context += f"\n{subject} Average Scores:\n"
                    for record in score_info['data']:
                        time_label = record.get('time_label', '')
                        avg_score = record.get('avg_score', 0)
                        data_context += f"  - {time_label}: {avg_score:.1f}\n"
    
    context = "\n".join(context_parts) if context_parts else "This is an NWEA assessment data visualization chart."
    if data_context:
        context += data_context
    
    # Create prompt for analysis
    prompt = f"""Analyze this NWEA assessment data visualization chart and provide insights in JSON format.

{context}

Please analyze the chart and provide:
1. A clear, concise title (max 80 characters)
2. A brief description/summary (2-3 sentences)
3. Key insights as an array of EXACTLY 2 bullet points - only the most important and impactful insights
4. Subject (math or reading, if applicable)
5. Grade level (if visible in chart)
6. Key metrics or trends observed

Return your response as a JSON object with this exact structure:
{{
    "title": "Chart title here",
    "description": "Brief description of what the chart shows",
    "insights": [
        "Most important insight - be specific with numbers",
        "Second most important insight - be specific with numbers"
    ],
    "subject": "math" or "reading" or null,
    "grade": "grade level if visible" or null,
    "keyMetrics": ["metric1", "metric2"]
}}

IMPORTANT: Provide EXACTLY 2 insights only. Prioritize:
- The most significant trends or changes (largest percentage changes, biggest score improvements/declines)
- Actionable findings that require attention (areas needing intervention, notable improvements)
- Focus on the most impactful data points

Be specific about numbers, percentages, and comparisons when visible in the chart.

If actual chart data is provided above, use those exact numbers and metrics in your analysis rather than estimating from the visual."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-vision-preview" for older models
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.3  # Lower temperature for more consistent, factual output
        )
        
        # Extract JSON from response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON if wrapped in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Clean up common JSON issues
        # Remove trailing commas before closing brackets/braces (fixes "Illegal trailing comma" errors)
        # Fix trailing commas in arrays: ], -> ]
        content = re.sub(r',(\s*])', r'\1', content)
        # Fix trailing commas in objects: }, -> }
        content = re.sub(r',(\s*})', r'\1', content)
        # Also handle trailing commas before newlines followed by closing brackets
        content = re.sub(r',\s*\n\s*]', '\n]', content)
        content = re.sub(r',\s*\n\s*}', '\n}', content)
        
        # Try to parse JSON
        try:
            analysis = json.loads(content)
        except json.JSONDecodeError as e:
            # If parsing fails, try to fix truncated JSON
            print(f"Initial JSON parse failed: {e}")
            print(f"Response content (first 500 chars): {content[:500]}")
            
            # Try to extract valid JSON from partial response
            # Look for the last complete object/array structure
            try:
                # Try to find the last complete insights array
                if '"insights"' in content:
                    insights_match = re.search(r'"insights"\s*:\s*\[(.*?)\]', content, re.DOTALL)
                    if insights_match:
                        insights_content = insights_match.group(1)
                        # Count complete strings (those ending with ")
                        # This is a simple heuristic - if we have at least one complete insight, use it
                        complete_insights = []
                        for match in re.finditer(r'"([^"]*)"', insights_content):
                            insight_text = match.group(1)
                            if insight_text.strip():
                                complete_insights.append(insight_text)
                        
                        # Rebuild JSON with only complete insights
                        if complete_insights:
                            # Extract other fields
                            title_match = re.search(r'"title"\s*:\s*"([^"]*)"', content)
                            desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', content)
                            
                            analysis = {
                                "title": title_match.group(1) if title_match else Path(chart_path).stem.replace('_', ' ').title(),
                                "description": desc_match.group(1) if desc_match else "Chart analysis",
                                "insights": complete_insights[:2],  # Limit to 2 insights
                                "subject": None,
                                "grade": None,
                                "keyMetrics": []
                            }
                        else:
                            raise json.JSONDecodeError("No complete insights found", content, e.pos)
                    else:
                        raise e
                else:
                    raise e
            except (json.JSONDecodeError, AttributeError, IndexError) as recovery_error:
                # Try one more time: find last complete closing brace and parse up to that point
                try:
                    last_brace = content.rfind('}')
                    if last_brace > 100:  # Make sure we have substantial content
                        truncated_content = content[:last_brace + 1]
                        # Clean trailing commas again
                        truncated_content = re.sub(r',(\s*])', r'\1', truncated_content)
                        truncated_content = re.sub(r',(\s*})', r'\1', truncated_content)
                        analysis = json.loads(truncated_content)
                    else:
                        raise recovery_error
                except json.JSONDecodeError:
                    # If all else fails, create a fallback structure
                    print(f"Could not recover from JSON error, using fallback")
                    raise e
        
        # Add metadata
        analysis['chart_path'] = chart_path
        analysis['chart_name'] = Path(chart_path).stem
        
        return analysis
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response content: {content[:500]}")
        # Return fallback structure
        return {
            "title": Path(chart_path).stem.replace('_', ' ').title(),
            "description": "Chart analysis unavailable",
            "insights": [],
            "subject": None,
            "grade": None,
            "keyMetrics": [],
            "chart_path": chart_path,
            "chart_name": Path(chart_path).stem,
            "error": str(e)
        }
    except Exception as e:
        print(f"Error analyzing chart {chart_path}: {e}")
        raise


def analyze_charts_batch(
    chart_batch: List[Tuple[str, Dict]],
    api_key: Optional[str] = None,
    batch_num: int = 1
) -> List[Dict]:
    """
    Analyze a batch of charts in parallel (up to 10 charts)
    
    Args:
        chart_batch: List of tuples (chart_path, metadata_dict)
        api_key: OpenAI API key
        batch_num: Batch number for logging
    
    Returns:
        List of analysis dictionaries
    """
    analyses = []
    
    def analyze_single(chart_path: str, metadata: Dict) -> Dict:
        """Helper function to analyze a single chart"""
        try:
            return analyze_chart_with_gpt(chart_path, metadata, api_key)
        except Exception as e:
            print(f"Error analyzing chart {chart_path}: {e}")
            return {
                "title": Path(chart_path).stem.replace('_', ' ').title(),
                "description": f"Error analyzing chart: {str(e)}",
                "insights": [],
                "subject": None,
                "grade": None,
                "keyMetrics": [],
                "chart_path": chart_path,
                "chart_name": Path(chart_path).stem,
                "error": str(e)
            }
    
    # Process batch in parallel (max 10 concurrent requests)
    print(f"[Batch {batch_num}] Analyzing {len(chart_batch)} charts in parallel...")
    with ThreadPoolExecutor(max_workers=min(10, len(chart_batch))) as executor:
        # Submit all tasks
        future_to_chart = {
            executor.submit(analyze_single, chart_path, metadata): (chart_path, metadata)
            for chart_path, metadata in chart_batch
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_chart):
            chart_path, metadata = future_to_chart[future]
            try:
                analysis = future.result()
                analyses.append(analysis)
                chart_name = metadata.get('chart_name', Path(chart_path).stem)
                print(f"  ✓ Completed: {chart_name}")
            except Exception as e:
                print(f"  ✗ Failed: {chart_path} - {e}")
                analyses.append({
                    "title": Path(chart_path).stem.replace('_', ' ').title(),
                    "description": f"Error analyzing chart: {str(e)}",
                    "insights": [],
                    "subject": None,
                    "grade": None,
                    "keyMetrics": [],
                    "chart_path": chart_path,
                    "chart_name": Path(chart_path).stem,
                    "error": str(e)
                })
    
    print(f"[Batch {batch_num}] Completed {len(analyses)}/{len(chart_batch)} analyses")
    return analyses


def analyze_charts_batch_paths(
    chart_paths: List[str],
    api_key: Optional[str] = None,
    batch_size: int = 10
) -> List[Dict]:
    """
    Analyze multiple charts from a list of file paths in batches of 10
    
    Args:
        chart_paths: List of chart file paths
        api_key: OpenAI API key
        batch_size: Number of charts to process in each batch (default: 10)
    
    Returns:
        List of analysis dictionaries, one per chart
    """
    # Prepare chart list with metadata
    chart_list = []
    for chart_path in chart_paths:
        chart_path_obj = Path(chart_path)
        if not chart_path_obj.exists():
            print(f"Warning: Chart not found: {chart_path}")
            continue
        
        metadata = {
            'chart_name': chart_path_obj.stem,
            'scope': '',
            'section': ''
        }
        
        chart_list.append((str(chart_path), metadata))
    
    print(f"Total charts to analyze: {len(chart_list)}")
    
    # Process in batches
    all_analyses = []
    total_batches = (len(chart_list) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(chart_list), batch_size):
        batch = chart_list[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} charts...")
        batch_analyses = analyze_charts_batch(batch, api_key, batch_num)
        all_analyses.extend(batch_analyses)
    
    print(f"\n✅ Completed analysis of {len(all_analyses)} charts")
    return all_analyses


def analyze_charts_from_index(
    chart_index_path: str,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    max_charts: Optional[int] = None,
    batch_size: int = 10
) -> List[Dict]:
    """
    Analyze multiple charts from a chart_index.csv file in batches of 10
    
    Args:
        chart_index_path: Path to chart_index.csv file
        output_dir: Base directory for chart paths (defaults to chart_index.csv parent)
        api_key: OpenAI API key
        max_charts: Maximum number of charts to analyze (for testing/rate limiting)
        batch_size: Number of charts to process in each batch (default: 10)
    
    Returns:
        List of analysis dictionaries, one per chart
    """
    chart_index_path = Path(chart_index_path)
    if not chart_index_path.exists():
        raise FileNotFoundError(f"Chart index not found: {chart_index_path}")
    
    # Read chart index
    df = pd.read_csv(chart_index_path)
    
    # Determine base directory
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = chart_index_path.parent
    
    # Limit charts if specified
    if max_charts:
        df = df.head(max_charts)
    
    # Prepare chart list with metadata
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
    
    print(f"Total charts to analyze: {len(chart_list)}")
    
    # Process in batches
    all_analyses = []
    total_batches = (len(chart_list) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(chart_list), batch_size):
        batch = chart_list[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} charts...")
        batch_analyses = analyze_charts_batch(batch, api_key, batch_num)
        all_analyses.extend(batch_analyses)
    
    print(f"\n✅ Completed analysis of {len(all_analyses)} charts")
    return all_analyses


def save_analyses_to_json(analyses: List[Dict], output_path: str):
    """Save chart analyses to a JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(analyses, f, indent=2)
    
    print(f"Saved {len(analyses)} analyses to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze charts using OpenAI Vision API")
    parser.add_argument("--chart-index", required=True, help="Path to chart_index.csv")
    parser.add_argument("--output-dir", help="Base directory for chart paths")
    parser.add_argument("--output-json", help="Output JSON file path")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--max-charts", type=int, help="Maximum number of charts to analyze")
    
    args = parser.parse_args()
    
    analyses = analyze_charts_from_index(
        args.chart_index,
        args.output_dir,
        args.api_key,
        args.max_charts
    )
    
    if args.output_json:
        save_analyses_to_json(analyses, args.output_json)
    else:
        print(json.dumps(analyses, indent=2))

