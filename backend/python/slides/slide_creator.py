"""
Google Slides creation module for Python backend
Ports the slide creation logic from TypeScript to Python
"""
from pathlib import Path
from typing import List, Dict, Optional, Any
from ..google_slides_client import get_slides_client
from ..google_drive_upload import upload_images_to_drive_batch, extract_folder_id_from_url
from ..chart_analyzer import analyze_charts_batch_paths
from ..decision_llm import should_use_ai_insights, parse_chart_instructions
from .slide_constants import SLIDE_WIDTH_EMU, SLIDE_HEIGHT_EMU
from .cover_slide import create_cover_slide_requests
from .chart_slides import (
    create_chart_slide_request,
    create_single_chart_slide_requests,
    create_dual_chart_slide_requests
)


def is_subject_graph_pair(chart_paths: List[str]) -> bool:
    """Check if charts are a math+reading pair"""
    if len(chart_paths) < 2:
        return False
    
    chart_names = [Path(p).stem.lower() for p in chart_paths]
    has_math = any('math' in name for name in chart_names)
    has_reading = any('reading' in name or 'read' in name for name in chart_names)
    return has_math and has_reading


def create_slides_presentation(
    title: str,
    chart_paths: List[str],
    drive_folder_url: Optional[str] = None,
    enable_ai_insights: bool = True,
    user_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to create a Google Slides presentation
    
    Args:
        title: Presentation title
        chart_paths: List of paths to chart image files
        drive_folder_url: Optional Google Drive folder URL
        enable_ai_insights: Whether to use AI-generated insights (can be overridden by decision LLM)
        user_prompt: Optional user prompt describing preferences (used by decision LLM)
    
    Returns:
        Dict with presentationId, presentationUrl, title, decision_info
    """
    print(f"[Slides] Creating presentation: {title}")
    
    # Get clients
    slides_service = get_slides_client()
    
    # Normalize chart paths to absolute
    normalized_charts = []
    for chart_path in chart_paths:
        if isinstance(chart_path, str):
            resolved_path = Path(chart_path).resolve() if Path(chart_path).is_absolute() else Path.cwd() / chart_path
            normalized_charts.append(str(resolved_path))
    
    # Parse user instructions for chart selection and ordering
    chart_selection_info = None
    if user_prompt and user_prompt.strip() and normalized_charts:
        print(f"[Chart Selection] Parsing user instructions for chart selection...")
        chart_selection_info = parse_chart_instructions(user_prompt, normalized_charts)
        if chart_selection_info.get('chart_selection'):
            original_count = len(normalized_charts)
            normalized_charts = chart_selection_info['chart_selection']
            print(f"[Chart Selection] Filtered charts: {original_count} → {len(normalized_charts)}")
            if chart_selection_info.get('reasoning'):
                print(f"[Chart Selection] {chart_selection_info['reasoning']}")
    
    # Extract folder ID
    folder_id = extract_folder_id_from_url(drive_folder_url) if drive_folder_url else None
    
    # Create presentation
    print(f"[Slides] Creating new presentation...")
    presentation = slides_service.presentations().create(
        body={'title': title}
    ).execute()
    
    presentation_id = presentation.get('presentationId')
    if not presentation_id:
        raise Exception('Failed to create presentation: No presentation ID returned')
    
    print(f"[Slides] Created presentation: {presentation_id}")
    
    # Create cover slide
    cover_slide_id = 'cover_slide_001'
    create_slide_requests = create_cover_slide_requests(cover_slide_id)
    
    # Calculate chart slides needed
    if normalized_charts:
        total_slides_needed = 0
        i = 0
        while i < len(normalized_charts):
            current_charts = normalized_charts[i:i+2]
            if is_subject_graph_pair(current_charts) and i + 1 < len(normalized_charts):
                total_slides_needed += 1
                i += 2
            else:
                total_slides_needed += 1
                i += 1
        
        print(f"[Slides] Creating {total_slides_needed} chart slide(s) for {len(normalized_charts)} chart(s)")
        
        for slide_idx in range(total_slides_needed):
            chart_slide_id = f'chart_slide_{slide_idx}'
            insertion_index = 1 + slide_idx  # After cover slide
            create_slide_requests.append(create_chart_slide_request(presentation_id, chart_slide_id, insertion_index))
    
    # Execute slide creation
    print(f"[Slides] Executing slide creation requests...")
    slides_service.presentations().batchUpdate(
        presentationId=presentation_id,
        body={'requests': create_slide_requests}
    ).execute()
    
    # Get updated presentation
    updated_presentation = slides_service.presentations().get(presentationId=presentation_id).execute()
    all_slides = updated_presentation.get('slides', [])
    
    # Decision LLM: Determine if we should use AI insights
    decision_info = None
    use_ai_insights = enable_ai_insights
    
    if user_prompt and user_prompt.strip():
        print(f"[Decision LLM] Evaluating user prompt: '{user_prompt[:100]}...'")
        decision_info = should_use_ai_insights(
            user_prompt=user_prompt,
            chart_count=len(normalized_charts) if normalized_charts else 0,
            default_enable=enable_ai_insights
        )
        use_ai_insights = decision_info['use_ai']
        print(f"[Decision LLM] Decision: use_ai={use_ai_insights}, confidence={decision_info['confidence']:.2f}")
        print(f"[Decision LLM] Reasoning: {decision_info['reasoning']}")
        if decision_info.get('analysis_focus'):
            print(f"[Decision LLM] Analysis focus: {decision_info['analysis_focus']}")
    else:
        print(f"[Decision LLM] No user prompt provided, using enable_ai_insights={enable_ai_insights}")
    
    # Analyze charts with AI if decision LLM says yes
    chart_insights_map = {}
    if normalized_charts and use_ai_insights:
        try:
            print(f"[AI] Analyzing {len(normalized_charts)} charts...")
            # Pass analysis focus to chart analyzer if available
            analysis_focus = decision_info.get('analysis_focus') if decision_info else None
            analyses = analyze_charts_batch_paths(
                normalized_charts, 
                batch_size=10,
                analysis_focus=analysis_focus
            )
            for insight in analyses:
                if 'chart_path' in insight:
                    chart_insights_map[insight['chart_path']] = insight
            print(f"[AI] Successfully analyzed {len(chart_insights_map)} charts")
        except Exception as e:
            print(f"[AI] Error analyzing charts: {e}, continuing without AI insights")
    elif normalized_charts and not use_ai_insights:
        print(f"[AI] Skipping chart analysis (decision LLM determined AI not needed)")
    
    # Upload charts to Drive
    chart_urls = []
    if normalized_charts:
        print(f"[Drive] Uploading {len(normalized_charts)} charts to Google Drive...")
        images_to_upload = [
            {'imagePath': chart_path, 'fileName': Path(chart_path).name}
            for chart_path in normalized_charts
        ]
        chart_urls = upload_images_to_drive_batch(images_to_upload, folder_id, batch_size=10, make_public=True)
        print(f"[Drive] Uploaded {sum(1 for url in chart_urls if url)}/{len(normalized_charts)} charts")
    
    # Add charts to slides
    if chart_urls:
        slide_index = 1  # Start after cover slide
        global_chart_index = 0
        
        # Verify chart_urls and normalized_charts are aligned
        if len(chart_urls) != len(normalized_charts):
            print(f"[Slides] WARNING: Mismatch - {len(chart_urls)} URLs but {len(normalized_charts)} charts")
        
        i = 0
        while i < len(chart_urls) and i < len(normalized_charts):
            current_charts = normalized_charts[i:i+2]
            current_urls = [url for url in chart_urls[i:i+2] if url is not None]  # Filter out None URLs
            is_subject_pair = is_subject_graph_pair(current_charts) and len(current_urls) == 2
            
            # Log which charts we're processing
            chart_names = [Path(p).name for p in current_charts]
            print(f"[Slides] Processing charts at index {i}: {chart_names}")
            
            if slide_index < len(all_slides):
                chart_slide = all_slides[slide_index]
                slide_object_id = chart_slide.get('objectId')
                
                if not slide_object_id:
                    print(f"[Slides] Warning: Slide at index {slide_index} has no objectId")
                    slide_index += 1
                    continue
                
                chart_requests = []
                
                if is_subject_pair and len(current_urls) >= 2:
                    # Dual chart slide
                    title_text = None
                    
                    # Get AI insights for both charts
                    chart_path_for_lookup1 = current_charts[0]
                    chart_path_for_lookup2 = current_charts[1]
                    chart_path_resolved1 = str(Path(chart_path_for_lookup1).resolve())
                    chart_path_resolved2 = str(Path(chart_path_for_lookup2).resolve())
                    
                    # Try multiple lookup strategies for first chart
                    insight1 = None
                    insight1 = chart_insights_map.get(chart_path_for_lookup1)
                    if not insight1:
                        insight1 = chart_insights_map.get(chart_path_resolved1)
                    if not insight1:
                        chart_filename1 = Path(chart_path_for_lookup1).name
                        for path, insight_data in chart_insights_map.items():
                            if Path(path).name == chart_filename1:
                                insight1 = insight_data
                                break
                    
                    # Try multiple lookup strategies for second chart
                    insight2 = None
                    insight2 = chart_insights_map.get(chart_path_for_lookup2)
                    if not insight2:
                        insight2 = chart_insights_map.get(chart_path_resolved2)
                    if not insight2:
                        chart_filename2 = Path(chart_path_for_lookup2).name
                        for path, insight_data in chart_insights_map.items():
                            if Path(path).name == chart_filename2:
                                insight2 = insight_data
                                break
                    
                    # Use title from first chart's insight, or second chart's, or fallback
                    if insight1:
                        title_text = insight1.get('title')
                    elif insight2:
                        title_text = insight2.get('title')
                    
                    if not title_text:
                        title_text = 'Math & Reading Performance'
                        print(f"[Slides] No AI insights found for dual chart, using default title")
                    else:
                        print(f"[AI] Using AI-generated title for dual chart: {title_text}")
                    
                    chart_requests = create_dual_chart_slide_requests(
                        slide_object_id,
                        current_urls[0],
                        current_urls[1],
                        title_text,
                        SLIDE_WIDTH_EMU,
                        SLIDE_HEIGHT_EMU,
                        global_chart_index
                    )
                    print(f"[Slides] Using dual chart template for subject graphs with title: {title_text}")
                    print(f"[Slides]   Chart 1: {Path(current_charts[0]).name} -> {current_urls[0][:50]}...")
                    print(f"[Slides]   Chart 2: {Path(current_charts[1]).name} -> {current_urls[1][:50]}...")
                    i += 2
                else:
                    # Single chart slide
                    if not current_urls or len(current_urls) == 0:
                        print(f"[Slides] Warning: No valid URLs for chart at index {i}, skipping")
                        i += 1
                        continue
                    
                    title_text = None
                    summary = None
                    
                    # Get AI insights for title
                    # Normalize path for lookup (handle both absolute and relative paths)
                    chart_path_for_lookup = current_charts[0]
                    chart_path_resolved = str(Path(chart_path_for_lookup).resolve())
                    
                    # Try multiple lookup strategies
                    insight = None
                    # Strategy 1: Exact match with original path
                    insight = chart_insights_map.get(chart_path_for_lookup)
                    # Strategy 2: Exact match with resolved path
                    if not insight:
                        insight = chart_insights_map.get(chart_path_resolved)
                    # Strategy 3: Match by filename
                    if not insight:
                        chart_filename = Path(chart_path_for_lookup).name
                        for path, insight_data in chart_insights_map.items():
                            if Path(path).name == chart_filename:
                                insight = insight_data
                                print(f"[Slides] Found insight by filename match: {path} -> {chart_filename}")
                                break
                    
                    if insight:
                        title_text = insight.get('title')
                        insights_list = insight.get('insights', [])
                        if insights_list:
                            summary = '\n\n'.join(f'• {i}' for i in insights_list)
                        elif insight.get('description'):
                            summary = insight['description']
                        print(f"[AI] Using AI-generated title: {title_text}")
                    else:
                        # Fallback if no AI insights available
                        title_text = 'Chart Analysis'
                        print(f"[Slides] No AI insights found for: {Path(chart_path_for_lookup).name}, using default title")
                        print(f"[Slides] Available insights keys: {[Path(k).name for k in list(chart_insights_map.keys())[:5]]}")
                    
                    chart_requests = create_single_chart_slide_requests(
                        slide_object_id,
                        current_urls[0],
                        title_text,
                        SLIDE_WIDTH_EMU,
                        SLIDE_HEIGHT_EMU,
                        global_chart_index,
                        summary
                    )
                    print(f"[Slides] Using single chart template with title: {title_text}")
                    print(f"[Slides]   Chart: {Path(current_charts[0]).name} -> {current_urls[0][:50]}...")
                    i += 1
                
                if chart_requests:
                    slides_service.presentations().batchUpdate(
                        presentationId=presentation_id,
                        body={'requests': chart_requests}
                    ).execute()
                    print(f"[Slides] ✓ Added chart elements to slide {slide_index}")
                
                global_chart_index += len(current_urls)
                slide_index += 1
            else:
                print(f"[Slides] Warning: Ran out of slides at index {slide_index}")
                break
    
    presentation_url = f'https://docs.google.com/presentation/d/{presentation_id}/edit'
    
    print(f"[Slides] ✓ Presentation created successfully: {presentation_url}")
    
    # Clean up chart files after successful upload to Drive
    if normalized_charts:
        print(f"[Slides] Cleaning up {len(normalized_charts)} chart files...")
        cleaned_count = 0
        for chart_path in normalized_charts:
            try:
                chart_file = Path(chart_path)
                if chart_file.exists():
                    chart_file.unlink()
                    cleaned_count += 1
                    # Also try to clean up parent directories if empty
                    parent = chart_file.parent
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
            except Exception as e:
                print(f"[Slides] Warning: Could not delete {chart_path}: {e}")
        print(f"[Slides] Cleaned up {cleaned_count}/{len(normalized_charts)} chart files")
    
    return {
        'success': True,
        'presentationId': presentation_id,
        'presentationUrl': presentation_url,
        'title': title
    }

