"""
Google Slides creation module for Python backend
Ports the slide creation logic from TypeScript to Python
"""
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from ..google_slides_client import get_slides_client
from ..google_drive_upload import upload_images_to_drive_batch, extract_folder_id_from_url, create_drive_folder, move_file_to_folder
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


def find_math_reading_pair(chart_paths: List[str], start_index: int, paired_indices: set = None) -> Optional[int]:
    """
    Find the index of a chart that pairs with the chart at start_index.
    Looks for charts with same grade, scope (district/school), and section.
    If current chart is math, finds reading. If current chart is reading, finds math.
    
    Args:
        chart_paths: List of chart file paths
        start_index: Index of the current chart to find a pair for
        paired_indices: Set of indices that are already paired (to skip)
    """
    if start_index >= len(chart_paths):
        return None
    
    if paired_indices is None:
        paired_indices = set()
    
    current_chart = Path(chart_paths[start_index]).stem.lower()
    
    # Strip DISTRICT_/SCHOOL_ prefix if present for matching
    chart_without_prefix = current_chart
    is_district_chart = False
    if current_chart.startswith('district_'):
        chart_without_prefix = current_chart.replace('district_', '', 1)
        is_district_chart = True
    elif current_chart.startswith('school_'):
        chart_without_prefix = current_chart.replace('school_', '', 1)
        is_district_chart = False
    
    # Determine what we're looking for (math or reading/ela)
    # Support both NWEA (math/reading) and iReady (math/ela)
    is_math = 'math' in chart_without_prefix
    is_reading = 'reading' in chart_without_prefix or 'read' in chart_without_prefix
    is_ela = 'ela' in chart_without_prefix and not is_reading  # ELA for iReady
    
    if not (is_math or is_reading or is_ela):
        return None
    
    # Extract grade from current chart - be more precise
    grade_match = re.search(r'grade(\d+)', chart_without_prefix)
    if not grade_match:
        return None
    grade = grade_match.group(1)
    
    # Extract section
    section_match = re.search(r'section(\d+)', chart_without_prefix)
    if not section_match:
        return None
    section = section_match.group(1)
    
    # Extract school name pattern for more precise matching (without DISTRICT_/SCHOOL_ prefix)
    # For district: "parsec_academy_charter_schools"
    # For school: "parsec_academy" (without charter_schools)
    school_pattern = None
    parts_before_section = chart_without_prefix.split('_section')[0].split('_')
    
    # Check if it's a district chart by looking for "charter_schools" pattern
    if 'charter' in parts_before_section and 'schools' in parts_before_section:
        # Find where charter_schools appears
        charter_idx = parts_before_section.index('charter')
        school_pattern = '_'.join(parts_before_section[:charter_idx+2])  # Include charter_schools
    else:
        # For school charts, extract school name (everything before section)
        school_pattern = '_'.join(parts_before_section)
    
    # Look ahead for matching chart (opposite subject)
    # Also look backward if forward search fails (in case charts are out of order)
    print(f"[Pairing] Searching for pair from index {start_index + 1} to {len(chart_paths) - 1}")
    found_pair_index = None
    
    # First, try forward search (preferred - assumes math comes before reading)
    for i in range(start_index + 1, len(chart_paths)):
        # Skip if already paired
        if i in paired_indices:
            print(f"[Pairing] Skipping index {i} (already paired)")
            continue
            
        other_chart = Path(chart_paths[i]).stem.lower()
        print(f"[Pairing] Checking index {i}: {Path(chart_paths[i]).name}")
        
        # Strip DISTRICT_/SCHOOL_ prefix from other chart
        other_chart_without_prefix = other_chart
        other_is_district_chart = False
        if other_chart.startswith('district_'):
            other_chart_without_prefix = other_chart.replace('district_', '', 1)
            other_is_district_chart = True
        elif other_chart.startswith('school_'):
            other_chart_without_prefix = other_chart.replace('school_', '', 1)
            other_is_district_chart = False
        
        # CRITICAL: Must match district vs school scope
        if other_is_district_chart != is_district_chart:
            continue  # Scope mismatch - skip
        
        # Check if it's the opposite subject
        # Support both NWEA (math/reading) and iReady (math/ela)
        other_is_math = 'math' in other_chart_without_prefix
        other_is_reading = 'reading' in other_chart_without_prefix or 'read' in other_chart_without_prefix
        other_is_ela = 'ela' in other_chart_without_prefix and not other_is_reading
        
        # Pair math with reading (NWEA) or ela (iReady)
        if is_math and not (other_is_reading or other_is_ela):
            continue
        if (is_reading or is_ela) and not other_is_math:
            continue
        
        # Check if same grade - CRITICAL: must match exactly
        other_grade_match = re.search(r'grade(\d+)', other_chart_without_prefix)
        if not other_grade_match:
            continue
        other_grade = other_grade_match.group(1)
        if other_grade != grade:
            continue  # Grade mismatch - skip this chart
        
        # Check if same section
        other_section_match = re.search(r'section(\d+)', other_chart_without_prefix)
        if not other_section_match:
            continue
        other_section = other_section_match.group(1)
        if other_section != section:
            continue
        
        # Check if same scope using school pattern (without prefix)
        other_parts_before_section = other_chart_without_prefix.split('_section')[0].split('_')
        other_school_pattern = None
        if 'charter' in other_parts_before_section and 'schools' in other_parts_before_section:
            charter_idx = other_parts_before_section.index('charter')
            other_school_pattern = '_'.join(other_parts_before_section[:charter_idx+2])
        else:
            other_school_pattern = '_'.join(other_parts_before_section)
        
        if school_pattern and other_school_pattern != school_pattern:
            continue
        
        # Found a match!
        found_pair_index = i
        break
    
    # If no forward match found, try backward search (in case charts are out of order)
    if found_pair_index is None:
        print(f"[Pairing] No forward match found, searching backward from index {start_index - 1} to 0")
        for i in range(start_index - 1, -1, -1):
            # Skip if already paired
            if i in paired_indices:
                print(f"[Pairing] Skipping backward index {i} (already paired)")
                continue
                
            other_chart = Path(chart_paths[i]).stem.lower()
            print(f"[Pairing] Checking backward index {i}: {Path(chart_paths[i]).name}")
            
            # Strip DISTRICT_/SCHOOL_ prefix from other chart
            other_chart_without_prefix = other_chart
            other_is_district_chart = False
            if other_chart.startswith('district_'):
                other_chart_without_prefix = other_chart.replace('district_', '', 1)
                other_is_district_chart = True
            elif other_chart.startswith('school_'):
                other_chart_without_prefix = other_chart.replace('school_', '', 1)
                other_is_district_chart = False
            
            # CRITICAL: Must match district vs school scope
            if other_is_district_chart != is_district_chart:
                continue  # Scope mismatch - skip
            
            # Check if it's the opposite subject
            # Support both NWEA (math/reading) and iReady (math/ela)
            other_is_math = 'math' in other_chart_without_prefix
            other_is_reading = 'reading' in other_chart_without_prefix or 'read' in other_chart_without_prefix
            other_is_ela = 'ela' in other_chart_without_prefix and not other_is_reading
            
            # Pair math with reading (NWEA) or ela (iReady)
            if is_math and not (other_is_reading or other_is_ela):
                continue
            if (is_reading or is_ela) and not other_is_math:
                continue
            
            # Check if same grade - CRITICAL: must match exactly
            other_grade_match = re.search(r'grade(\d+)', other_chart_without_prefix)
            if not other_grade_match:
                continue
            other_grade = other_grade_match.group(1)
            if other_grade != grade:
                continue  # Grade mismatch - skip this chart
            
            # Check if same section
            other_section_match = re.search(r'section(\d+)', other_chart_without_prefix)
            if not other_section_match:
                continue
            other_section = other_section_match.group(1)
            if other_section != section:
                continue
            
            # Check if same scope using school pattern (without prefix)
            other_parts_before_section = other_chart_without_prefix.split('_section')[0].split('_')
            other_school_pattern = None
            if 'charter' in other_parts_before_section and 'schools' in other_parts_before_section:
                charter_idx = other_parts_before_section.index('charter')
                other_school_pattern = '_'.join(other_parts_before_section[:charter_idx+2])
            else:
                other_school_pattern = '_'.join(other_parts_before_section)
            
            if school_pattern and other_school_pattern != school_pattern:
                continue
            
            # Found a match backward!
            found_pair_index = i
            print(f"[Pairing] ✓ Found backward pair at index {i}")
            break
    
    return found_pair_index


def create_slides_presentation(
    title: str,
    chart_paths: List[str],
    drive_folder_url: Optional[str] = None,
    enable_ai_insights: bool = True,
    user_prompt: Optional[str] = None,
    deck_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to create a Google Slides presentation
    
    Args:
        title: Presentation title
        chart_paths: List of paths to chart image files
        drive_folder_url: Optional Google Drive folder URL
        enable_ai_insights: Whether to use AI-generated insights (can be overridden by decision LLM)
        user_prompt: Optional user prompt describing preferences (used by decision LLM)
        deck_type: Type of deck being created - 'BOY', 'MOY', or 'EOY' (determines which reference decks to use for training)
    
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
    original_chart_count = len(normalized_charts)
    if user_prompt and user_prompt.strip() and normalized_charts:
        print(f"[Chart Selection] Parsing user instructions for chart selection...")
        print(f"[Chart Selection] User prompt: '{user_prompt[:100]}...'")
        chart_selection_info = parse_chart_instructions(user_prompt, normalized_charts, deck_type=deck_type)
        if chart_selection_info.get('chart_selection'):
            original_count = len(normalized_charts)
            normalized_charts = chart_selection_info['chart_selection']
            filtered_count = len(normalized_charts)
            print(f"[Chart Selection] ✓ Filtered charts: {original_count} → {filtered_count} (removed {original_count - filtered_count} charts)")
            if chart_selection_info.get('reasoning'):
                print(f"[Chart Selection] Reasoning: {chart_selection_info['reasoning']}")
        else:
            print(f"[Chart Selection] ⚠ No chart_selection returned - using all {original_chart_count} charts")
    else:
        if not user_prompt or not user_prompt.strip():
            print(f"[Chart Selection] No user prompt provided - using all {original_chart_count} charts")
        else:
            print(f"[Chart Selection] No charts to filter - using all {original_chart_count} charts")
    
    # Extract parent folder ID
    parent_folder_id = extract_folder_id_from_url(drive_folder_url) if drive_folder_url else None
    
    # Import default Shared Drive folder ID
    from ..google_slides_client import DEFAULT_SLIDES_FOLDER_ID, get_drive_client
    from googleapiclient.errors import HttpError
    
    # Verify parent folder is in a Shared Drive (if provided)
    if parent_folder_id:
        try:
            drive_service = get_drive_client()
            folder_info = drive_service.files().get(
                fileId=parent_folder_id,
                fields='id, name, driveId',
                supportsAllDrives=True
            ).execute()
            
            if not folder_info.get('driveId'):
                print(f"[Slides] ⚠ Warning: Parent folder {parent_folder_id} is not in a Shared Drive")
                print(f"[Slides]   This may cause storage quota issues. Using default Shared Drive instead.")
                parent_folder_id = DEFAULT_SLIDES_FOLDER_ID
            else:
                print(f"[Slides] ✓ Parent folder is in Shared Drive (Drive ID: {folder_info.get('driveId')})")
        except HttpError as e:
            print(f"[Slides] ⚠ Warning: Could not verify parent folder: {e}")
            print(f"[Slides]   Using default Shared Drive instead.")
            parent_folder_id = DEFAULT_SLIDES_FOLDER_ID
    
    # Create a subfolder for this deck's charts inside the parent folder
    # If no parent provided, use the default Shared Drive folder
    folder_id = None
    if parent_folder_id:
        # Create a subfolder with a timestamp-based name
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subfolder_name = f"{title}_{timestamp}".replace(" ", "_")[:100]  # Limit length and sanitize
        folder_id = create_drive_folder(subfolder_name, parent_folder_id)
        if not folder_id:
            print(f"[Slides] Warning: Failed to create subfolder, uploading to parent folder instead")
            folder_id = parent_folder_id
    else:
        # No parent folder provided - use default Shared Drive and create a subfolder there
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subfolder_name = f"{title}_{timestamp}".replace(" ", "_")[:100]  # Limit length and sanitize
        print(f"[Slides] No parent folder provided, using default Shared Drive: {DEFAULT_SLIDES_FOLDER_ID}")
        folder_id = create_drive_folder(subfolder_name, DEFAULT_SLIDES_FOLDER_ID)
        if not folder_id:
            print(f"[Slides] Warning: Failed to create subfolder in Shared Drive, using Shared Drive root instead")
            folder_id = DEFAULT_SLIDES_FOLDER_ID
    
    # Create presentation via Drive API (required for service account)
    print(f"[Slides] Creating new presentation via Drive API...")
    from ..google_slides_client import create_presentation_via_drive
    presentation_id = create_presentation_via_drive(title, folder_id)
    
    # Create cover slide
    cover_slide_id = 'cover_slide_001'
    create_slide_requests = create_cover_slide_requests(cover_slide_id)
    
    # Calculate chart slides needed
    if normalized_charts:
        total_slides_needed = 0
        i = 0
        while i < len(normalized_charts):
            # Check for dual chart slide (math + reading pair)
            if is_subject_graph_pair(normalized_charts[i:i+2]) and i + 1 < len(normalized_charts):
                total_slides_needed += 1
                i += 2
            else:
                # Single chart slide
                total_slides_needed += 1
                i += 1
        
        print(f"[Slides] Creating {total_slides_needed} chart slide(s) for {len(normalized_charts)} chart(s)")
        
        for slide_idx in range(total_slides_needed):
            chart_slide_id = f'chart_slide_{slide_idx}'
            insertion_index = 1 + slide_idx  # After cover slide
            create_slide_requests.append(create_chart_slide_request(presentation_id, chart_slide_id, insertion_index))
    
    # Execute slide creation
    print(f"[Slides] Executing slide creation requests...")
    try:
        slides_service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': create_slide_requests}
        ).execute()
        print(f"[Slides] ✓ Created {len(create_slide_requests)} slide(s)")
    except HttpError as e:
        error_details = e.error_details if hasattr(e, 'error_details') else []
        print(f"[Slides] ✗ Error creating slides:")
        if error_details:
            for idx, error in enumerate(error_details, 1):
                if isinstance(error, dict):
                    print(f"[Slides]   Error {idx}: {error.get('message', 'Unknown error')} (reason: {error.get('reason', 'unknown')})")
                else:
                    # error might be a string
                    print(f"[Slides]   Error {idx}: {error}")
        else:
            print(f"[Slides]   Error: {e}")
        raise
    
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
                analysis_focus=analysis_focus,
                charts_per_api_call=8  # Analyze 8 charts per API call - optimal for single-worker environments
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
        
        # Small delay to ensure files are accessible (especially for Shared Drive files)
        import time
        print(f"[Slides] Waiting 2 seconds for files to be fully accessible...")
        time.sleep(2)
    
    # Add charts to slides
    if chart_urls:
        slide_index = 1  # Start after cover slide
        global_chart_index = 0
        
        # Verify chart_urls and normalized_charts are aligned
        if len(chart_urls) != len(normalized_charts):
            print(f"[Slides] WARNING: Mismatch - {len(chart_urls)} URLs but {len(normalized_charts)} charts")
        
        i = 0
        processed_indices = set()  # Track which charts have been processed
        while i < len(chart_urls) and i < len(normalized_charts):
            # Skip if this chart was already processed
            if i in processed_indices:
                i += 1
                continue
            
            is_subject_pair = False
            pair_index = None
            
            # No triple chart slides - only single or dual (math+reading pairs)
            current_chart_name = Path(normalized_charts[i]).stem.lower()
            is_math = 'math' in current_chart_name
            is_reading = 'reading' in current_chart_name or 'read' in current_chart_name
            is_ela = 'ela' in current_chart_name and not is_reading  # ELA for iReady
            
            # Try to find a math+reading/ela pair starting from current index
            # Only look for pairs if current chart is math, reading, or ela
            if is_math or is_reading or is_ela:
                print(f"[Pairing] Looking for pair for chart at index {i}: {Path(normalized_charts[i]).name}")
                current_subj = 'math' if is_math else ('reading' if is_reading else 'ela')
                looking_for = 'reading/ela' if is_math else 'math'
                print(f"[Pairing] Current chart is {current_subj}, looking for {looking_for}")
                pair_index = find_math_reading_pair(normalized_charts, i, processed_indices)
                if pair_index is not None:
                    print(f"[Pairing] ✓ Found pair at index {pair_index}: {Path(normalized_charts[pair_index]).name}")
                else:
                    print(f"[Pairing] ✗ No pair found for chart at index {i}")
            
            if pair_index is not None and pair_index not in processed_indices:
                # Found a math+reading/ela pair - use those two charts
                current_charts = [normalized_charts[i], normalized_charts[pair_index]]
                current_urls = [chart_urls[i], chart_urls[pair_index]]
                if current_urls[0] is None or current_urls[1] is None:
                    # Skip if URLs are None
                    i += 1
                    continue
                is_subject_pair = True
                processed_indices.add(i)
                processed_indices.add(pair_index)
            else:
                # No pair found - process as single chart slide
                # Only pair sequentially if the next chart is already a valid math+reading/ela pair
                if i + 1 < len(normalized_charts) and (i + 1) not in processed_indices:
                    next_chart_name = Path(normalized_charts[i + 1]).stem.lower()
                    next_is_math = 'math' in next_chart_name
                    next_is_reading = 'reading' in next_chart_name or 'read' in next_chart_name
                    next_is_ela = 'ela' in next_chart_name and not next_is_reading
                    # Only pair if current and next form a valid math+reading/ela pair
                    # Support both NWEA (math/reading) and iReady (math/ela)
                    if ((is_math and (next_is_reading or next_is_ela)) or 
                        ((is_reading or is_ela) and next_is_math)):
                        # Check if they're the same grade and scope
                        current_grade_match = re.search(r'grade(\d+)', current_chart_name)
                        next_grade_match = re.search(r'grade(\d+)', next_chart_name)
                        current_is_district = current_chart_name.startswith('district_')
                        next_is_district = next_chart_name.startswith('district_')
                        
                        if (current_grade_match and next_grade_match and 
                            current_grade_match.group(1) == next_grade_match.group(1) and
                            current_is_district == next_is_district):
                            # Valid sequential pair
                            current_charts = normalized_charts[i:i+2]
                            current_urls = [url for url in chart_urls[i:i+2] if url is not None]
                            is_subject_pair = len(current_urls) == 2
                            processed_indices.add(i)
                            processed_indices.add(i + 1)
                        else:
                            # Not a valid pair - process as single chart
                            current_charts = [normalized_charts[i]]
                            current_urls = [chart_urls[i]] if chart_urls[i] is not None else []
                            is_subject_pair = False
                            processed_indices.add(i)
                    else:
                        # Not a math+reading pair - process as single chart
                        current_charts = [normalized_charts[i]]
                        current_urls = [chart_urls[i]] if chart_urls[i] is not None else []
                        is_subject_pair = False
                        processed_indices.add(i)
                else:
                    # No next chart or next chart already processed - process as single chart
                    current_charts = [normalized_charts[i]]
                    current_urls = [chart_urls[i]] if chart_urls[i] is not None else []
                    is_subject_pair = False
                    processed_indices.add(i)
            
            # Log which charts we're processing
            chart_names = [Path(p).name for p in current_charts]
            if pair_index is not None:
                print(f"[Slides] Processing charts at index {i} (paired with {pair_index}): {chart_names}")
            else:
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
                    
                    # Generate a generic title for dual chart slides (both math and reading)
                    # Extract grade and school from insights or chart filenames
                    title_text = None
                    grade = None
                    school_name = None
                    
                    # Grade name mapping
                    grade_names = {
                        0: "Kindergarten", 1: "First Grade", 2: "Second Grade", 3: "Third Grade",
                        4: "Fourth Grade", 5: "Fifth Grade", 6: "Sixth Grade", 7: "Seventh Grade",
                        8: "Eighth Grade", 9: "Ninth Grade", 10: "Tenth Grade",
                        11: "Eleventh Grade", 12: "Twelfth Grade"
                    }
                    
                    # Try to extract grade and school from insights
                    for insight in [insight1, insight2]:
                        if insight:
                            # Extract grade
                            if not grade:
                                grade_str = insight.get('grade')
                                if grade_str:
                                    # Convert "1" or "Grade 1" to "First Grade"
                                    try:
                                        grade_num = int(grade_str.replace('Grade', '').replace('grade', '').strip())
                                        grade = grade_names.get(grade_num, f"Grade {grade_num}")
                                    except:
                                        grade = grade_str
                            
                            # Extract school name from title if available
                            if not school_name:
                                title = insight.get('title', '')
                                # Look for "at [School Name]" pattern
                                if ' at ' in title:
                                    school_name = title.split(' at ')[-1].strip()
                    
                    # If no grade found, try to extract from chart filename
                    if not grade:
                        chart_name = Path(current_charts[0]).name.lower()
                        # Look for "grade1", "grade_1", "grade-1" patterns
                        grade_match = re.search(r'grade[_\s-]?(\d+)', chart_name)
                        if grade_match:
                            grade_num = int(grade_match.group(1))
                            grade = grade_names.get(grade_num, f"Grade {grade_num}")
                    
                    # If no school name found, try to extract from chart filename
                    if not school_name:
                        chart_name = Path(current_charts[0]).name
                        # Look for school name patterns (usually before "section" or "grade")
                        parts = chart_name.replace('_', ' ').split()
                        # School name is usually the first part before "section" or "grade"
                        school_parts = []
                        for part in parts:
                            if part.lower() in ['section', 'grade', 'math', 'reading', 'fall', 'trends']:
                                break
                            school_parts.append(part)
                        if school_parts:
                            school_name = ' '.join(school_parts)
                    
                    # Build generic title
                    if grade and school_name:
                        title_text = f"{grade} Performance Trends at {school_name}"
                    elif grade:
                        title_text = f"{grade} Performance Trends"
                    elif school_name:
                        title_text = f"Performance Trends at {school_name}"
                    else:
                        title_text = 'Math & Reading Performance'
                    
                    print(f"[Slides] Generated dual chart title: {title_text}")
                    
                    chart_requests = create_dual_chart_slide_requests(
                        slide_object_id,
                        current_urls[0],
                        current_urls[1],
                        title_text,
                        SLIDE_WIDTH_EMU,
                        SLIDE_HEIGHT_EMU,
                        global_chart_index,
                        insight1=insight1,
                        insight2=insight2
                    )
                    print(f"[Slides] Using dual chart template for subject graphs with title: {title_text}")
                    print(f"[Slides]   Chart 1: {Path(current_charts[0]).name} -> {current_urls[0][:50]}...")
                    print(f"[Slides]   Chart 2: {Path(current_charts[1]).name} -> {current_urls[1][:50]}...")
                    # Advance index based on whether we found a pair or used sequential
                    if pair_index is not None:
                        # Skip both charts (math at i, reading at pair_index)
                        # Need to skip all charts between i and pair_index
                        i = pair_index + 1
                    elif len(current_charts) == 2:
                        # Sequential pair was used
                        i += 2
                    else:
                        # Single chart (shouldn't happen here, but just in case)
                        i += 1
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
                            # Handle both old format (strings) and new format (objects with finding/implication/recommendation)
                            insight_texts = []
                            for insight_item in insights_list:
                                finding = ''
                                implication = ''
                                recommendation = ''
                                
                                # Try to parse as dict first
                                if isinstance(insight_item, dict):
                                    finding = insight_item.get('finding', '')
                                    implication = insight_item.get('implication', '')
                                    recommendation = insight_item.get('recommendation', '')
                                elif isinstance(insight_item, str):
                                    # Check if it's a string representation of a dict
                                    if (insight_item.strip().startswith("{'") or insight_item.strip().startswith('{"')) and ('finding' in insight_item or 'implication' in insight_item):
                                        try:
                                            import ast
                                            # Try to parse as Python dict literal
                                            parsed = ast.literal_eval(insight_item)
                                            if isinstance(parsed, dict):
                                                finding = parsed.get('finding', '')
                                                implication = parsed.get('implication', '')
                                                recommendation = parsed.get('recommendation', '')
                                        except:
                                            # Fallback: try regex extraction
                                            # Note: re is already imported at module level
                                            finding_match = re.search(r"['\"]finding['\"]:\s*['\"]([^'\"]+)['\"]", insight_item)
                                            if finding_match:
                                                finding = finding_match.group(1)
                                            implication_match = re.search(r"['\"]implication['\"]:\s*['\"]([^'\"]+)['\"]", insight_item)
                                            if implication_match:
                                                implication = implication_match.group(1)
                                            recommendation_match = re.search(r"['\"]recommendation['\"]:\s*['\"]([^'\"]+)['\"]", insight_item)
                                            if recommendation_match:
                                                recommendation = recommendation_match.group(1)
                                    else:
                                        # Plain string format
                                        finding = insight_item
                                
                                # Format the insight text
                                if finding:
                                    text = f"• {finding}"
                                    if implication:
                                        text += f"\n  → {implication}"
                                    if recommendation:
                                        text += f"\n  → {recommendation}"
                                    insight_texts.append(text)
                                elif isinstance(insight_item, str):
                                    # Fallback for plain strings
                                    insight_texts.append(f'• {insight_item}')
                            
                            summary = '\n\n'.join(insight_texts)
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
                
                # Execute chart requests for this slide
                if chart_requests:
                    try:
                        # Log request details for debugging
                        print(f"[Slides] Executing batchUpdate with {len(chart_requests)} requests for slide {slide_index}")
                        if current_urls:
                            print(f"[Slides]   Image URLs: {[url[:60] + '...' if len(url) > 60 else url for url in current_urls]}")
                        
                        slides_service.presentations().batchUpdate(
                            presentationId=presentation_id,
                            body={'requests': chart_requests}
                        ).execute()
                        print(f"[Slides] ✓ Added chart elements to slide {slide_index}")
                    except HttpError as e:
                        error_details = e.error_details if hasattr(e, 'error_details') else []
                        print(f"[Slides] ✗ Error updating slide {slide_index}:")
                        if error_details:
                            for idx, error in enumerate(error_details, 1):
                                if isinstance(error, dict):
                                    print(f"[Slides]   Error {idx}: {error.get('message', 'Unknown error')} (reason: {error.get('reason', 'unknown')})")
                                else:
                                    # error might be a string
                                    print(f"[Slides]   Error {idx}: {error}")
                        else:
                            print(f"[Slides]   Error: {e}")
                        
                        # If it's a 500 error, it might be due to inaccessible images
                        if e.resp.status == 500:
                            print(f"[Slides]   ⚠️ 500 Internal Error - This often means:")
                            print(f"[Slides]     1. Image URLs are not publicly accessible")
                            print(f"[Slides]     2. Image URLs are invalid or files were deleted")
                            print(f"[Slides]     3. Request is too large or malformed")
                            if current_urls:
                                print(f"[Slides]   Verify these URLs are accessible:")
                                for url in current_urls:
                                    print(f"[Slides]     - {url}")
                        
                        raise
                
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
    
    # Count total slides (cover + chart slides)
    slide_count = len(all_slides)
    
    # Determine target folder for moving the presentation
    # Priority: subfolder > parent folder > default folder
    target_folder_id = None
    if folder_id and folder_id != parent_folder_id:
        # Subfolder was created
        target_folder_id = folder_id
        print(f"[Slides] Moving presentation to subfolder: {target_folder_id}")
    elif parent_folder_id:
        # Parent folder was provided
        target_folder_id = parent_folder_id
        print(f"[Slides] Moving presentation to parent folder: {target_folder_id}")
    else:
        # No folder specified, use default folder
        from ..google_slides_client import DEFAULT_SLIDES_FOLDER_ID
        target_folder_id = DEFAULT_SLIDES_FOLDER_ID
        print(f"[Slides] No folder specified, moving to default folder: {target_folder_id}")
    
    # Move the presentation to the target folder
    if target_folder_id:
        from ..google_slides_client import get_drive_client
        try:
            drive_service = get_drive_client()
            # Get current parents of the presentation
            file = drive_service.files().get(
                fileId=presentation_id,
                fields='parents',
                supportsAllDrives=True
            ).execute()
            previous_parents = ",".join(file.get('parents', []))
            
            # Move the presentation to the target folder
            drive_service.files().update(
                fileId=presentation_id,
                addParents=target_folder_id,
                removeParents=previous_parents,
                fields='id, parents',
                supportsAllDrives=True
            ).execute()
            print(f"[Slides] ✓ Moved presentation to folder")
        except Exception as e:
            print(f"[Slides] Warning: Could not move presentation to folder: {e}")
    
    return {
        'success': True,
        'presentationId': presentation_id,
        'presentationUrl': presentation_url,
        'title': title,
        'slideCount': slide_count
    }

