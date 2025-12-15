"""
Simple test script to verify speaker notes functionality
Usage: python test_speaker_notes.py <presentation_id> <slide_object_id>
   OR: python test_speaker_notes.py  (will test formatting only)
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.google.google_slides_client import get_slides_client
from python.slides.slide_creator import create_speaker_notes_requests, format_analysis_for_speaker_notes

def test_formatting_only():
    """Test just the formatting function"""
    print("=" * 60)
    print("Testing Analysis Formatting Function")
    print("=" * 60)
    
    test_insight = {
        'title': 'Test Chart Analysis',
        'description': 'This is a test description of the chart data.',
        'groundTruths': [
            'Math scores increased by 15% from Fall to Winter',
            'Reading scores remained stable at 75% proficiency'
        ],
        'insights': [
            {
                'finding': 'Math performance shows significant improvement',
                'implication': 'The math intervention program appears to be effective',
                'recommendation': 'Continue the current math intervention strategies'
            }
        ],
        'hypotheses': [
            'If current trends continue, math scores will reach 80% proficiency by Spring'
        ],
        'opportunities': {
            'classroom': 'Teachers can use math success strategies in reading instruction',
            'school': 'Consider expanding successful math interventions to other subjects'
        }
    }
    
    analysis_text = format_analysis_for_speaker_notes(test_insight)
    print(f"\n[Test] ✓ Formatted analysis text ({len(analysis_text)} characters)")
    print("\n" + "=" * 60)
    print("FORMATTED ANALYSIS TEXT:")
    print("=" * 60)
    print(analysis_text)
    print("=" * 60)
    
    # Test empty insight
    empty_text = format_analysis_for_speaker_notes(None)
    print(f"\n[Test] Empty insight test: '{empty_text}' (should be empty)")
    assert empty_text == "", "Empty insight should return empty string"
    
    print("\n[Test] ✓ Formatting tests passed!")
    return True

def test_speaker_notes(presentation_id: str, slide_object_id: str):
    """Test adding speaker notes to an existing slide"""
    print("=" * 60)
    print("Testing Speaker Notes Functionality")
    print("=" * 60)
    print(f"[Test] Presentation ID: {presentation_id}")
    print(f"[Test] Slide Object ID: {slide_object_id}")
    
    slides_service = get_slides_client()
    
    # Get the slide's notesPage
    print("\n[Test] Retrieving slide's notesPage...")
    import time
    time.sleep(0.5)  # Brief delay
    
    presentation = slides_service.presentations().get(
        presentationId=presentation_id
    ).execute()
    
    slides_list = presentation.get('slides', [])
    notes_page_id = None
    
    for slide in slides_list:
        if slide.get('objectId') == slide_object_id:
            notes_page = slide.get('notesPage')
            print(f"[Test] Slide notesPage object: {notes_page}")
            if notes_page:
                notes_page_id = notes_page.get('objectId')
                print(f"[Test] ✓ Found notesPageId: {notes_page_id}")
            else:
                print(f"[Test] ✗ Slide has no notesPage object")
                print(f"[Test] Full slide object keys: {list(slide.keys())}")
            break
    
    if not notes_page_id:
        print(f"[Test] ✗ ERROR: Could not find notesPageId")
        print(f"[Test] Available slides: {[s.get('objectId') for s in slides_list[:5]]}")
        return False
    
    # Test formatting analysis
    print("\n[Test] Testing analysis formatting...")
    test_insight = {
        'title': 'Test Chart Analysis',
        'description': 'This is a test description of the chart data.',
        'groundTruths': [
            'Math scores increased by 15% from Fall to Winter',
            'Reading scores remained stable at 75% proficiency'
        ],
        'insights': [
            {
                'finding': 'Math performance shows significant improvement',
                'implication': 'The math intervention program appears to be effective',
                'recommendation': 'Continue the current math intervention strategies'
            }
        ],
        'hypotheses': [
            'If current trends continue, math scores will reach 80% proficiency by Spring'
        ],
        'opportunities': {
            'classroom': 'Teachers can use math success strategies in reading instruction',
            'school': 'Consider expanding successful math interventions to other subjects'
        }
    }
    
    analysis_text = format_analysis_for_speaker_notes(test_insight)
    print(f"[Test] ✓ Formatted analysis text ({len(analysis_text)} characters)")
    
    # Create speaker notes requests
    print("\n[Test] Creating speaker notes requests...")
    notes_requests = create_speaker_notes_requests(notes_page_id, analysis_text)
    print(f"[Test] ✓ Created {len(notes_requests)} requests")
    
    if not notes_requests:
        print(f"[Test] ✗ ERROR: No requests created")
        return False
    
    # Execute speaker notes requests
    print("\n[Test] Executing speaker notes requests...")
    try:
        slides_service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': notes_requests}
        ).execute()
        print(f"[Test] ✓ Successfully added speaker notes!")
        print(f"\n[Test] ✓ TEST PASSED!")
        print(f"[Test] Check the presentation to verify speaker notes:")
        print(f"[Test] https://docs.google.com/presentation/d/{presentation_id}/edit")
        print(f"[Test] Open the presentation and check the speaker notes panel (View > Show speaker notes)")
        return True
    except Exception as e:
        print(f"[Test] ✗ ERROR: Failed to add speaker notes: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        # Test with existing presentation
        presentation_id = sys.argv[1]
        slide_object_id = sys.argv[2]
        success = test_speaker_notes(presentation_id, slide_object_id)
    else:
        # Test formatting only
        print("Usage: python test_speaker_notes.py <presentation_id> <slide_object_id>")
        print("   OR: python test_speaker_notes.py  (tests formatting only)")
        print("\nRunning formatting test only...\n")
        success = test_formatting_only()
    
    sys.exit(0 if success else 1)

