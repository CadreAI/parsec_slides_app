"""
Layout learning module for extracting and learning slide layout patterns from reference decks
"""
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Try alternative PDF libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pymupdf  # PyMuPDF / fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract text content from PDF file using multiple methods
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text content or None if extraction fails
    """
    text = None
    
    # Try pdfplumber first (better for complex layouts)
    if PDFPLUMBER_AVAILABLE:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"\n--- Page {i+1} ---\n")
                        text_parts.append(page_text)
                text = "\n".join(text_parts) if text_parts else None
                if text and len(text.strip()) > 50:
                    return text
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
    
    # Try PyMuPDF (fitz) - good for image-based PDFs
    if PYMUPDF_AVAILABLE and (not text or len(text.strip()) < 50):
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            text_parts = []
            for i, page in enumerate(doc):
                page_text = page.get_text()
                if page_text:
                    text_parts.append(f"\n--- Page {i+1} ---\n")
                    text_parts.append(page_text)
            doc.close()
            text = "\n".join(text_parts) if text_parts else None
            if text and len(text.strip()) > 50:
                return text
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
    
    # Fallback to PyPDF2
    if PyPDF2 is None:
        print("Warning: No PDF libraries available. Install with: pip install PyPDF2 pdfplumber pymupdf")
        return None
    
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"\n--- Page {i+1} ---\n")
                    text_parts.append(page_text)
            text = "\n".join(text_parts) if text_parts else None
            
            # If very little text extracted, PDF might be image-based
            if not text or len(text.strip()) < 50:
                print(f"Warning: Only extracted {len(text.strip()) if text else 0} characters from {Path(pdf_path).name}.")
                print(f"PDF may be image-based. Consider using OCR or manual annotation.")
                if text:
                    print(f"Sample text: {text[:200]}")
            
            return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return None


def parse_chart_references_from_text(text: str) -> List[Dict[str, str]]:
    """
    Parse chart references from PDF text
    Looks for patterns like:
    - "Section 3: Grade 1 Math"
    - "Section 1: Fall Trends"
    - Chart filenames or titles
    - Slide titles and headings
    
    Args:
        text: Extracted PDF text
    
    Returns:
        List of chart references with section, grade, subject, scope info
    """
    references = []
    
    if not text or len(text.strip()) < 50:
        # Text too short, might be image-based PDF
        return references
    
    # Pattern 1: Section X: [Description] or Section X [Description]
    section_patterns = [
        r'Section\s*(\d+)[:\s]+([^\n]+)',
        r'Section\s*(\d+)\s+([^\n]+)',
        r'Sec\s*(\d+)[:\s]+([^\n]+)',
    ]
    
    for pattern in section_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            section_num = match.group(1)
            description = match.group(2).strip()
            
            # Extract grade, subject, scope from description
            grade_match = re.search(r'Grade\s*(\d+)', description, re.IGNORECASE)
            grade = grade_match.group(1) if grade_match else None
            
            subject = None
            if re.search(r'\bmath\b|\bmathematics\b', description, re.IGNORECASE):
                subject = 'math'
            elif re.search(r'\breading\b|\bela\b|\benglish\b', description, re.IGNORECASE):
                subject = 'reading'
            
            scope = None
            if re.search(r'\bdistrict\b', description, re.IGNORECASE):
                scope = 'district'
            elif re.search(r'\bschool\b', description, re.IGNORECASE):
                scope = 'school'
            
            references.append({
                'section': f'section{section_num}',
                'grade': grade,
                'subject': subject,
                'scope': scope,
                'description': description
            })
    
    # Pattern 2: Chart filenames in text (e.g., "district_section3_grade1_math_fall_trends.png")
    filename_patterns = [
        r'([a-z_]+_section\d+[_\w]*\.png)',
        r'([a-z_]+_section\d+[_\w]*\.jpg)',
        r'(section\d+[_\w]*\.png)',
    ]
    
    for pattern in filename_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            filename = match.group(1).lower()
            ref = parse_chart_filename(filename)
            if ref:
                references.append(ref)
    
    # Pattern 3: Infer sections from slide titles and content structure
    # Look for assessment names (NWEA, STAR, iReady) and chart types
    lines = text.split('\n')
    current_section = None
    page_num = 0
    
    # Map chart types to sections based on common patterns
    chart_type_to_section = {
        'fall trends': 'section1',
        'winter trends': 'section1', 
        'spring trends': 'section1',
        'year to year': 'section1',
        'fall to fall': 'section1',
        'growth': 'section4',
        'cohort': 'section6',
        'grade': 'section3',
        'student group': 'section2',
        'demographic': 'section2',
        'prediction': 'section0',
        'predicted': 'section0',
        'caaspp': 'section0',
    }
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        
        # Track page numbers
        if '--- Page' in line_stripped:
            page_match = re.search(r'Page\s+(\d+)', line_stripped)
            if page_match:
                page_num = int(page_match.group(1))
        
        # Look for explicit section indicators
        section_match = re.search(r'section\s*(\d+)', line_lower)
        if section_match:
            current_section = f'section{section_match.group(1)}'
            continue
        
        # Infer section from chart type keywords and create reference
        for keyword, section in chart_type_to_section.items():
            if keyword in line_lower:
                current_section = section
                # Create a reference for this chart type
                # Check context for subject
                context_lines = lines[max(0, i-2):min(len(lines), i+3)]
                context_text = ' '.join(context_lines).lower()
                
                subject = None
                if 'math' in context_text or 'mathematics' in context_text:
                    subject = 'math'
                elif 'reading' in context_text or 'ela' in context_text or 'english' in context_text:
                    subject = 'reading'
                
                # Check for grade in context
                grade_match_context = re.search(r'grade\s*(\d+)', context_text)
                grade = grade_match_context.group(1) if grade_match_context else None
                
                references.append({
                    'section': section,
                    'grade': grade,
                    'subject': subject,
                    'scope': None,
                    'description': line_stripped[:100],
                    'page': page_num
                })
                break
        
        # Look for grade indicators (often indicates section3)
        grade_match = re.search(r'grade\s*(\d+)', line_lower)
        if grade_match:
            grade = grade_match.group(1)
            # If we see grade without a section, assume section3
            if not current_section:
                current_section = 'section3'
            
            # Check context for subject
            subject = None
            context_lines = lines[max(0, i-3):min(len(lines), i+5)]
            context_text = ' '.join(context_lines).lower()
            
            if 'math' in context_text or 'mathematics' in context_text:
                subject = 'math'
            elif 'reading' in context_text or 'ela' in context_text or 'english' in context_text:
                subject = 'reading'
            
            # Check for scope
            scope = None
            if 'district' in context_text:
                scope = 'district'
            elif 'school' in context_text:
                scope = 'school'
            
            references.append({
                'section': current_section or 'section3',
                'grade': grade,
                'subject': subject,
                'scope': scope,
                'description': line_stripped[:100],
                'page': page_num
            })
        
        # Look for subject mentions without grade (might be section1 or section2)
        if ('math' in line_lower or 'reading' in line_lower or 'ela' in line_lower) and not grade_match:
            subject = None
            if 'math' in line_lower or 'mathematics' in line_lower:
                subject = 'math'
            elif 'reading' in line_lower or 'ela' in line_lower:
                subject = 'reading'
            
            # If we have a subject but no section yet, infer from context
            if subject and not current_section:
                if 'trend' in line_lower or 'year' in line_lower:
                    current_section = 'section1'
                elif 'group' in line_lower or 'demographic' in line_lower:
                    current_section = 'section2'
                else:
                    current_section = 'section1'  # Default
            
            if subject and current_section:
                references.append({
                    'section': current_section,
                    'grade': None,
                    'subject': subject,
                    'scope': None,
                    'description': line_stripped[:100],
                    'page': page_num
                })
    
    # Pattern 4: Look for common assessment chart patterns in text
    # NWEA, STAR, iReady patterns
    assessment_patterns = [
        (r'nwea.*section\s*(\d+)', 'nwea'),
        (r'star.*section\s*(\d+)', 'star'),
        (r'iready.*section\s*(\d+)', 'iready'),
    ]
    
    for pattern, assessment in assessment_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            section_num = match.group(1)
            # Extract context around the match
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            
            grade_match = re.search(r'grade\s*(\d+)', context, re.IGNORECASE)
            grade = grade_match.group(1) if grade_match else None
            
            subject = None
            if re.search(r'\bmath\b', context, re.IGNORECASE):
                subject = 'math'
            elif re.search(r'\breading\b|\bela\b', context, re.IGNORECASE):
                subject = 'reading'
            
            references.append({
                'section': f'section{section_num}',
                'grade': grade,
                'subject': subject,
                'scope': None,
                'description': context[:50]
            })
    
    # Remove duplicates
    seen = set()
    unique_references = []
    for ref in references:
        key = (ref.get('section'), ref.get('grade'), ref.get('subject'), ref.get('scope'))
        if key not in seen:
            seen.add(key)
            unique_references.append(ref)
    
    return unique_references


def parse_chart_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse chart filename to extract metadata
    
    Args:
        filename: Chart filename (e.g., "district_section3_grade1_math_fall_trends.png")
    
    Returns:
        Dict with section, grade, subject, scope, window info
    """
    filename_lower = filename.lower().replace('.png', '').replace('.jpg', '')
    
    # Extract scope
    scope = None
    if filename_lower.startswith('district_'):
        scope = 'district'
        filename_lower = filename_lower.replace('district_', '', 1)
    elif filename_lower.startswith('school_'):
        scope = 'school'
        filename_lower = filename_lower.replace('school_', '', 1)
    
    # Extract section
    section_match = re.search(r'section(\d+)', filename_lower)
    section = f'section{section_match.group(1)}' if section_match else None
    
    # Extract grade
    grade_match = re.search(r'grade[_\s-]?(\d+)', filename_lower)
    grade = grade_match.group(1) if grade_match else None
    
    # Extract subject
    subject = None
    if 'math' in filename_lower:
        subject = 'math'
    elif 'reading' in filename_lower or 'read' in filename_lower:
        subject = 'reading'
    elif 'ela' in filename_lower:
        subject = 'reading'  # Treat ELA as reading
    
    # Extract window
    window = None
    if 'fall' in filename_lower or 'boy' in filename_lower:
        window = 'fall'
    elif 'winter' in filename_lower or 'moy' in filename_lower:
        window = 'winter'
    elif 'spring' in filename_lower or 'eoy' in filename_lower:
        window = 'spring'
    
    if section:  # Only return if we found at least a section
        return {
            'section': section,
            'grade': grade,
            'subject': subject,
            'scope': scope,
            'window': window,
            'filename': filename
        }
    
    return None


def analyze_reference_deck_layout(pdf_path: str) -> Dict[str, any]:
    """
    Analyze a reference deck PDF to extract layout patterns
    
    Args:
        pdf_path: Path to reference deck PDF
    
    Returns:
        Dict with:
            - slide_order: List of chart references in order
            - section_sequence: List of sections in order
            - scope_pattern: Pattern of scope ordering (district first, school first, mixed)
            - subject_pairing: Whether math/reading are paired together
    """
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return {
            'slide_order': [],
            'section_sequence': [],
            'scope_pattern': None,
            'subject_pairing': False,
            'error': 'Could not extract text from PDF',
            'text_length': 0
        }
    
    references = parse_chart_references_from_text(text)
    
    if not references:
        # Debug: show what text we got
        text_sample = text[:500] if text else "No text"
        print(f"  Debug: Extracted {len(text)} chars, sample: {text_sample[:200]}...")
        return {
            'slide_order': [],
            'section_sequence': [],
            'scope_pattern': None,
            'subject_pairing': False,
            'error': 'No chart references found in PDF',
            'text_length': len(text),
            'text_sample': text[:500]
        }
    
    # Extract section sequence
    section_sequence = []
    seen_sections = set()
    for ref in references:
        section = ref.get('section')
        if section and section not in seen_sections:
            section_sequence.append(section)
            seen_sections.add(section)
    
    # Analyze scope pattern
    scopes = [ref.get('scope') for ref in references if ref.get('scope')]
    scope_pattern = None
    if scopes:
        if all(s == 'district' for s in scopes[:len(scopes)//2]):
            scope_pattern = 'district_first'
        elif all(s == 'school' for s in scopes[:len(scopes)//2]):
            scope_pattern = 'school_first'
        else:
            scope_pattern = 'mixed'
    
    # Check for subject pairing (math/reading pairs)
    subject_pairing = False
    for i in range(len(references) - 1):
        curr = references[i]
        next_ref = references[i + 1]
        if (curr.get('section') == next_ref.get('section') and
            curr.get('grade') == next_ref.get('grade') and
            curr.get('scope') == next_ref.get('scope') and
            curr.get('subject') != next_ref.get('subject') and
            curr.get('subject') and next_ref.get('subject')):
            subject_pairing = True
            break
    
    return {
        'slide_order': references,
        'section_sequence': section_sequence,
        'scope_pattern': scope_pattern,
        'subject_pairing': subject_pairing,
        'total_slides': len(references)
    }


def analyze_deck_structure(pdf_path: str) -> Dict[str, any]:
    """
    Analyze overall deck structure - what charts appear, in what order, 
    and how they're organized
    
    Args:
        pdf_path: Path to reference deck PDF
    
    Returns:
        Dict with deck structure analysis:
            - chart_types: List of chart types found (by keywords/descriptions)
            - presentation_flow: Sequence of chart appearances
            - chart_groupings: How charts are grouped together
            - optional_charts: Charts that appear inconsistently
            - required_charts: Charts that appear in most/all decks
    """
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return {
            'chart_types': [],
            'presentation_flow': [],
            'chart_groupings': [],
            'optional_charts': [],
            'required_charts': []
        }
    
    references = parse_chart_references_from_text(text)
    
    # Analyze chart types and flow
    chart_types = []
    presentation_flow = []
    
    for ref in references:
        section = ref.get('section', 'unknown')
        grade = ref.get('grade')
        subject = ref.get('subject')
        page = ref.get('page', 0)
        desc = ref.get('description', '')
        
        # Build chart type identifier
        chart_type_parts = [section]
        if grade:
            chart_type_parts.append(f'grade{grade}')
        if subject:
            chart_type_parts.append(subject)
        
        chart_type = '_'.join(chart_type_parts)
        chart_types.append(chart_type)
        
        presentation_flow.append({
            'page': page,
            'section': section,
            'grade': grade,
            'subject': subject,
            'chart_type': chart_type,
            'description': desc[:50]
        })
    
    # Sort by page number
    presentation_flow.sort(key=lambda x: x.get('page', 999))
    
    # Analyze groupings (charts that appear close together)
    chart_groupings = []
    for i in range(len(presentation_flow) - 1):
        curr = presentation_flow[i]
        next_item = presentation_flow[i + 1]
        
        # If charts are on same or adjacent pages, they might be grouped
        if abs(curr.get('page', 0) - next_item.get('page', 0)) <= 1:
            grouping = {
                'charts': [curr['chart_type'], next_item['chart_type']],
                'sections': [curr['section'], next_item['section']],
                'pages': [curr.get('page'), next_item.get('page')]
            }
            chart_groupings.append(grouping)
    
    return {
        'chart_types': chart_types,
        'presentation_flow': presentation_flow,
        'chart_groupings': chart_groupings,
        'total_charts': len(references),
        'deck_name': Path(pdf_path).name
    }


def learn_layout_patterns(reference_deck_dir: str) -> Dict[str, any]:
    """
    Learn layout patterns from all reference decks
    
    Args:
        reference_deck_dir: Directory containing reference deck PDFs
    
    Returns:
        Dict with learned patterns:
            - section_order: Most common section ordering
            - scope_preference: Preferred scope ordering
            - subject_pairing_frequency: How often subjects are paired
            - grade_ordering: Preferred grade ordering within sections
    """
    deck_dir = Path(reference_deck_dir)
    if not deck_dir.exists():
        print(f"Reference deck directory not found: {reference_deck_dir}")
        return {}
    
    pdf_files = list(deck_dir.glob('*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in {reference_deck_dir}")
        return {}
    
    all_section_sequences = []
    scope_patterns = []
    subject_pairing_count = 0
    grade_orders = defaultdict(list)
    
    # Track overall deck structure
    all_chart_types = []  # All chart types across all decks
    all_presentation_flows = []  # Flow patterns from each deck
    all_chart_groupings = []  # Grouping patterns
    chart_type_counts = Counter()  # How often each chart type appears
    
    for pdf_path in pdf_files:
        print(f"Analyzing reference deck: {pdf_path.name}")
        layout = analyze_reference_deck_layout(str(pdf_path))
        
        if layout.get('section_sequence'):
            all_section_sequences.append(layout['section_sequence'])
        
        if layout.get('scope_pattern'):
            scope_patterns.append(layout['scope_pattern'])
        
        if layout.get('subject_pairing'):
            subject_pairing_count += 1
        
        # Analyze overall deck structure
        structure = analyze_deck_structure(str(pdf_path))
        
        # Track overall structure
        chart_types = structure.get('chart_types', [])
        all_chart_types.extend(chart_types)
        chart_type_counts.update(chart_types)
        
        if structure.get('presentation_flow'):
            all_presentation_flows.append(structure['presentation_flow'])
        
        if structure.get('chart_groupings'):
            all_chart_groupings.extend(structure['chart_groupings'])
        
        # Extract grade ordering per section
        for ref in layout.get('slide_order', []):
            section = ref.get('section')
            grade = ref.get('grade')
            if section and grade:
                try:
                    grade_num = int(grade)
                    # Filter out years (grades should be 1-12, not 2022-2026)
                    if 1 <= grade_num <= 12:
                        grade_orders[section].append(grade_num)
                except (ValueError, TypeError):
                    pass  # Skip non-numeric grades
    
    # Find most common section order
    section_order = None
    if all_section_sequences:
        # Count section transitions
        section_transitions = defaultdict(int)
        for seq in all_section_sequences:
            for i in range(len(seq) - 1):
                transition = (seq[i], seq[i + 1])
                section_transitions[transition] += 1
        
        # Build most common sequence
        if section_transitions:
            # Start with most common first section
            first_sections = [seq[0] for seq in all_section_sequences if seq]
            if first_sections:
                first_section = Counter(first_sections).most_common(1)[0][0]
                section_order = [first_section]
                
                # Build sequence by following most common transitions
                current = first_section
                seen = {first_section}
                while True:
                    next_options = [(s2, count) for (s1, s2), count in section_transitions.items() 
                                  if s1 == current and s2 not in seen]
                    if not next_options:
                        break
                    next_section = max(next_options, key=lambda x: x[1])[0]
                    section_order.append(next_section)
                    seen.add(next_section)
                    current = next_section
    
    # Most common scope pattern
    scope_preference = Counter(scope_patterns).most_common(1)[0][0] if scope_patterns else None
    
    # Subject pairing frequency
    subject_pairing_frequency = subject_pairing_count / len(pdf_files) if pdf_files else 0
    
    # Most common grade ordering per section
    grade_ordering = {}
    for section, grades in grade_orders.items():
        if grades:
            # Count grade order
            grade_counts = Counter(grades)
            sorted_grades = sorted(grade_counts.items(), key=lambda x: (-x[1], x[0]))
            grade_ordering[section] = [g for g, _ in sorted_grades]
    
    # Analyze chart selection patterns
    # Required charts: appear in most decks (>= 80%)
    # Optional charts: appear inconsistently (< 80%)
    total_decks = len(pdf_files)
    required_charts = []
    optional_charts = []
    
    for chart_type, count in chart_type_counts.items():
        frequency = count / total_decks
        if frequency >= 0.8:  # Appears in 80%+ of decks
            required_charts.append(chart_type)
        elif frequency >= 0.3:  # Appears in 30%+ but < 80%
            optional_charts.append(chart_type)
    
    # Find common chart groupings (charts that appear together)
    common_groupings = []
    if all_chart_groupings:
        grouping_counts = Counter()
        for grouping in all_chart_groupings:
            charts_tuple = tuple(sorted(grouping.get('charts', [])))
            grouping_counts[charts_tuple] += 1
        
        # Get groupings that appear in multiple decks
        for grouping_tuple, count in grouping_counts.items():
            if count >= 2:  # Appears in at least 2 decks
                common_groupings.append({
                    'charts': list(grouping_tuple),
                    'frequency': count / total_decks
                })
    
    # Analyze presentation flow patterns
    # Find common sequences of chart types
    common_sequences = []
    if all_presentation_flows:
        # Extract sequences of sections/chart types
        sequence_counts = Counter()
        for flow in all_presentation_flows:
            # Create sequence of sections
            section_sequence = tuple([item['section'] for item in flow[:10]])  # First 10 items
            if len(section_sequence) >= 3:  # At least 3 items
                sequence_counts[section_sequence] += 1
        
        # Get sequences that appear in multiple decks
        for sequence_tuple, count in sequence_counts.items():
            if count >= 2:
                common_sequences.append({
                    'sequence': list(sequence_tuple),
                    'frequency': count / total_decks
                })
    
    return {
        'section_order': section_order or [],
        'scope_preference': scope_preference,
        'subject_pairing_frequency': subject_pairing_frequency,
        'grade_ordering': grade_ordering,
        'chart_selection_patterns': {
            'required_charts': required_charts,
            'optional_charts': optional_charts,
            'chart_type_frequencies': dict(chart_type_counts)
        },
        'presentation_structure': {
            'common_sequences': common_sequences[:5],  # Top 5 sequences
            'common_groupings': common_groupings[:10]  # Top 10 groupings
        },
        'decks_analyzed': total_decks,
        'confidence': len(pdf_files) / max(len(pdf_files), 1)  # Higher confidence with more decks
    }


def get_layout_context(reference_deck_dir: Optional[str] = None, deck_type: Optional[str] = None) -> str:
    """
    Get layout context string to include in LLM prompts
    
    Args:
        reference_deck_dir: Directory containing reference decks (defaults to standard location)
        deck_type: Type of deck being created - 'BOY', 'MOY', or 'EOY' (determines which reference decks to use)
    
    Returns:
        Formatted string with layout guidance including chart selection and organization
    """
    if reference_deck_dir is None:
        # Default to standard reference deck directory
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
                # Unknown deck type, use base directory
                reference_deck_dir = base_reference_dir
                print(f"[Layout Learner] Unknown deck_type '{deck_type}', using base reference directory")
        else:
            # No deck type specified, use base directory (backward compatibility)
            reference_deck_dir = base_reference_dir
            print(f"[Layout Learner] No deck_type specified, using base reference directory")
    else:
        # User provided specific directory, use it
        reference_deck_dir = Path(reference_deck_dir)
    
    patterns = learn_layout_patterns(str(reference_deck_dir))
    
    if not patterns:
        return ""
    
    context_parts = ["**LAYOUT GUIDANCE FROM REFERENCE DECKS:**"]
    context_parts.append("")
    
    # Chart selection guidance
    selection_patterns = patterns.get('chart_selection_patterns', {})
    if selection_patterns:
        required = selection_patterns.get('required_charts', [])
        optional = selection_patterns.get('optional_charts', [])
        
        if required:
            context_parts.append("**CHART SELECTION:**")
            context_parts.append("- Charts that typically appear in most decks (consider including these):")
            for chart_type in required[:10]:  # Top 10
                # Format chart type for readability
                chart_display = chart_type.replace('_', ' ').title()
                context_parts.append(f"  • {chart_display}")
            context_parts.append("")
        
        if optional:
            context_parts.append("- Charts that appear inconsistently (include if relevant to user's request):")
            for chart_type in optional[:10]:  # Top 10
                chart_display = chart_type.replace('_', ' ').title()
                context_parts.append(f"  • {chart_display}")
            context_parts.append("")
    
    # Presentation structure
    structure = patterns.get('presentation_structure', {})
    if structure.get('common_sequences'):
        context_parts.append("**PRESENTATION FLOW:**")
        context_parts.append("- Common chart sequences found in reference decks:")
        for seq_info in structure['common_sequences'][:3]:  # Top 3
            sequence = seq_info['sequence']
            seq_str = " → ".join(sequence)
            context_parts.append(f"  • {seq_str} (appears in {seq_info['frequency']:.0%} of decks)")
        context_parts.append("")
    
    if structure.get('common_groupings'):
        context_parts.append("**CHART GROUPINGS:**")
        context_parts.append("- Charts that typically appear together:")
        for grouping_info in structure['common_groupings'][:5]:  # Top 5
            charts = grouping_info['charts']
            charts_str = " + ".join([c.replace('_', ' ').title() for c in charts])
            context_parts.append(f"  • {charts_str} (appears together in {grouping_info['frequency']:.0%} of decks)")
        context_parts.append("")
    
    # Section ordering
    if patterns.get('section_order'):
        section_order_str = " → ".join(patterns['section_order'])
        context_parts.append("**SECTION ORDERING:**")
        context_parts.append(f"- Preferred section order: {section_order_str}")
        context_parts.append("")
    
    if patterns.get('scope_preference'):
        context_parts.append("**SCOPE ORDERING:**")
        if patterns['scope_preference'] == 'district_first':
            context_parts.append("- Show district-level charts before school-level charts")
        elif patterns['scope_preference'] == 'school_first':
            context_parts.append("- Show school-level charts before district-level charts")
        context_parts.append("")
    
    if patterns.get('subject_pairing_frequency', 0) > 0.5:
        context_parts.append("**SUBJECT PAIRING:**")
        context_parts.append("- Math and Reading charts should be paired on the same slide when possible")
        context_parts.append("")
    
    if patterns.get('grade_ordering'):
        context_parts.append("**GRADE ORDERING:**")
        context_parts.append("- Grade ordering within sections:")
        for section, grades in patterns['grade_ordering'].items():
            if grades:
                grade_str = ", ".join([f"Grade {g}" for g in grades[:5]])  # Limit to first 5
                context_parts.append(f"  - {section}: {grade_str}")
        context_parts.append("")
    
    context_parts.append("**INSTRUCTIONS:**")
    context_parts.append("- When selecting charts, prioritize charts that match the 'required charts' patterns above.")
    context_parts.append("- If a chart type doesn't appear in reference decks and isn't explicitly requested, consider omitting it.")
    context_parts.append("- Follow the presentation flow and grouping patterns when organizing charts.")
    context_parts.append("- Only include charts that are relevant to the user's request and match reference deck patterns.")
    
    return "\n".join(context_parts)


def save_layout_patterns(patterns: Dict[str, any], output_path: str):
    """Save learned layout patterns to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    print(f"Saved layout patterns to {output_path}")


def load_layout_patterns(input_path: str) -> Dict[str, any]:
    """Load learned layout patterns from JSON file"""
    try:
        with open(input_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading layout patterns: {e}")
        return {}

