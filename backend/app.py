"""
Flask backend application for data ingestion and chart generation
"""
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Try loading from backend/.env, then project root .env
    backend_env = Path(__file__).parent / '.env'
    root_env = Path(__file__).parent.parent / '.env'
    if backend_env.exists():
        load_dotenv(backend_env)
        print(f"[Backend] Loaded environment variables from {backend_env}")
    elif root_env.exists():
        load_dotenv(root_env)
        print(f"[Backend] Loaded environment variables from {root_env}")
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# Set matplotlib backend to non-interactive before importing chart modules
# This is required when running in Flask/threaded environments (macOS requires main thread for GUI)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI required)

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / 'python'))

from python.data_ingestion import ingest_nwea, ingest_iready
from python.nwea.nwea_charts import generate_nwea_charts
from python.iready.iready_charts import generate_iready_charts
from python.chart_analyzer import analyze_charts_from_index, analyze_charts_batch_paths
from python.slides import create_slides_presentation

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200

@app.route('/config/student-groups', methods=['GET'])
def get_student_groups():
    """
    Get student groups configuration
    
    Returns:
    - success: bool
    - student_groups: dict with student group definitions
    - student_group_order: dict with ordering for student groups
    """
    try:
        student_groups = {
            "All Students": { "type": "all" },
            "English Learners": { "column": "englishlearner", "in": ["Y", "Yes", "True", 1] },
            "Students with Disabilities": { "column": "studentswithdisabilities", "in": ["Y", "Yes", "True", 1] },
            "Socioeconomically Disadvantaged": { "column": "socioeconomicallydisadvantaged", "in": ["Y", "Yes", "True", 1] },
            "Hispanic or Latino": { "column": "ethnicityrace", "in": ["Hispanic", "Hispanic or Latino"] },
            "White": { "column": "ethnicityrace", "in": ["White"] },
            "Black or African American": { "column": "ethnicityrace", "in": ["Black", "African American", "Black or African American"] },
            "Asian": { "column": "ethnicityrace", "in": ["Asian"] },
            "Filipino": { "column": "ethnicityrace", "in": ["Filipino"] },
            "American Indian or Alaska Native": { "column": "ethnicityrace", "in": ["American Indian", "Alaska Native", "American Indian or Alaska Native"] },
            "Native Hawaiian or Pacific Islander": { "column": "ethnicityrace", "in": ["Pacific Islander", "Native Hawaiian", "Native Hawaiian or Other Pacific Islander"] },
            "Two or More Races": { "column": "ethnicityrace", "in": ["Two or More Races", "Multiracial", "Multiple Races"] },
            "Not Stated": { "column": "ethnicityrace", "in": ["Not Stated", "Unknown", ""] },
            "Foster": { "column": "foster", "in": ["Y", "Yes", "True", 1] },
            "Homeless": { "column": "homeless", "in": ["Y", "Yes", "True", 1] }
        }
        
        student_group_order = {
            "All Students": 1,
            "English Learners": 2,
            "Students with Disabilities": 3,
            "Socioeconomically Disadvantaged": 4,
            "Hispanic or Latino": 5,
            "White": 6,
            "Black or African American": 7,
            "Asian": 8,
            "Filipino": 9,
            "American Indian or Alaska Native": 10,
            "Native Hawaiian or Pacific Islander": 11,
            "Two or More Races": 12,
            "Not Stated": 13,
            "Foster": 14,
            "Homeless": 15
        }
        
        return jsonify({
            'success': True,
            'student_groups': student_groups,
            'student_group_order': student_group_order
        }), 200
    except Exception as e:
        error_msg = str(e)
        print(f"[Backend] Error getting student groups: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/config/assessment-filters', methods=['GET'])
def get_assessment_filters():
    """
    Get available filters for each assessment type
    
    Query params:
    - assessments: comma-separated list of assessment IDs (e.g., "nwea,iready")
    
    Returns:
    - success: bool
    - filters: dict mapping assessment ID to available filter options
    """
    try:
        assessments_param = request.args.get('assessments', '')
        requested_assessments = [a.strip() for a in assessments_param.split(',') if a.strip()] if assessments_param else []
        
        # Define available filters for each assessment
        assessment_filters = {
            'nwea': {
                'subjects': ['Reading', 'Mathematics'],
                'quarters': ['Fall', 'Winter', 'Spring'],
                'supports_grades': True,
                'supports_student_groups': True,
                'supports_race': True
            },
            'iready': {
                'subjects': ['ELA', 'Math'],
                'quarters': ['Fall', 'Winter', 'Spring'],
                'supports_grades': True,
                'supports_student_groups': True,
                'supports_race': True
            },
            'star': {
                'subjects': ['Reading', 'Mathematics'],
                'quarters': ['Fall', 'Winter', 'Spring'],
                'supports_grades': True,
                'supports_student_groups': True,
                'supports_race': True
            },
            'cers': {
                'subjects': ['ELA', 'Math'],
                'quarters': [],  # CERS typically doesn't use quarters
                'supports_grades': True,
                'supports_student_groups': True,
                'supports_race': True
            }
        }
        
        # If specific assessments requested, filter to those
        # If no assessments specified, return all available assessments
        if requested_assessments:
            filtered_filters = {k: v for k, v in assessment_filters.items() if k in requested_assessments}
        else:
            filtered_filters = assessment_filters  # Return all assessments
        
        # Merge subjects and quarters from all selected assessments
        all_subjects = set()
        all_quarters = set()
        supports_grades = any(f.get('supports_grades', False) for f in filtered_filters.values())
        supports_student_groups = any(f.get('supports_student_groups', False) for f in filtered_filters.values())
        supports_race = any(f.get('supports_race', False) for f in filtered_filters.values())
        
        for assessment_config in filtered_filters.values():
            all_subjects.update(assessment_config.get('subjects', []))
            all_quarters.update(assessment_config.get('quarters', []))
        
        return jsonify({
            'success': True,
            'filters': {
                'subjects': sorted(list(all_subjects)),
                'quarters': sorted(list(all_quarters)),
                'supports_grades': supports_grades,
                'supports_student_groups': supports_student_groups,
                'supports_race': supports_race
            },
            'assessment_details': filtered_filters
        }), 200
    except Exception as e:
        error_msg = str(e)
        print(f"[Backend] Error getting assessment filters: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/analyze-charts', methods=['POST'])
def analyze_charts():
    """
    Analyze charts using OpenAI Vision API to generate insights
    
    Request body:
    - chartIndexPath: str (path to chart_index.csv)
    - outputDir: str (base directory for chart paths, optional)
    - chartPaths: list[str] (alternative: list of chart file paths)
    - maxCharts: int (optional, limit number of charts to analyze)
    
    Returns:
    - success: bool
    - analyses: list of chart analysis objects with title, description, insights, etc.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        chart_index_path = data.get('chartIndexPath')
        output_dir = data.get('outputDir')
        chart_paths = data.get('chartPaths', [])
        max_charts = data.get('maxCharts')
        
        analyses = []
        
        if chart_index_path:
            # Analyze from chart index CSV
            print(f"[Backend] Analyzing charts from index: {chart_index_path}")
            analyses = analyze_charts_from_index(
                chart_index_path,
                output_dir,
                max_charts=max_charts
            )
        elif chart_paths:
            # Analyze individual chart paths in batches
            print(f"[Backend] Analyzing {len(chart_paths)} charts in batches of 10")
            
            # Limit charts if specified
            if max_charts:
                chart_paths = chart_paths[:max_charts]
            
            analyses = analyze_charts_batch_paths(chart_paths, batch_size=10)
        else:
            return jsonify({
                'success': False,
                'error': 'Either chartIndexPath or chartPaths must be provided'
            }), 400
        
        print(f"[Backend] Completed analysis of {len(analyses)} charts")
        
        return jsonify({
            'success': True,
            'analyses': analyses,
            'count': len(analyses)
        }), 200

    except Exception as e:
        error_msg = str(e)
        print(f"[Backend] Error analyzing charts: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/ingest-and-generate', methods=['POST'])
def ingest_and_generate():
    """
    Combined endpoint for data ingestion and chart generation
    
    Request body:
    - partnerName: str
    - outputDir: str (path to output directory)
    - config: dict (partner configuration)
    - chartFilters: dict (filters for chart generation)
    
    Returns:
    - success: bool
    - charts: list of chart file paths
    - summary: dict with data ingestion summary
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400

        print(f"[Backend] Received data keys: {list(data.keys())}")
        
        partner_name = data.get('partnerName')
        output_dir = data.get('outputDir', './data')
        config = data.get('config', {})
        chart_filters = data.get('chartFilters', {})
        
        print(f"[Backend] Config type: {type(config).__name__}")
        print(f"[Backend] Chart filters type: {type(chart_filters).__name__}")
        print(f"[Backend] Chart filters received: {chart_filters}")
        print(f"[Backend] Config district_name: {config.get('district_name')}")
        print(f"[Backend] Config selected_schools: {config.get('selected_schools')}")

        # Handle case where config might be a JSON string
        if isinstance(config, str):
            print(f"[Backend] Parsing config from JSON string...")
            try:
                config = json.loads(config)
            except json.JSONDecodeError as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid config JSON string: {str(e)}'
                }), 400
        
        # Ensure config is a dict
        if not isinstance(config, dict):
            return jsonify({
                'success': False,
                'error': f'config must be a dict, got {type(config).__name__}'
            }), 400
        
        # Handle case where chart_filters might be a JSON string
        if isinstance(chart_filters, str):
            print(f"[Backend] Parsing chart_filters from JSON string...")
            try:
                chart_filters = json.loads(chart_filters)
            except json.JSONDecodeError:
                print(f"[Backend] Warning: Failed to parse chart_filters, using empty dict")
                chart_filters = {}
        
        # Ensure chart_filters is a dict
        if not isinstance(chart_filters, dict):
            print(f"[Backend] Warning: chart_filters is not a dict ({type(chart_filters).__name__}), using empty dict")
            chart_filters = {}
        
        if not partner_name:
            return jsonify({'success': False, 'error': 'partnerName is required'}), 400

        # Check which data sources are configured
        sources = config.get('sources', {})
        has_nwea = bool(sources.get('nwea'))
        has_iready = bool(sources.get('iready'))
        
        if not has_nwea and not has_iready:
            return jsonify({
                'success': False,
                'error': 'At least one data source (nwea or iready) must be configured in config.sources'
            }), 400
        
        print(f"[Backend] Starting data ingestion for {partner_name}...")
        print(f"[Backend] Data sources configured: NWEA={has_nwea}, iReady={has_iready}")
        
        # Ingest data for configured sources
        nwea_data = []
        iready_data = []
        
        if has_nwea:
            try:
                nwea_data = ingest_nwea(
                    partner_name=partner_name,
                    config=config,
                    chart_filters=chart_filters
                )
                print(f"[Backend] NWEA data ingested: {len(nwea_data)} rows")
            except Exception as e:
                print(f"[Backend] Error ingesting NWEA data: {e}")
                if not has_iready:
                    # Only fail if NWEA is the only source
                    raise
                print(f"[Backend] Continuing with iReady only...")
        
        if has_iready:
            try:
                iready_data = ingest_iready(
                    partner_name=partner_name,
                    config=config,
                    chart_filters=chart_filters
                )
                print(f"[Backend] iReady data ingested: {len(iready_data)} rows")
            except Exception as e:
                print(f"[Backend] Error ingesting iReady data: {e}")
                if not has_nwea:
                    # Only fail if iReady is the only source
                    raise
                print(f"[Backend] Continuing with NWEA only...")
        
        # Generate charts in temporary directory
        print(f"[Backend] Starting chart generation...")
        data_dir = config.get('paths', {}).get('data_dir', './data')
        
        # Create temporary directory for charts
        temp_charts_dir = tempfile.mkdtemp(prefix='parsec_charts_')
        print(f"[Backend] Created temporary charts directory: {temp_charts_dir}")
        
        try:
            all_chart_paths = []
            
            # Generate NWEA charts if data is available
            if nwea_data:
                print(f"[Backend] Generating NWEA charts...")
                nwea_charts = generate_nwea_charts(
                    partner_name=partner_name,
                    output_dir=temp_charts_dir,
                    config=config,
                    chart_filters=chart_filters,
                    data_dir=data_dir,
                    nwea_data=nwea_data
                )
                all_chart_paths.extend(nwea_charts)
                print(f"[Backend] Generated {len(nwea_charts)} NWEA charts")
            
            # Generate iReady charts if data is available
            if iready_data:
                print(f"[Backend] Generating iReady charts...")
                iready_charts = generate_iready_charts(
                    partner_name=partner_name,
                    output_dir=temp_charts_dir,
                    config=config,
                    chart_filters=chart_filters,
                    data_dir=data_dir,
                    iready_data=iready_data
                )
                all_chart_paths.extend(iready_charts)
                print(f"[Backend] Generated {len(iready_charts)} iReady charts")

            print(f"[Backend] Generated {len(all_chart_paths)} total charts")
            
            # Build summary
            summary = {}
            if nwea_data:
                summary['nwea'] = {
                    'rows': len(nwea_data),
                    'columns': len(nwea_data[0]) if nwea_data else 0
                }
            if iready_data:
                summary['iready'] = {
                    'rows': len(iready_data),
                    'columns': len(iready_data[0]) if iready_data else 0
                }
            
            return jsonify({
                'success': True,
                'charts': all_chart_paths,
                'summary': summary,
                'charts_generated': len(all_chart_paths)
            }), 200
        except Exception as chart_error:
            # Clean up temp directory on error
            if os.path.exists(temp_charts_dir):
                print(f"[Backend] Cleaning up temp directory on error: {temp_charts_dir}")
                shutil.rmtree(temp_charts_dir, ignore_errors=True)
            raise chart_error
    except Exception as e:
        error_msg = str(e)
        print(f"[Backend] Error: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/create-slides', methods=['POST'])
def create_slides():
    """
    Create a Google Slides presentation with charts
    
    Request body:
    - title: str (required)
    - charts: list of chart file paths (required)
    - driveFolderUrl: str (optional)
    - enableAIInsights: bool (optional, default True)
    - slides: list of additional slide data (optional)
    - schoolName: str (optional)
    - quarters: list (optional)
    - partnerName: str (optional)
    
    Returns:
    - success: bool
    - presentationId: str
    - presentationUrl: str
    - title: str
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        title = data.get('title')
        if not title:
            return jsonify({'success': False, 'error': 'title is required'}), 400
        
        chart_paths = data.get('charts', [])
        drive_folder_url = data.get('driveFolderUrl')
        enable_ai_insights = data.get('enableAIInsights', True)
        user_prompt = data.get('userPrompt')  # User preferences for decision LLM
        
        print(f"[Backend] Creating slides presentation: {title}")
        print(f"[Backend] Charts: {len(chart_paths)}")
        print(f"[Backend] AI Insights: {enable_ai_insights}")
        if user_prompt:
            print(f"[Backend] User Prompt: {user_prompt[:100]}...")
        
        result = create_slides_presentation(
            title=title,
            chart_paths=chart_paths,
            drive_folder_url=drive_folder_url,
            enable_ai_insights=enable_ai_insights,
            user_prompt=user_prompt
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"[Backend] Error creating slides: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

