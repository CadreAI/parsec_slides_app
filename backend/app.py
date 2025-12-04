"""
Flask backend application for data ingestion and chart generation
"""
import os
import sys
import json
from pathlib import Path

# Set matplotlib backend to non-interactive before importing chart modules
# This is required when running in Flask/threaded environments (macOS requires main thread for GUI)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI required)

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / 'python'))

from python.data_ingestion import ingest_nwea
from python.nwea_charts import generate_nwea_charts

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200

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
        
        # Ingest NWEA data
        print(f"[Backend] Starting data ingestion for {partner_name}...")
        nwea_data = ingest_nwea(
            partner_name=partner_name,
            config=config,
            chart_filters=chart_filters
        )
        
        print(f"[Backend] Data ingested: {len(nwea_data)} rows")
        
        # Generate charts
        print(f"[Backend] Starting chart generation...")
        data_dir = config.get('paths', {}).get('data_dir', './data')
        chart_paths = generate_nwea_charts(
            partner_name=partner_name,
            output_dir=output_dir,
            config=config,
            chart_filters=chart_filters,
            data_dir=data_dir
        )
        
        print(f"[Backend] Generated {len(chart_paths)} charts")
        
        return jsonify({
            'success': True,
            'charts': chart_paths,
            'summary': {
                'nwea': {
                    'rows': len(nwea_data),
                    'columns': len(nwea_data[0]) if nwea_data else 0
                }
            }
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"[Backend] Error: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

