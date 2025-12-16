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
import httpx
from clerk_backend_api import Clerk
from clerk_backend_api.security.types import AuthenticateRequestOptions

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / 'python'))

from python.data_ingestion import ingest_nwea, ingest_iready, ingest_star
from python.nwea.nwea_charts import generate_nwea_charts
from python.iready.iready_charts import generate_iready_charts
from python.star.star_charts import generate_star_charts
from python.llm.chart_analyzer import analyze_charts_from_index, analyze_charts_batch_paths
from python.slides import create_slides_presentation
from celery_app import celery_app
from python.tasks.slides import create_deck_with_slides_task

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

CLERK_SECRET_KEY = os.environ.get('CLERK_SECRET_KEY')
CLERK_AUTHORIZED_PARTIES = [
    p.strip() for p in os.environ.get('CLERK_AUTHORIZED_PARTIES', '').split(',') if p.strip()
]


def authenticate_request():
    """
    Verify Clerk session using the Python SDK. Raises on failure.
    Expects Authorization: Bearer <token> header forwarded from frontend.
    """
    try:
        client = Clerk(bearer_auth=CLERK_SECRET_KEY)
        httpx_request = httpx.Request(
            method=request.method,
            url=request.url,
            headers=dict(request.headers)
        )
        opts = AuthenticateRequestOptions(
            authorized_parties=CLERK_AUTHORIZED_PARTIES
        )
        state = client.authenticate_request(httpx_request, opts)
        if not state.is_signed_in:
            raise ValueError('Unauthorized')
        return state
    except Exception as e:
        raise ValueError(f'Auth failed: {e}') from e

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        authenticate_request()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 401
    return jsonify({'status': 'ok'}), 200


@app.route('/tasks/create-deck-with-slides', methods=['POST'])
def queue_create_deck_with_slides_task():
    """
    Queue a Celery task that ingests data, generates charts, and creates a Google Slides presentation.

    Request body (required):
    - partnerName: str
    - config: dict (partner configuration)
    - chartFilters: dict (filters for chart generation)
    - title: str (presentation title)
    - clerkUserId: str (for tracking in DB)

    Request body (optional):
    - driveFolderUrl: str
    - enableAIInsights: bool (default True)
    - userPrompt: str
    - description: str

    Returns:
    - success: bool
    - taskId: str (Celery task ID)
    - status: str (queued)
    """
    try:
        authenticate_request()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 401

    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400

        # Extract required fields
        partner_name = data.get('partnerName')
        config = data.get('config', {})
        chart_filters = data.get('chartFilters', {})
        title = data.get('title')
        clerk_user_id = data.get('clerkUserId')

        # Validate required fields
        if not partner_name:
            return jsonify({'success': False, 'error': 'partnerName is required'}), 400
        if not title:
            return jsonify({'success': False, 'error': 'title is required'}), 400
        if not clerk_user_id:
            return jsonify({'success': False, 'error': 'clerkUserId is required'}), 400

        # Extract optional fields
        drive_folder_url = data.get('driveFolderUrl')
        enable_ai_insights = data.get('enableAIInsights', True)
        user_prompt = data.get('userPrompt')
        description = data.get('description')
        theme_color_raw = data.get('themeColor')
        theme_color = theme_color_raw if theme_color_raw and theme_color_raw.strip() else '#0094bd'  # Default to Parsec blue
        print(f"[Backend] Received themeColor from request: '{theme_color_raw}', using: '{theme_color}'")

        print(f"[Backend] Queueing create_deck_with_slides task for partner: {partner_name}, title: {title}")

        # Queue the Celery task
        task = create_deck_with_slides_task.apply_async(kwargs={
            'partner_name': partner_name,
            'config': config,
            'chart_filters': chart_filters,
            'title': title,
            'clerk_user_id': clerk_user_id,
            'drive_folder_url': drive_folder_url,
            'enable_ai_insights': enable_ai_insights,
            'user_prompt': user_prompt,
            'description': description,
            'theme_color': theme_color
        })

        # Store task in Supabase
        try:
            from python.supabase_client import get_supabase_client
            supabase = get_supabase_client()

            task_data = {
                'clerk_user_id': clerk_user_id,
                'task_type': 'create_deck_with_slides',
                'celery_task_id': task.id,
                'status': 'PENDING',
                'result': None
            }

            supabase.table('tasks').insert(task_data).execute()
            print(f"[Backend] Task {task.id} stored in DB for user {clerk_user_id}")
        except Exception as e:
            print(f"[Backend] Failed to store task in DB: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the request if DB insert fails

        return jsonify({'success': True, 'taskId': task.id, 'status': 'queued'}), 202

    except Exception as e:
        error_msg = str(e)
        print(f"[Backend] Error queueing task: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/tasks/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """
    Get Celery task status and result for a given task ID.

    Path params:
    - task_id: str (Celery task ID)

    Returns:
    - taskId: str
    - state: str (PENDING, STARTED, RETRY, FAILURE, SUCCESS)
    - result: any (if SUCCESS)
    - error: str (if FAILURE)
    """
    try:
        authenticate_request()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 401

    result = celery_app.AsyncResult(task_id)
    response = {
        'taskId': task_id,
        'state': result.state,
    }
    print(f"[Backend] Task {task_id} state: {result.state}")
    if result.state == 'SUCCESS':
        response['result'] = result.result
    elif result.state == 'FAILURE':
        response['error'] = str(result.info)
    return jsonify(response), 200


@app.route('/tasks', methods=['GET'])
def get_user_tasks():
    """
    Get all tasks for a user.

    Query params:
    - clerkUserId: str (required) - Clerk user ID
    - status: str (optional) - Filter by status (comma-separated: PENDING,STARTED,SUCCESS,FAILURE)
    - limit: int (optional) - Limit number of tasks returned (default: all)

    Returns:
    - success: bool
    - tasks: list of task objects
    """
    try:
        authenticate_request()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 401

    try:
        clerk_user_id = request.args.get('clerkUserId')
        if not clerk_user_id:
            return jsonify({'success': False, 'error': 'clerkUserId is required'}), 400

        status_filter = request.args.get('status', '')
        limit = request.args.get('limit')

        from python.supabase_client import get_supabase_client
        supabase = get_supabase_client()

        # Query tasks for this user
        query = supabase.table('tasks').select('*').eq('clerk_user_id', clerk_user_id)

        # Apply status filter if provided
        if status_filter:
            statuses = [s.strip() for s in status_filter.split(',') if s.strip()]
            if statuses:
                query = query.in_('status', statuses)

        # Apply limit if provided
        if limit:
            try:
                query = query.limit(int(limit))
            except ValueError:
                pass  # Ignore invalid limit values

        response = query.order('created_at', desc=True).execute()

        tasks = response.data if response.data else []

        status_msg = f"with status {status_filter}" if status_filter else "all statuses"
        print(f"[Backend] Found {len(tasks)} tasks for user {clerk_user_id} ({status_msg})")

        return jsonify({'success': True, 'tasks': tasks}), 200

    except Exception as e:
        error_msg = str(e)
        print(f"[Backend] Error fetching tasks: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

