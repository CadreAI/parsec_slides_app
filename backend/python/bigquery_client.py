"""
BigQuery client utilities for Python backend
"""
import os
from google.cloud import bigquery
from google.oauth2 import service_account
from pathlib import Path


def _find_credentials_path():
    """
    Find service account credentials file using priority order:
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable
    2. GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_PATH environment variable
    3. google/service_account.json (project root)
    4. ../google/service_account.json (if running from backend/)
    
    Returns:
        Path to credentials file or None if not found
    """
    # Priority 1: GOOGLE_APPLICATION_CREDENTIALS (standard)
    app_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if app_creds and Path(app_creds).exists():
        print(f"[BigQuery] Using GOOGLE_APPLICATION_CREDENTIALS: {app_creds}")
        return app_creds
    
    # Priority 2: GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_PATH
    sa_path = os.environ.get('GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_PATH')
    if sa_path and Path(sa_path).exists():
        print(f"[BigQuery] Using GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_PATH: {sa_path}")
        return sa_path
    
    # Priority 3: google/service_account.json (project root)
    # Try current directory first
    current_dir = Path.cwd()
    default_path = current_dir / 'google' / 'service_account.json'
    if default_path.exists():
        print(f"[BigQuery] Using default location: {default_path}")
        return str(default_path)
    
    # Priority 4: ../google/service_account.json (if running from backend/)
    parent_path = current_dir.parent / 'google' / 'service_account.json'
    if parent_path.exists():
        print(f"[BigQuery] Using parent directory location: {parent_path}")
        return str(parent_path)
    
    # Try one more level up (in case we're in backend/python/)
    grandparent_path = current_dir.parent.parent / 'google' / 'service_account.json'
    if grandparent_path.exists():
        print(f"[BigQuery] Using grandparent directory location: {grandparent_path}")
        return str(grandparent_path)
    
    return None


def get_bigquery_client(project_id: str, credentials_path: str = None):
    """
    Get or create a BigQuery client instance
    
    Args:
        project_id: GCP project ID
        credentials_path: Optional path to service account JSON file
    
    Returns:
        BigQuery client instance
    """
    # If credentials_path not provided, try to find it automatically
    if not credentials_path:
        credentials_path = _find_credentials_path()
    
    # Set up credentials
    if credentials_path and Path(credentials_path).exists():
        print(f"[BigQuery] Loading credentials from: {credentials_path}")
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/bigquery"]
        )
        client = bigquery.Client(project=project_id, credentials=credentials)
        print(f"[BigQuery] BigQuery client initialized with service account")
    else:
        # Use Application Default Credentials or GOOGLE_APPLICATION_CREDENTIALS
        print(f"[BigQuery] No credentials file found, using Application Default Credentials")
        print(f"[BigQuery] If this fails, set GOOGLE_APPLICATION_CREDENTIALS or place credentials at google/service_account.json")
        client = bigquery.Client(project=project_id)
    
    return client


def run_query(sql: str, client: bigquery.Client, params: dict = None):
    """
    Execute a BigQuery SQL query
    
    Args:
        sql: SQL query string
        client: BigQuery client instance
        params: Optional query parameters
    
    Returns:
        List of dictionaries representing rows
    """
    job_config = bigquery.QueryJobConfig()
    
    # Note: SQL builders use string interpolation instead of query parameters
    # to avoid BigQuery parameter type issues. Query parameters are not set here.
    # If you need to use query parameters, update sql_builders.py to use @parameter syntax.
    
    # Set query priority
    job_config.priority = bigquery.QueryPriority.INTERACTIVE
    
    print(f"[BigQuery] Executing query...")
    try:
        query_job = client.query(sql, job_config=job_config)
        
        # Wait for query to complete and log progress
        import time
        start_time = time.time()
        status_check_interval = 10  # seconds
        last_log_time = start_time
        
        while not query_job.done():
            elapsed = time.time() - start_time
            if elapsed - last_log_time >= status_check_interval:
                print(f"[BigQuery] Still running... ({elapsed:.1f}s elapsed)")
                last_log_time = elapsed
                status_check_interval = min(status_check_interval + 10, 60)  # Cap at 60s intervals
        
        # Check for errors
        if query_job.errors:
            error_msg = f"Query failed: {query_job.errors}"
            print(f"[BigQuery] ERROR: {error_msg}")
            raise Exception(error_msg)
        
        # Get results
        elapsed_time = time.time() - start_time
        print(f"[BigQuery] Query completed in {elapsed_time:.1f}s")
        print(f"[BigQuery] Converting results to list...")
        
        results = query_job.result()
        rows = []
        
        # Log progress every 5,000 rows (more frequent updates for large datasets)
        log_interval = 5000
        row_start_time = time.time()
        
        for i, row in enumerate(results):
            rows.append(dict(row))
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - row_start_time
                print(f"[BigQuery] Processed {i + 1:,} rows... ({elapsed:.1f}s)")
        
        print(f"[BigQuery] Total rows: {len(rows):,}")
        return rows
    except Exception as e:
        error_msg = f"BigQuery error: {str(e)}"
        print(f"[BigQuery] ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        raise Exception(error_msg) from e

