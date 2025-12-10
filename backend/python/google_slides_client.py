"""
Google Slides API client for Python backend
Handles both service account (for production/Render) and OAuth (for local dev) authentication
"""
import json
import os
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/presentations', 'https://www.googleapis.com/auth/drive']


def _find_oauth_credentials_path():
    """
    Find OAuth credentials.json file using priority order:
    1. GOOGLE_OAUTH_CREDENTIALS_PATH environment variable
    2. google/credentials.json (project root)
    3. ../google/credentials.json (if running from backend/)
    
    Returns:
        Path to credentials file or None if not found
    """
    # Priority 1: Environment variable
    creds_path = os.environ.get('GOOGLE_OAUTH_CREDENTIALS_PATH')
    if creds_path and Path(creds_path).exists():
        return creds_path
    
    # Priority 2: google/credentials.json (project root)
    current_dir = Path.cwd()
    default_path = current_dir / 'google' / 'credentials.json'
    if default_path.exists():
        return str(default_path)
    
    # Priority 3: ../google/credentials.json (if running from backend/)
    parent_path = current_dir.parent / 'google' / 'credentials.json'
    if parent_path.exists():
        return str(parent_path)
    
    # Try one more level up (in case we're in backend/python/)
    grandparent_path = current_dir.parent.parent / 'google' / 'credentials.json'
    if grandparent_path.exists():
        return str(grandparent_path)
    
    return None


def _find_token_path():
    """
    Find OAuth token.json file using priority order:
    1. GOOGLE_OAUTH_TOKEN_PATH environment variable
    2. google/token.json (project root)
    3. ../google/token.json (if running from backend/)
    
    Returns:
        Path to token file
    """
    # Priority 1: Environment variable
    token_path = os.environ.get('GOOGLE_OAUTH_TOKEN_PATH')
    if token_path:
        return token_path
    
    # Priority 2: google/token.json (project root)
    current_dir = Path.cwd()
    default_path = current_dir / 'google' / 'token.json'
    if default_path.exists():
        return str(default_path)
    
    # Priority 3: ../google/token.json (if running from backend/)
    parent_path = current_dir.parent / 'google' / 'token.json'
    if parent_path.exists():
        return str(parent_path)
    
    # Try one more level up (in case we're in backend/python/)
    grandparent_path = current_dir.parent.parent / 'google' / 'token.json'
    if grandparent_path.exists():
        return str(grandparent_path)
    
    # Default to project root
    return str(default_path)


def _get_service_account_credentials():
    """
    Try to get service account credentials for production use.
    Uses the same logic as bigquery_client.py to find service account JSON.
    
    Returns:
        Service account credentials or None if not found
    """
    # Import the helper from bigquery_client to reuse the same logic
    from python.bigquery_client import _find_credentials_path
    
    creds_path = _find_credentials_path()
    if creds_path:
        print(f"[Google Slides] Loading service account credentials from: {creds_path}")
        return service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=SCOPES
        )
    return None


def get_slides_client():
    """
    Get or create a Google Slides API client instance.
    Tries service account first (for production/Render), falls back to OAuth (for local dev).
    
    Returns:
        Google Slides API service client
    """
    # Priority 1: Try service account (for production/Render)
    sa_creds = _get_service_account_credentials()
    if sa_creds:
        print("[Google Slides] Using service account credentials")
        return build('slides', 'v1', credentials=sa_creds)
    
    # Priority 2: Try OAuth (for local development)
    print("[Google Slides] Service account not found, trying OAuth...")
    creds = None
    token_path = _find_token_path()
    creds_path = _find_oauth_credentials_path()
    
    # Load existing token if available
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception as e:
            print(f"[Google Slides] Error loading token: {e}")
            print(f"[Google Slides] Token file may be in wrong format. Will re-authenticate.")
            # Try to back up the old token
            try:
                backup_path = f"{token_path}.backup"
                import shutil
                shutil.copy2(token_path, backup_path)
                print(f"[Google Slides] Backed up old token to: {backup_path}")
            except:
                pass
            # Delete the invalid token file
            try:
                os.remove(token_path)
                print(f"[Google Slides] Removed invalid token file")
            except:
                pass
            creds = None
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[Google Slides] Refreshing expired token...")
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"[Google Slides] Error refreshing token: {e}")
                creds = None
        
        if not creds:
            if not creds_path or not os.path.exists(creds_path):
                raise FileNotFoundError(
                    f"OAuth credentials not found. Please place credentials.json at:\n"
                    f"  - {Path.cwd() / 'google' / 'credentials.json'}\n"
                    f"  - Or set GOOGLE_OAUTH_CREDENTIALS_PATH environment variable"
                )
            
            print(f"[Google Slides] Starting OAuth flow...")
            print(f"[Google Slides] Credentials: {creds_path}")
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            # Use fixed port 8080 - must be registered in Google Cloud Console
            # Redirect URI: http://localhost:8080 (without trailing slash)
            # Google Cloud Console requires URIs without trailing slash
            creds = flow.run_local_server(port=8080, open_browser=True, redirect_uri_trailing_slash=False)
        
        # Save the credentials for the next run
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, 'w') as token_file:
            token_file.write(creds.to_json())
        print(f"[Google Slides] Token saved to: {token_path}")
    
    return build('slides', 'v1', credentials=creds)


def get_drive_client():
    """
    Get or create a Google Drive API client instance.
    Tries service account first (for production/Render), falls back to OAuth (for local dev).
    
    Returns:
        Google Drive API service client
    """
    # Priority 1: Try service account (for production/Render)
    sa_creds = _get_service_account_credentials()
    if sa_creds:
        print("[Google Drive] Using service account credentials")
        return build('drive', 'v3', credentials=sa_creds)
    
    # Priority 2: Try OAuth (for local development)
    print("[Google Drive] Service account not found, trying OAuth...")
    creds = None
    token_path = _find_token_path()
    creds_path = _find_oauth_credentials_path()
    
    # Load existing token if available
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception as e:
            print(f"[Google Drive] Error loading token: {e}")
            print(f"[Google Drive] Token file may be in wrong format. Will re-authenticate.")
            # Try to back up the old token
            try:
                backup_path = f"{token_path}.backup"
                import shutil
                shutil.copy2(token_path, backup_path)
                print(f"[Google Drive] Backed up old token to: {backup_path}")
            except:
                pass
            # Delete the invalid token file
            try:
                os.remove(token_path)
                print(f"[Google Drive] Removed invalid token file")
            except:
                pass
            creds = None
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[Google Drive] Refreshing expired token...")
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"[Google Drive] Error refreshing token: {e}")
                creds = None
        
        if not creds:
            if not creds_path or not os.path.exists(creds_path):
                raise FileNotFoundError(
                    f"OAuth credentials not found. Please place credentials.json at:\n"
                    f"  - {Path.cwd() / 'google' / 'credentials.json'}\n"
                    f"  - Or set GOOGLE_OAUTH_CREDENTIALS_PATH environment variable"
                )
            
            print(f"[Google Drive] Starting OAuth flow...")
            print(f"[Google Drive] Credentials: {creds_path}")
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            # Use fixed port 8080 - must be registered in Google Cloud Console
            # Redirect URI: http://localhost:8080 (without trailing slash)
            # Google Cloud Console requires URIs without trailing slash
            creds = flow.run_local_server(port=8080, open_browser=True, redirect_uri_trailing_slash=False)
        
        # Save the credentials for the next run
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, 'w') as token_file:
            token_file.write(creds.to_json())
        print(f"[Google Drive] Token saved to: {token_path}")
    
    return build('drive', 'v3', credentials=creds)

