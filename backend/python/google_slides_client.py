"""
Google Slides API client for Python backend
Uses service account authentication only
"""
import os
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
    'https://www.googleapis.com/auth/presentations',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.readonly'
]

# Default Shared Drive folder for creating presentations (parsectest Shared Drive)
# This is the root folder of the "parsectest" Shared Drive (Drive ID: 0AAPtE6vMyRK8Uk9PVA)
DEFAULT_SLIDES_FOLDER_ID = '0AAPtE6vMyRK8Uk9PVA'


def _get_service_account_credentials():
    """
    Get service account credentials for Google APIs.
    Uses the same logic as bigquery_client.py to find service account JSON.
    
    Returns:
        Service account credentials
    
    Raises:
        FileNotFoundError: If service account file not found
    """
    try:
        from python.bigquery_client import _find_credentials_path
        creds_path = _find_credentials_path()
        if creds_path:
            print(f"[Google Slides] Loading service account credentials from: {creds_path}")
            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=SCOPES
            )
            print(f"[Google Slides] Service account email: {credentials.service_account_email}")
            return credentials
        else:
            raise FileNotFoundError(
                "Service account credentials not found. Please ensure service_account.json exists."
            )
    except Exception as e:
        print(f"[Google Slides] Error loading service account: {e}")
        raise


def get_slides_client():
    """
    Get or create a Google Slides API client instance using service account.
    
    Returns:
        Google Slides API service client
    """
    credentials = _get_service_account_credentials()
    return build('slides', 'v1', credentials=credentials)


def get_drive_client():
    """
    Get or create a Google Drive API client instance using service account.
    
    Returns:
        Google Drive API service client
    """
    credentials = _get_service_account_credentials()
    return build('drive', 'v3', credentials=credentials)


def create_presentation_via_drive(title: str, folder_id: str = None) -> str:
    """
    Create a Google Slides presentation via Drive API (required for service account).
    Service accounts cannot create presentations directly via Slides API, so we use Drive API.
    
    Args:
        title: Title of the presentation
        folder_id: Optional folder ID to create the presentation in (must be Shared Drive folder).
                   If not provided, uses DEFAULT_SLIDES_FOLDER_ID.
    
    Returns:
        Presentation ID
    """
    # Use default Shared Drive folder if not provided
    if not folder_id:
        folder_id = DEFAULT_SLIDES_FOLDER_ID
        print(f"[Google Slides] No folder provided, using default Shared Drive folder: {folder_id}")
        print(f"[Google Slides]   This is the 'parsectest' Shared Drive (Drive ID: {folder_id})")
    else:
        print(f"[Google Slides] Using provided folder: {folder_id}")
    
    # Get service account credentials
    credentials = _get_service_account_credentials()
    print("[Google Slides] Using service account to create presentation via Drive API")
    drive_service = build('drive', 'v3', credentials=credentials)
    
    # Verify folder is accessible and in Shared Drive (matching reference script)
    try:
        print(f"[Google Slides] Verifying folder access: {folder_id}")
        folder_info = drive_service.files().get(
            fileId=folder_id,
            fields='id, name, mimeType, shared, owners, permissions, driveId',
            supportsAllDrives=True
        ).execute()
        
        print(f"[Google Slides] ✓ Folder accessible!")
        print(f"[Google Slides]   Name: {folder_info.get('name', 'Unknown')}")
        print(f"[Google Slides]   Shared: {folder_info.get('shared', False)}")
        
        drive_id = folder_info.get('driveId')
        if drive_id:
            print(f"[Google Slides]   ✓ This folder is in a Shared Drive (Drive ID: {drive_id})")
            print(f"[Google Slides]   → Service account can create files here!")
        else:
            print(f"[Google Slides]   ⚠ This folder appears to be in \"My Drive\" (not a Shared Drive)")
            print(f"[Google Slides]   → Service accounts work best with Shared Drive folders")
            print(f"[Google Slides]   → You may encounter ownership/storage quota issues")
        
        if folder_info.get('owners'):
            owner_emails = [owner.get('emailAddress', '') for owner in folder_info['owners']]
            print(f"[Google Slides]   Owner: {', '.join(owner_emails)}")
    except HttpError as e:
        print(f"[Google Slides] ✗ Cannot access folder")
        error_details = e.error_details if hasattr(e, 'error_details') else []
        if error_details:
            error = error_details[0]
            print(f"[Google Slides]   Error: {error.get('message', str(e))}")
            print(f"[Google Slides]   Reason: {error.get('reason', 'unknown')}")
        else:
            print(f"[Google Slides]   Error: {e}")
        print(f"[Google Slides]   Continuing anyway - creation will fail if there's a real issue")
    
    # Match the exact format from the working test script
    file_metadata = {
        'name': title,
        'mimeType': 'application/vnd.google-apps.presentation',  # This MIME type makes it a Slides file
        'parents': [folder_id]  # Google Drive folder ID (must be in Shared Drive)
    }
    
    try:
        # Match the exact API call format from the working test script
        response = drive_service.files().create(
            body=file_metadata,
            supportsAllDrives=True,  # Required for shared drives
            fields='id, name, mimeType, parents'  # Request specific fields in the response
        ).execute()
        
        file_id = response['id']
        print(f"[Google Slides] ✓ Created presentation successfully!")
        print(f"[Google Slides] Title: {response['name']}")
        print(f"[Google Slides] File ID: {file_id}")
        print(f"[Google Slides] URL: https://docs.google.com/presentation/d/{file_id}/edit")
        return file_id
    except HttpError as e:
        print(f"[Google Slides]")
        print(f"[Google Slides] Error details:")
        error_details = e.error_details if hasattr(e, 'error_details') else []
        error_reason = None
        if error_details:
            error_reason = error_details[0].get('reason', 'unknown')
            for index, error in enumerate(error_details, 1):
                print(f"[Google Slides] Error {index}: {error.get('message', 'Unknown error')} (reason: {error.get('reason', 'unknown')})")
        else:
            print(f"[Google Slides] Error: {e}")
        
        # If storage quota exceeded, provide helpful message
        if error_reason == 'storageQuotaExceeded':
            print(f"[Google Slides]")
            print(f"[Google Slides] ⚠️  STORAGE QUOTA ISSUE:")
            print(f"[Google Slides]   The service account's personal Drive storage is full.")
            print(f"[Google Slides]   Solution: Clean up files from the service account's personal Drive")
            print(f"[Google Slides]   Folder ID used: {folder_id}")
            print(f"[Google Slides]   Verify this folder is in a Shared Drive, not the service account's My Drive")
        
        raise
