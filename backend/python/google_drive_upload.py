"""
Google Drive image upload utilities for Python backend
Handles uploading chart images to Google Drive for use in Slides
"""
import os
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from .google_slides_client import get_drive_client


def extract_folder_id_from_url(url: str) -> str:
    """
    Extract folder ID from Google Drive folder URL
    
    Args:
        url: Google Drive folder URL (e.g., https://drive.google.com/drive/folders/1ABC...)
    
    Returns:
        Folder ID or None if not found
    """
    if not url:
        return None
    
    # Try to extract from URL
    if '/folders/' in url:
        parts = url.split('/folders/')
        if len(parts) > 1:
            folder_id = parts[1].split('?')[0].split('/')[0]
            return folder_id
    
    # If it's already just an ID
    if len(url) > 0 and '/' not in url:
        return url
    
    return None


def upload_image_to_drive(
    image_path: str,
    file_name: str = None,
    folder_id: str = None,
    make_public: bool = True
) -> str:
    """
    Upload a single image to Google Drive
    
    Args:
        image_path: Path to the image file
        file_name: Optional custom file name (defaults to basename of image_path)
        folder_id: Optional Google Drive folder ID to upload to
        make_public: Whether to make the file publicly accessible (required for Slides API)
    
    Returns:
        Public URL of the uploaded file, or None if upload failed
    """
    try:
        drive_service = get_drive_client()
        
        if not os.path.exists(image_path):
            print(f"[Drive Upload] Error: File not found: {image_path}")
            return None
        
        if not file_name:
            file_name = os.path.basename(image_path)
        
        # Prepare file metadata
        file_metadata = {
            'name': file_name
        }
        
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        # Upload file
        media = MediaFileUpload(image_path, mimetype='image/png', resumable=True)
        
        print(f"[Drive Upload] Uploading {file_name}...")
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink, webContentLink'
        ).execute()
        
        file_id = file.get('id')
        
        if not file_id:
            print(f"[Drive Upload] Error: No file ID returned for {file_name}")
            return None
        
        # Make file publicly accessible if requested
        if make_public:
            try:
                drive_service.permissions().create(
                    fileId=file_id,
                    body={'role': 'reader', 'type': 'anyone'}
                ).execute()
                print(f"[Drive Upload] Made {file_name} publicly accessible")
            except HttpError as e:
                print(f"[Drive Upload] Warning: Could not make {file_name} public: {e}")
        
        # Return public URL (use webContentLink for direct image access)
        public_url = f"https://drive.google.com/uc?export=view&id={file_id}"
        print(f"[Drive Upload] ✓ Uploaded {file_name} -> {public_url}")
        return public_url
        
    except HttpError as e:
        print(f"[Drive Upload] Error uploading {image_path}: {e}")
        return None
    except Exception as e:
        print(f"[Drive Upload] Unexpected error uploading {image_path}: {e}")
        return None


def upload_images_to_drive_batch(
    images: list,
    folder_id: str = None,
    batch_size: int = 10,
    make_public: bool = True
) -> list:
    """
    Upload multiple images to Google Drive in parallel batches
    
    Args:
        images: List of dicts with 'imagePath' and optionally 'fileName'
        folder_id: Optional Google Drive folder ID to upload to
        batch_size: Number of concurrent uploads per batch
        make_public: Whether to make files publicly accessible
    
    Returns:
        List of public URLs (None for failed uploads)
    """
    if not images:
        return []
    
    print(f"[Drive Upload] Starting batch upload of {len(images)} images (batch size: {batch_size})...")
    
    results = []
    total = len(images)
    
    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = images[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        print(f"[Drive Upload] Processing batch {batch_num}/{total_batches} ({len(batch)} images)...")
        
        # Upload batch in parallel - preserve order by tracking indices
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all uploads and track their original positions
            future_to_index = {}
            batch_results = {}
            
            for local_idx, img_info in enumerate(batch):
                image_path = img_info.get('imagePath') or img_info.get('image_path')
                original_file_name = img_info.get('fileName') or img_info.get('file_name')
                global_idx = batch_start + local_idx
                
                if not image_path:
                    print(f"[Drive Upload] Warning: Skipping image with no path at index {global_idx}")
                    batch_results[global_idx] = None
                    continue
                
                # Add unique identifier to filename to prevent collisions when multiple users upload simultaneously
                if original_file_name:
                    path_obj = Path(original_file_name)
                    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
                    file_name = f"{path_obj.stem}_{unique_id}{path_obj.suffix}"
                else:
                    path_obj = Path(image_path)
                    unique_id = str(uuid.uuid4())[:8]
                    file_name = f"{path_obj.stem}_{unique_id}{path_obj.suffix}"
                
                future = executor.submit(
                    upload_image_to_drive,
                    image_path,
                    file_name,
                    folder_id,
                    make_public
                )
                future_to_index[future] = global_idx
            
            # Collect results as they complete, but store by index
            for future in as_completed(future_to_index.keys()):
                idx = future_to_index[future]
                try:
                    url = future.result()
                    batch_results[idx] = url
                except Exception as e:
                    print(f"[Drive Upload] Error uploading image at index {idx}: {e}")
                    batch_results[idx] = None
            
            # Append results in order for this batch
            for idx in range(batch_start, batch_end):
                results.append(batch_results.get(idx, None))
        
        print(f"[Drive Upload] Batch {batch_num}/{total_batches} complete")
    
    successful = sum(1 for url in results if url is not None)
    print(f"[Drive Upload] ✓ Batch upload complete: {successful}/{total} successful")
    
    return results

