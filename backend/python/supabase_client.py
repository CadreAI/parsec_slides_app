"""
Supabase client for Python backend
Uses service role key for admin access (bypasses RLS)
"""
import os
from supabase import create_client, Client


def get_supabase_client() -> Client:
    """Get Supabase client with service role key for backend operations"""
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment variables"
        )
    
    return create_client(supabase_url, supabase_key)
