import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")


def connect_to_db(supabase_key: str) -> Client:
    """
    Returns a Supabase client instance.
    """
    if not SUPABASE_URL or not supabase_key:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY is not set")
    return create_client(SUPABASE_URL, supabase_key)
