from supabase import create_client, Client
from typing import Optional
from functools import lru_cache
import logging
from backend.config.settings import Settings

logger = logging.getLogger(__name__)

class SupabaseClient:

    _instance: Optional['SupabaseClient'] = None
    _client: Optional[Client] = None
    _admin_client: Optional[Client] = None

    def __new__(cls):
        # Call ts before init when new instance, it checks if it already exists and returns that one or creates a new one
        if cls._instance is None:
            cls._instance = super(SupabaseClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Check if clients are already initialized so we don't reinitialize them
        if self._client is None:
            self._initialize_clients()

    def _initialize_clients(self):
        # Initializes both regular and admin supabase clients
        """
        1. Loads settings
        2. Creates regular client or admin client
        - Regular client has row level security (RLS) policies
        - Admin can bypass 
        3. Logs success or failure
        """
        settings = get_settings()

        try:
            # Ts key exposed to frontend
            self._client = create_client(
                supabase_url=settings.SUPABASE_URL,
                supabase_key = settings.SUPABASE_ANON_KEY
            )

            # Ts key never exposed to frontend (full perms)
            self._admin_client = create_client(
                supabase_url=settings.SUPABASE_URL,
                supabase_key = settings.SUPABASE_SERVICE_ROLE_KEY
            )

            logger.ingo("Supabase clients initialized succesfully")
        except Exception as e:
            logger.error(f"Failed to intialize Supabase clients: {e}")
            raise

    @property
    def client(self) -> Client:
        # Property to get the regular client
        """
        Used for
        - User level db operations
        - Operations w RLS policies
        - Most endpoints
        """

        if self._client is None:
            # lazy
            self._initialize_clients()
        return self._client
    
    @property
    def admin_client(self) -> Client:
        # Gets the admin supabase client
        """
        Used for
        - Admin operations
        - Bypass RLS, so bulk operations
        - Background jobs
        - Operations w full db access
        """

        if self._admin_client is None:
            self._initialize_clients()
        return self._admin_client
    
    def get_client_with_auth(self, access_token: str) -> Client:
        # Creates new client w a specific user token
        settings = get_settings()
        client = create_client(
            supabase_url=settings.SUPABASE_URL,
            supabase_key=settings.SUPABASE_ANON_KEY
        )
        # Client thinks its authenitcated used
        client.auth.set_session(access_token, refresh_token="")
        return client
    

@lru_cache()
def get_settings() -> Settings:
    # Get the cached instance of the settings
    return Settings()