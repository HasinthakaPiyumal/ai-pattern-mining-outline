# Cluster 35

class ResearchSessionState:
    """
    Thread-safe session state manager for deep research.
    Ensures only one browser session is active per research query.
    """

    def __init__(self):
        self._sessions: Dict[str, BrowserSessionManager] = {}
        self._lock = threading.Lock()
        self._session_timestamps: Dict[str, float] = {}
        self._max_session_age = 300

    def get_or_create_session(self, session_id: str, headless: bool=False, timeout: int=30) -> Optional[BrowserSessionManager]:
        """
        Get an existing session or create a new one for the given session ID.
        """
        with self._lock:
            print(f'🔍 Session state: {len(self._sessions)} active sessions, checking for ID: {session_id}')
            self._cleanup_old_sessions()
            if session_id in self._sessions:
                session = self._sessions[session_id]
                print(f'📋 Found existing session for ID: {session_id}, active: {session.is_active()}, instance: {id(session)}')
                if session.is_active():
                    print(f'♻️  Reusing existing browser session for research ID: {session_id}')
                    return session
                else:
                    print(f'🔄 Removing inactive session for research ID: {session_id}')
                    del self._sessions[session_id]
                    if session_id in self._session_timestamps:
                        del self._session_timestamps[session_id]
            print(f'🌐 Creating new browser session for research ID: {session_id}')
            session = BrowserSessionManager(headless=headless, timeout=timeout)
            session.get_or_create_searcher()
            self._sessions[session_id] = session
            self._session_timestamps[session_id] = time.time()
            print(f'✅ Created new session instance: {id(session)} for ID: {session_id}')
            print(f'📊 Total active sessions: {len(self._sessions)}')
            return session

    def remove_session(self, session_id: str):
        """
        Remove and close a session.
        """
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                try:
                    session.close()
                except Exception as e:
                    print(f'⚠️ Error closing session {session_id}: {e}')
                del self._sessions[session_id]
                if session_id in self._session_timestamps:
                    del self._session_timestamps[session_id]
                print(f'🏁 Removed session for research ID: {session_id}')

    def _cleanup_old_sessions(self):
        """
        Clean up sessions older than max_session_age.
        """
        current_time = time.time()
        sessions_to_remove = []
        for session_id, timestamp in self._session_timestamps.items():
            if current_time - timestamp > self._max_session_age:
                sessions_to_remove.append(session_id)
        for session_id in sessions_to_remove:
            print(f'🧹 Cleaning up old session: {session_id}')
            if session_id in self._sessions:
                try:
                    self._sessions[session_id].close()
                except:
                    pass
                del self._sessions[session_id]
            del self._session_timestamps[session_id]

