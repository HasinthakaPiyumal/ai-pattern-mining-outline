# Cluster 10

class App:
    """The TraceRoot API app."""

    def __init__(self):
        self.limiter = Limiter(key_func=get_remote_address)
        self.app = FastAPI(title='TraceRoot API', version=version)
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        self.local_mode = os.getenv('REST_LOCAL_MODE', 'false').lower() == 'true'
        self.add_middleware()
        self.telemetry_router = TelemetryRouter(self.local_mode, self.limiter)
        self.app.include_router(self.telemetry_router.router, prefix='/v1/explore', tags=['telemetry'])
        self.chat_router = ChatRouterClass(self.local_mode, self.limiter)
        self.app.include_router(self.chat_router.router, prefix='/v1/explore', tags=['chat'])
        self.verify_router = VerifyRouter(self.limiter)
        self.app.include_router(self.verify_router.router, prefix='/v1/verify', tags=['verify'])
        self.internal_router = InternalRouter()
        self.app.include_router(self.internal_router.router, prefix='/v1/internal', tags=['internal'])

    def add_middleware(self):
        main_domain = os.getenv('MAIN_DOMAIN')
        allow_origins = [f'https://{main_domain}', f'https://api.{main_domain}', 'http://localhost:3000', 'http://localhost:3001', 'http://127.0.0.1:3000', 'http://127.0.0.1:3001']
        self.app.add_middleware(CORSMiddleware, allow_origins=allow_origins, allow_credentials=True, allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'], allow_headers=['*'])

