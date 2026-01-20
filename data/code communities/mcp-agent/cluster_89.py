# Cluster 89

def _install_tracer_stubs(monkeypatch):
    recorded_exporters = []
    provider_kwargs = []

    class StubOTLPExporter:

        def __init__(self, *, endpoint=None, headers=None):
            self.endpoint = endpoint
            self.headers = headers
            recorded_exporters.append(self)

    class StubBatchSpanProcessor:

        def __init__(self, exporter):
            self.exporter = exporter

        def on_start(self, *_, **__):
            pass

        def on_end(self, *_, **__):
            pass

        def shutdown(self, *_, **__):
            pass

        def force_flush(self, *_, **__):
            pass

    class StubTracerProvider:

        def __init__(self, **kwargs):
            provider_kwargs.append(kwargs)
            self.processors = []

        def add_span_processor(self, processor):
            self.processors.append(processor)

        def shutdown(self):
            pass
    monkeypatch.setattr('mcp_agent.tracing.tracer.OTLPSpanExporter', StubOTLPExporter)
    monkeypatch.setattr('mcp_agent.tracing.tracer.BatchSpanProcessor', StubBatchSpanProcessor)
    monkeypatch.setattr('mcp_agent.tracing.tracer.TracerProvider', StubTracerProvider)
    monkeypatch.setattr(TracingConfig, '_global_provider_set', True, raising=False)
    monkeypatch.setattr(TracingConfig, '_instrumentation_initialized', True, raising=False)
    return (recorded_exporters, provider_kwargs)

