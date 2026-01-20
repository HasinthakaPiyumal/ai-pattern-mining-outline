# Cluster 14

def set_header_from_context(input: _InputWithHeaders, payload_converter: temporalio.converter.PayloadConverter) -> None:
    execution_id_val = get_execution_id()
    if execution_id_val:
        input.headers = {**input.headers, EXECUTION_ID_KEY: payload_converter.to_payload(execution_id_val)}

@contextmanager
def context_from_header(input: _InputWithHeaders, payload_converter: temporalio.converter.PayloadConverter):
    prev_exec_id = get_execution_id()
    execution_id_payload = input.headers.get(EXECUTION_ID_KEY)
    execution_id_from_header = payload_converter.from_payload(execution_id_payload, str) if execution_id_payload else None
    set_execution_id(execution_id_from_header if execution_id_from_header else None)
    try:
        yield
    finally:
        set_execution_id(prev_exec_id)

class _ContextPropagationWorkflowInboundInterceptor(temporalio.worker.WorkflowInboundInterceptor):

    def init(self, outbound: temporalio.worker.WorkflowOutboundInterceptor) -> None:
        self.next.init(_ContextPropagationWorkflowOutboundInterceptor(outbound))

    async def execute_workflow(self, input: temporalio.worker.ExecuteWorkflowInput) -> Any:
        with context_from_header(input, temporalio.workflow.payload_converter()):
            return await self.next.execute_workflow(input)

    async def handle_signal(self, input: temporalio.worker.HandleSignalInput) -> None:
        with context_from_header(input, temporalio.workflow.payload_converter()):
            return await self.next.handle_signal(input)

    async def handle_query(self, input: temporalio.worker.HandleQueryInput) -> Any:
        with context_from_header(input, temporalio.workflow.payload_converter()):
            return await self.next.handle_query(input)

    def handle_update_validator(self, input: temporalio.worker.HandleUpdateInput) -> None:
        with context_from_header(input, temporalio.workflow.payload_converter()):
            self.next.handle_update_validator(input)

    async def handle_update_handler(self, input: temporalio.worker.HandleUpdateInput) -> Any:
        with context_from_header(input, temporalio.workflow.payload_converter()):
            return await self.next.handle_update_handler(input)

class _ContextPropagationWorkflowOutboundInterceptor(temporalio.worker.WorkflowOutboundInterceptor):

    async def signal_child_workflow(self, input: temporalio.worker.SignalChildWorkflowInput) -> None:
        set_header_from_context(input, temporalio.workflow.payload_converter())
        return await self.next.signal_child_workflow(input)

    async def signal_external_workflow(self, input: temporalio.worker.SignalExternalWorkflowInput) -> None:
        set_header_from_context(input, temporalio.workflow.payload_converter())
        return await self.next.signal_external_workflow(input)

    def start_activity(self, input: temporalio.worker.StartActivityInput) -> temporalio.workflow.ActivityHandle:
        set_header_from_context(input, temporalio.workflow.payload_converter())
        return self.next.start_activity(input)

    async def start_child_workflow(self, input: temporalio.worker.StartChildWorkflowInput) -> temporalio.workflow.ChildWorkflowHandle:
        set_header_from_context(input, temporalio.workflow.payload_converter())
        return await self.next.start_child_workflow(input)

    def start_local_activity(self, input: temporalio.worker.StartLocalActivityInput) -> temporalio.workflow.ActivityHandle:
        set_header_from_context(input, temporalio.workflow.payload_converter())
        return self.next.start_local_activity(input)

@pytest.mark.asyncio
@patch('temporalio.workflow.info')
@patch('temporalio.workflow.in_workflow', return_value=True)
def test_get_execution_id_in_workflow(_mock_in_wf, mock_info):
    from mcp_agent.executor.temporal.temporal_context import get_execution_id
    mock_info.return_value.run_id = 'run-123'
    assert get_execution_id() == 'run-123'

def get_execution_id() -> Optional[str]:
    """Return the current Temporal run identifier to use for gateway routing.

    Priority:
    - If inside a Temporal workflow, return workflow.info().run_id
    - Else if inside a Temporal activity, return activity.info().workflow_run_id
    - Else fall back to the global (best-effort)
    """
    try:
        from temporalio import workflow
        try:
            if workflow.in_workflow():
                return workflow.info().run_id
        except Exception:
            pass
    except Exception:
        pass
    try:
        from temporalio import activity
        try:
            info = activity.info()
            if info is not None and getattr(info, 'workflow_run_id', None):
                return info.workflow_run_id
        except Exception:
            pass
    except Exception:
        pass
    return _EXECUTION_ID

@pytest.mark.asyncio
@patch('temporalio.activity.info')
def test_get_execution_id_in_activity(mock_act_info):
    from mcp_agent.executor.temporal.temporal_context import get_execution_id
    mock_act_info.return_value.workflow_run_id = 'run-aaa'
    assert get_execution_id() == 'run-aaa'

def test_interceptor_restores_prev_value():
    from mcp_agent.executor.temporal.interceptor import context_from_header
    from mcp_agent.executor.temporal.temporal_context import EXECUTION_ID_KEY, set_execution_id, get_execution_id
    import temporalio.converter
    payload_converter = temporalio.converter.default().payload_converter

    class Input:
        headers = {}
    set_execution_id('prev')
    input = Input()
    input.headers[EXECUTION_ID_KEY] = payload_converter.to_payload('new')
    assert get_execution_id() == 'prev'
    with context_from_header(input, payload_converter):
        assert get_execution_id() == 'new'
    assert get_execution_id() == 'prev'

def set_execution_id(execution_id: Optional[str]) -> None:
    global _EXECUTION_ID
    _EXECUTION_ID = execution_id

