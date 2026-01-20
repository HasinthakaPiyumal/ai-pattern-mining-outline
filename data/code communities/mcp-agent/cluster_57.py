# Cluster 57

def create_display_layout() -> Layout:
    """Create the display layout"""
    layout = Layout()
    layout.split_column(Layout(name='header', size=3), Layout(name='top_section', size=12), Layout(name='buffer', size=6), Layout(name='bottom_section', size=10))
    layout['top_section'].split_row(Layout(name='queue', ratio=3), Layout(name='memory', ratio=2))
    layout['bottom_section'].split_row(Layout(name='left', ratio=1), Layout(name='center', ratio=1), Layout(name='right', ratio=1))
    return layout

def update_display(layout: Layout, monitor: DeepOrchestratorMonitor):
    """Update the display with current state"""
    layout['header'].update(Panel('🚀 Deep Orchestrator - Assignment Grader', style='bold blue'))
    layout['buffer'].update('')
    queue_plan_content = Columns([monitor.get_queue_tree(), monitor.get_plan_table()], padding=(1, 2))
    layout['queue'].update(queue_plan_content)
    layout['memory'].update(monitor.get_memory_panel())
    layout['left'].update(monitor.get_budget_table())
    layout['center'].update(monitor.get_status_summary())
    right_content = Layout()
    right_content.split_column(Layout(monitor.get_policy_panel(), size=7), Layout(monitor.get_agents_table(), size=10))
    layout['right'].update(right_content)

