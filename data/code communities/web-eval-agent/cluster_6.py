# Cluster 6

def format_agent_result(result_str: str, url: str, task: str, console_logs=None, network_requests=None) -> str:
    """Format the agent result in a readable way with emojis.
    
    Args:
        result_str: Raw string representation of the agent result
        url: The URL that was evaluated
        task: The task that was executed
        console_logs: Collected console logs from the browser
        network_requests: Collected network requests from the browser
        
    Returns:
        str: Formatted result with steps and conclusion
    """
    formatted = f'📊 Web Evaluation Report for {url} complete!\n'
    formatted += f'📝 Completed Task: {task}\n\n'
    if result_str.startswith('Error:'):
        return f'{formatted}❌ {result_str}'
    agent_steps_timeline = []

    def format_error_list(items, item_formatter):
        """Format a list of error items with character limit.
        
        Args:
            items: List of error items to format
            item_formatter: Function that takes (index, item) and returns a formatted string
            
        Returns:
            str: Formatted error list with potential truncation
        """
        if not items:
            return ' No items found.\n'
        result = f' ({len(items)} items)\n'
        all_items_text = ''
        for i, item in enumerate(items):
            item_line = item_formatter(i, item)
            all_items_text += item_line
        if len(all_items_text) > MAX_ERROR_OUTPUT_CHARS:
            truncated_text = all_items_text[:MAX_ERROR_OUTPUT_CHARS]
            last_newline = truncated_text.rfind('\n')
            if last_newline > MAX_ERROR_OUTPUT_CHARS * 0.9:
                truncated_text = truncated_text[:last_newline + 1]
            result += truncated_text
            result += f'  ... [Output truncated, {len(all_items_text) - len(truncated_text)} more characters not shown]\n'
        else:
            result += all_items_text
        return result
    try:
        if 'all_results=[' in result_str:
            results_part = result_str.split('all_results=[')[1].split(']')[0]
            action_results = results_part.split('ActionResult(')
            action_results = [r for r in action_results if r.strip()]
            for action in action_results:
                if 'is_done=True' in action:
                    if 'success=False' in action:
                        continue
            formatted += '🔍 Agent Steps:\n'
            earliest_browser_time = None
            latest_browser_time = None
            if console_logs:
                for log in console_logs:
                    timestamp = log.get('timestamp', 0)
                    if timestamp > 0:
                        if earliest_browser_time is None or timestamp < earliest_browser_time:
                            earliest_browser_time = timestamp
                        if latest_browser_time is None or timestamp > latest_browser_time:
                            latest_browser_time = timestamp
            if network_requests:
                for req in network_requests:
                    timestamp = req.get('timestamp', 0)
                    if timestamp > 0:
                        if earliest_browser_time is None or timestamp < earliest_browser_time:
                            earliest_browser_time = timestamp
                        if latest_browser_time is None or timestamp > latest_browser_time:
                            latest_browser_time = timestamp
                    resp_timestamp = req.get('response_timestamp', 0)
                    if resp_timestamp > 0:
                        if latest_browser_time is None or resp_timestamp > latest_browser_time:
                            latest_browser_time = resp_timestamp
            current_time = time.time()
            if earliest_browser_time and latest_browser_time:
                step_base_time = latest_browser_time + 2
                step_interval = 5
            else:
                step_base_time = current_time - len(action_results) * 5
                step_interval = 5
            for i, action in enumerate(action_results):
                if 'extracted_content=' in action:
                    content_part = action.split('extracted_content=')[1].split(',')[0]
                    content = content_part.strip('\'"')
                    if content == 'None':
                        continue
                    step_timestamp = step_base_time + i * step_interval
                    if 'error=' in action and 'error=None' not in action:
                        error_part = action.split('error=')[1].split(',')[0]
                        error = error_part.strip('\'"')
                        if error != 'None':
                            error_content = f'❌ Step {i + 1}: {error}'
                            formatted += f'  {error_content}\n'
                            agent_steps_timeline.append({'type': 'agent_error', 'text': error_content, 'timestamp': step_timestamp})
                            continue
                    is_final_message = 'is_done=True' in action
                    if not content.startswith(('🔗', '🖱️', '⌨️', '🔍', '✅', '❌', '⚠️', '🏁')):
                        if is_final_message:
                            content = f'🏁 {content}'
                        else:
                            content = f'✅ {content}'
                    if content.startswith('✅') and is_final_message:
                        content = '🏁' + content[1:]
                    if not is_final_message:
                        formatted_line = f'  📍 Step {i + 1}: {content}'
                        timeline_content = f'📍 Step {i + 1}: {content}'
                    else:
                        formatted_line = f'  {content}'
                        timeline_content = content
                    formatted += formatted_line + '\n'
                    agent_steps_timeline.append({'type': 'agent_step', 'text': timeline_content, 'timestamp': step_timestamp})
        conclusion = ''
        if "'done':" in result_str or '"done":' in result_str:
            done_match = None
            if "'done':" in result_str:
                done_parts = result_str.split("'done':")[1].split('}')[0]
                done_match = done_parts
            elif '"done":' in result_str:
                done_parts = result_str.split('"done":')[1].split('}')[0]
                done_match = done_parts
            if done_match:
                if "'text':" in done_match:
                    text_part = done_match.split("'text':")[1].split(',')[0]
                    conclusion = text_part.strip('\' "')
                elif '"text":' in done_match:
                    text_part = done_match.split('"text":')[1].split(',')[0]
                    conclusion = text_part.strip('\' "')
                if "'success': False" in done_match or '"success": False' in done_match:
                    pass
        if not conclusion and 'is_done=True' in result_str:
            for action in action_results:
                if 'is_done=True' in action and 'extracted_content=' in action:
                    content = action.split('extracted_content=')[1].split(',')[0].strip('\'"')
                    if content and content != 'None':
                        conclusion = content
                        break
        if conclusion:
            formatted += f'\n📋 Conclusion:\n{conclusion}\n'
            if agent_steps_timeline:
                conclusion_timestamp = agent_steps_timeline[-1]['timestamp'] + 2
            else:
                conclusion_timestamp = time.time()
            agent_steps_timeline.append({'type': 'conclusion', 'text': f'📋 Conclusion: {conclusion}', 'timestamp': conclusion_timestamp})
        console_errors = []
        if console_logs:
            for log in console_logs:
                if log.get('type') == 'error':
                    console_errors.append(log.get('text', 'Unknown error'))
        if console_errors:
            formatted += '\n🔴 Console Errors:'
            formatted += format_error_list(console_errors, lambda i, error: f'  {i + 1}. {error}\n')
        failed_requests = []
        if network_requests:
            for req in network_requests:
                is_xhr = req.get('resourceType') == 'xhr' or req.get('resourceType') == 'fetch'
                status = req.get('response_status')
                if is_xhr and status and (status >= 400):
                    failed_requests.append({'url': req.get('url', 'Unknown URL'), 'method': req.get('method', 'GET'), 'status': status})
        if failed_requests:
            formatted += '\n❌ Failed Network Requests:'
            formatted += format_error_list(failed_requests, lambda i, req: f'  {i + 1}. {req['method']} {req['url']} - Status: {req['status']}\n')
        all_console_logs = []
        if console_logs:
            all_console_logs = list(console_logs)
        formatted += '\n🖥️ All Console Logs:'
        formatted += format_error_list(all_console_logs, lambda i, log: f'  {i + 1}. [{log.get('type', 'log')}] {log.get('text', 'Unknown message')}\n')
        all_network_requests = []
        if network_requests:
            all_network_requests = list(network_requests)
        formatted += '\n🌐 All Network Requests:'
        formatted += format_error_list(all_network_requests, lambda i, req: f'  {i + 1}. {req.get('method', 'GET')} {req.get('url', 'Unknown URL')} - Status: {req.get('response_status', 'N/A')}\n')
        all_events = []
        for log in all_console_logs:
            all_events.append({'type': 'console', 'subtype': log.get('type', 'log'), 'text': log.get('text', 'Unknown message'), 'timestamp': log.get('timestamp', 0)})
        for req in all_network_requests:
            all_events.append({'type': 'network_request', 'method': req.get('method', 'GET'), 'url': req.get('url', 'Unknown URL'), 'timestamp': req.get('timestamp', 0)})
            if 'response_timestamp' in req:
                all_events.append({'type': 'network_response', 'method': req.get('method', 'GET'), 'url': req.get('url', 'Unknown URL'), 'status': req.get('response_status', 'N/A'), 'timestamp': req.get('response_timestamp', 0)})
        all_events.extend(agent_steps_timeline)
        all_events.sort(key=lambda x: x.get('timestamp', 0))
        formatted += '\n\n⏱️ Chronological Timeline of All Events:\n'
        timeline_text = ''
        for event in all_events:
            event_type = event.get('type')
            timestamp = event.get('timestamp', 0)
            from datetime import datetime
            time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]
            if event_type == 'console':
                subtype = event.get('subtype', 'log')
                text = event.get('text', '')
                emoji = '❌' if subtype == 'error' else '⚠️' if subtype == 'warning' else '🖥️'
                timeline_text += f'  {time_str} {emoji} Console [{subtype}]: {text}\n'
            elif event_type == 'network_request':
                method = event.get('method', 'GET')
                url = event.get('url', '')
                timeline_text += f'  {time_str} ➡️ Network Request: {method} {url}\n'
            elif event_type == 'network_response':
                method = event.get('method', 'GET')
                url = event.get('url', '')
                status = event.get('status', 'N/A')
                status_emoji = '❌' if str(status).startswith(('4', '5')) else '✅'
                timeline_text += f'  {time_str} ⬅️ Network Response: {method} {url} - Status: {status} {status_emoji}\n'
            elif event_type == 'agent_step':
                text = event.get('text', '')
                timeline_text += f'  {time_str} 🤖 {text}\n'
            elif event_type == 'agent_error':
                text = event.get('text', '')
                timeline_text += f'  {time_str} 🤖 Agent Error: {text}\n'
            elif event_type == 'conclusion':
                text = event.get('text', '')
                timeline_text += f'  {time_str} 🤖 {text}\n'
        if len(timeline_text) > MAX_TIMELINE_CHARS:
            truncated_text = timeline_text[:MAX_TIMELINE_CHARS]
            last_newline = truncated_text.rfind('\n')
            if last_newline > MAX_TIMELINE_CHARS * 0.9:
                truncated_text = truncated_text[:last_newline + 1]
            formatted += truncated_text
            formatted += f'  ... [Timeline truncated, {len(timeline_text) - len(truncated_text)} more characters not shown]\n'
        else:
            formatted += timeline_text
    except Exception as e:
        return f'{formatted}⚠️ Result parsing failed: {e}\nRaw result: {result_str[:10000]}...\n'
    return formatted

def format_error_list(items, item_formatter):
    """Format a list of error items with character limit.
        
        Args:
            items: List of error items to format
            item_formatter: Function that takes (index, item) and returns a formatted string
            
        Returns:
            str: Formatted error list with potential truncation
        """
    if not items:
        return ' No items found.\n'
    result = f' ({len(items)} items)\n'
    all_items_text = ''
    for i, item in enumerate(items):
        item_line = item_formatter(i, item)
        all_items_text += item_line
    if len(all_items_text) > MAX_ERROR_OUTPUT_CHARS:
        truncated_text = all_items_text[:MAX_ERROR_OUTPUT_CHARS]
        last_newline = truncated_text.rfind('\n')
        if last_newline > MAX_ERROR_OUTPUT_CHARS * 0.9:
            truncated_text = truncated_text[:last_newline + 1]
        result += truncated_text
        result += f'  ... [Output truncated, {len(all_items_text) - len(truncated_text)} more characters not shown]\n'
    else:
        result += all_items_text
    return result

