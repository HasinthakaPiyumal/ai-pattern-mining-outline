# Cluster 14

def main():
    """Main function to process hook input and decide on action."""
    try:
        input_data = sys.stdin.read()
        logger.info('Hook triggered')
        logger.info(f'Raw input: {input_data}')
        if not input_data.strip():
            logger.warning('Empty input received')
            print('{"action": "none"}')
            return
        try:
            hook_data = json.loads(input_data)
        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse JSON input: {e}')
            print('{"action": "none"}')
            return
        hook_event = hook_data.get('hook_event_name', '')
        message = hook_data.get('message') or hook_data.get('content', '')
        tool_name = hook_data.get('toolName', '')
        logger.info(f'Parsed - Event: {hook_event}, Message: {message}, Tool: {tool_name}')
        if hook_event == 'Stop':
            logger.info('Stop event detected - sending Continue')
            continue_message = 'Observe your environment by accessing all available resources and continue development'
            if send_to_tmux(continue_message):
                print(json.dumps({'action': 'continue', 'response': continue_message}))
                return
        if message:
            auto_continue_patterns = ['factorio.*complete', 'agent.*stopped', 'observation.*ready', 'waiting for input', 'Claude is waiting', 'execution complete', 'task finished']
            message_lower = message.lower()
            for pattern in auto_continue_patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    logger.info(f'Auto-continue condition met in message: {pattern}')
                    continue_message = 'Observe your environment by accessing all available resources and continue development'
                    if send_to_tmux(continue_message):
                        print(json.dumps({'action': 'continue', 'response': continue_message}))
                        return
                    break
        if message and ('warning' in message.lower() or 'error' in message.lower()):
            warning_message = f'Warning detected: {message}'
            logger.info(f'Warning detected, sending to Claude: {warning_message}')
            if send_to_tmux(warning_message):
                print(json.dumps({'action': 'warning', 'response': warning_message}))
                return
        print('{"action": "none"}')
    except Exception as e:
        logger.error(f'Unexpected error in main: {e}')
        print('{"action": "none"}')

def send_to_tmux(message: str) -> bool:
    """Send message to Claude Code tmux session."""
    try:
        result = subprocess.run(['tmux', 'has-session', '-t', 'claude-code'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error('No claude-code tmux session found')
            return False
        logger.info('Found claude-code tmux session')
        subprocess.run(['tmux', 'send-keys', '-t', 'claude-code', message, 'C-m'], check=True)
        subprocess.run(['tmux', 'send-keys', '-t', 'claude-code', 'C-m'], check=True)
        logger.info(f"Sent '{message}' with C-m to claude-code session")
        try:
            capture_result = subprocess.run(['tmux', 'capture-pane', '-t', 'claude-code', '-p'], capture_output=True, text=True, check=True)
            last_lines = '\n'.join(capture_result.stdout.split('\n')[-5:])
            logger.info(f'Last 5 lines from tmux:\n{last_lines}')
        except subprocess.CalledProcessError:
            logger.warning('Could not capture tmux pane content')
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f'Failed to send to tmux: {e}')
        return False
    except Exception as e:
        logger.error(f'Unexpected error sending to tmux: {e}')
        return False

