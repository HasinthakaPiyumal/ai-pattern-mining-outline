# Cluster 40

def _display_logs(log_entries: List[Dict[str, Any]], title: str='Logs', format: str='text') -> None:
    """Display logs in the specified format."""
    if not log_entries:
        return
    if format == 'json':
        cleaned_entries = [_clean_log_entry(entry) for entry in log_entries]
        print(json.dumps(cleaned_entries, indent=2))
    elif format == 'yaml':
        cleaned_entries = [_clean_log_entry(entry) for entry in log_entries]
        print(yaml.dump(cleaned_entries, default_flow_style=False))
    else:
        if title:
            console.print(f'[bold blue]{title}[/bold blue]\n')
        for entry in log_entries:
            _display_text_log_entry(entry)

def _clean_log_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Clean up a log entry for structured output formats."""
    cleaned_entry = entry.copy()
    cleaned_entry['severity'] = _parse_log_level(entry.get('level', 'INFO'))
    cleaned_entry['message'] = _clean_message(entry.get('message', ''))
    cleaned_entry.pop('level', None)
    return cleaned_entry

def _display_text_log_entry(entry: Dict[str, Any]) -> None:
    """Display a single log entry in text format."""
    timestamp = _format_timestamp(entry.get('timestamp', ''))
    raw_level = entry.get('level', 'INFO')
    level = _parse_log_level(raw_level)
    message = _clean_message(entry.get('message', ''))
    level_style = _get_level_style(level)
    message_text = Text.from_ansi(message)
    highlighter.highlight(message_text)
    console.print(f'[bright_black not bold]{timestamp}[/bright_black not bold] [{level_style}]{level:7}[/{level_style}] ', message_text)

def _display_log_entry(log_entry: Dict[str, Any], format: str='text') -> None:
    """Display a single log entry for streaming."""
    if format == 'json':
        cleaned_entry = _clean_log_entry(log_entry)
        print(json.dumps(cleaned_entry))
    elif format == 'yaml':
        cleaned_entry = _clean_log_entry(log_entry)
        print(yaml.dump([cleaned_entry], default_flow_style=False))
    else:
        _display_text_log_entry(log_entry)

