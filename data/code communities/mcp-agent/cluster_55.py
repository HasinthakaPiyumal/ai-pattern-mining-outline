# Cluster 55

def display_token_usage(usage: TokenUsage, label: str='Token Usage'):
    """Display token usage in a formatted way."""
    print(f'\n{label}:')
    print(f'  Total tokens: {usage.total_tokens:,}')
    print(f'  Input tokens: {usage.input_tokens:,}')
    print(f'  Output tokens: {usage.output_tokens:,}')

def display_node_tree(node: TokenNode, indent: str='', is_last: bool=True, context: Context | None=None, skip_empty: bool=True):
    """Display a node and its children with aggregate token usage and cost."""
    connector = '└── ' if is_last else '├── '
    usage = node.get_usage()
    cost = node.get_cost() if hasattr(node, 'get_cost') else 0.0
    if skip_empty and usage.total_tokens == 0:
        return
    cost_str = f' (${cost:.4f})' if cost and cost > 0 else ''
    print(f'{indent}{connector}{node.name} [{node.node_type}]')
    print(f'{indent}{('    ' if is_last else '│   ')}├─ Total: {usage.total_tokens:,} tokens{cost_str}')
    print(f'{indent}{('    ' if is_last else '│   ')}├─ Input: {usage.input_tokens:,}')
    print(f'{indent}{('    ' if is_last else '│   ')}└─ Output: {usage.output_tokens:,}')
    if node.usage.model_name:
        model_str = node.usage.model_name
        if node.usage.model_info and node.usage.model_info.provider:
            model_str += f' ({node.usage.model_info.provider})'
        print(f'{indent}{('    ' if is_last else '│   ')}   Model: {model_str}')
    if node.children:
        print(f'{indent}{('    ' if is_last else '│   ')}')
        child_indent = indent + ('    ' if is_last else '│   ')
        for i, child in enumerate(node.children):
            display_node_tree(child, child_indent, i == len(node.children) - 1, context=context, skip_empty=skip_empty)

def display_node_tree(node: TokenNode, indent: str='', is_last: bool=True, context: Context | None=None, skip_empty: bool=True):
    """Display a node and its children with aggregate token usage and cost."""
    connector = '└── ' if is_last else '├── '
    usage = node.get_usage()
    cost = node.get_cost() if hasattr(node, 'get_cost') else 0.0
    if skip_empty and usage.total_tokens == 0:
        return
    cost_str = f' (${cost:.4f})' if cost and cost > 0 else ''
    print(f'{indent}{connector}{node.name} [{node.node_type}]')
    print(f'{indent}{('    ' if is_last else '│   ')}├─ Total: {usage.total_tokens:,} tokens{cost_str}')
    print(f'{indent}{('    ' if is_last else '│   ')}├─ Input: {usage.input_tokens:,}')
    print(f'{indent}{('    ' if is_last else '│   ')}└─ Output: {usage.output_tokens:,}')
    if node.usage.model_name:
        model_str = node.usage.model_name
        if node.usage.model_info and node.usage.model_info.provider:
            model_str += f' ({node.usage.model_info.provider})'
        print(f'{indent}{('    ' if is_last else '│   ')}   Model: {model_str}')
    if node.children:
        print(f'{indent}{('    ' if is_last else '│   ')}')
        child_indent = indent + ('    ' if is_last else '│   ')
        for i, child in enumerate(node.children):
            display_node_tree(child, child_indent, i == len(node.children) - 1, context=context, skip_empty=skip_empty)

