# Cluster 49

def print_trading_output(result: dict) -> None:
    """
    Print formatted trading results with colored tables for multiple tickers.

    Args:
        result (dict): Dictionary containing decisions and analyst signals for multiple tickers
    """
    decisions = result.get('decisions')
    if not decisions:
        print(f'{Fore.RED}No trading decisions available{Style.RESET_ALL}')
        return
    for ticker, decision in decisions.items():
        print(f'\n{Fore.WHITE}{Style.BRIGHT}Analysis for {Fore.CYAN}{ticker}{Style.RESET_ALL}')
        print(f'{Fore.WHITE}{Style.BRIGHT}{'=' * 50}{Style.RESET_ALL}')
        table_data = []
        for agent, signals in result.get('analyst_signals', {}).items():
            if ticker not in signals:
                continue
            if agent == 'risk_management_agent':
                continue
            signal = signals[ticker]
            agent_name = agent.replace('_agent', '').replace('_', ' ').title()
            signal_type = signal.get('signal', '').upper()
            confidence = signal.get('confidence', 0)
            signal_color = {'BULLISH': Fore.GREEN, 'BEARISH': Fore.RED, 'NEUTRAL': Fore.YELLOW}.get(signal_type, Fore.WHITE)
            reasoning_str = ''
            if 'reasoning' in signal and signal['reasoning']:
                reasoning = signal['reasoning']
                if isinstance(reasoning, str):
                    reasoning_str = reasoning
                elif isinstance(reasoning, dict):
                    reasoning_str = json.dumps(reasoning, indent=2)
                else:
                    reasoning_str = str(reasoning)
                wrapped_reasoning = ''
                current_line = ''
                max_line_length = 60
                for word in reasoning_str.split():
                    if len(current_line) + len(word) + 1 > max_line_length:
                        wrapped_reasoning += current_line + '\n'
                        current_line = word
                    elif current_line:
                        current_line += ' ' + word
                    else:
                        current_line = word
                if current_line:
                    wrapped_reasoning += current_line
                reasoning_str = wrapped_reasoning
            table_data.append([f'{Fore.CYAN}{agent_name}{Style.RESET_ALL}', f'{signal_color}{signal_type}{Style.RESET_ALL}', f'{Fore.WHITE}{confidence}%{Style.RESET_ALL}', f'{Fore.WHITE}{reasoning_str}{Style.RESET_ALL}'])
        table_data = sort_agent_signals(table_data)
        print(f'\n{Fore.WHITE}{Style.BRIGHT}AGENT ANALYSIS:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]')
        print(tabulate(table_data, headers=[f'{Fore.WHITE}Agent', 'Signal', 'Confidence', 'Reasoning'], tablefmt='grid', colalign=('left', 'center', 'right', 'left')))
        action = decision.get('action', '').upper()
        action_color = {'BUY': Fore.GREEN, 'SELL': Fore.RED, 'HOLD': Fore.YELLOW, 'COVER': Fore.GREEN, 'SHORT': Fore.RED}.get(action, Fore.WHITE)
        reasoning = decision.get('reasoning', '')
        wrapped_reasoning = ''
        if reasoning:
            current_line = ''
            max_line_length = 60
            for word in reasoning.split():
                if len(current_line) + len(word) + 1 > max_line_length:
                    wrapped_reasoning += current_line + '\n'
                    current_line = word
                elif current_line:
                    current_line += ' ' + word
                else:
                    current_line = word
            if current_line:
                wrapped_reasoning += current_line
        decision_data = [['Action', f'{action_color}{action}{Style.RESET_ALL}'], ['Quantity', f'{action_color}{decision.get('quantity')}{Style.RESET_ALL}'], ['Confidence', f'{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}'], ['Reasoning', f'{Fore.WHITE}{wrapped_reasoning}{Style.RESET_ALL}']]
        print(f'\n{Fore.WHITE}{Style.BRIGHT}TRADING DECISION:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]')
        print(tabulate(decision_data, tablefmt='grid', colalign=('left', 'left')))
    print(f'\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}')
    portfolio_data = []
    portfolio_manager_reasoning = None
    for ticker, decision in decisions.items():
        if decision.get('reasoning'):
            portfolio_manager_reasoning = decision.get('reasoning')
            break
    for ticker, decision in decisions.items():
        action = decision.get('action', '').upper()
        action_color = {'BUY': Fore.GREEN, 'SELL': Fore.RED, 'HOLD': Fore.YELLOW, 'COVER': Fore.GREEN, 'SHORT': Fore.RED}.get(action, Fore.WHITE)
        portfolio_data.append([f'{Fore.CYAN}{ticker}{Style.RESET_ALL}', f'{action_color}{action}{Style.RESET_ALL}', f'{action_color}{decision.get('quantity')}{Style.RESET_ALL}', f'{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}'])
    headers = [f'{Fore.WHITE}Ticker', 'Action', 'Quantity', 'Confidence']
    print(tabulate(portfolio_data, headers=headers, tablefmt='grid', colalign=('left', 'center', 'right', 'right')))
    if portfolio_manager_reasoning:
        reasoning_str = ''
        if isinstance(portfolio_manager_reasoning, str):
            reasoning_str = portfolio_manager_reasoning
        elif isinstance(portfolio_manager_reasoning, dict):
            reasoning_str = json.dumps(portfolio_manager_reasoning, indent=2)
        else:
            reasoning_str = str(portfolio_manager_reasoning)
        wrapped_reasoning = ''
        current_line = ''
        max_line_length = 60
        for word in reasoning_str.split():
            if len(current_line) + len(word) + 1 > max_line_length:
                wrapped_reasoning += current_line + '\n'
                current_line = word
            elif current_line:
                current_line += ' ' + word
            else:
                current_line = word
        if current_line:
            wrapped_reasoning += current_line
        print(f'\n{Fore.WHITE}{Style.BRIGHT}Portfolio Strategy:{Style.RESET_ALL}')
        print(f'{Fore.CYAN}{wrapped_reasoning}{Style.RESET_ALL}')

def sort_agent_signals(signals):
    """Sort agent signals in a consistent order."""
    analyst_order = {display: idx for idx, (display, _) in enumerate(ANALYST_ORDER)}
    analyst_order['Risk Management'] = len(ANALYST_ORDER)
    return sorted(signals, key=lambda x: analyst_order.get(x[0], 999))

