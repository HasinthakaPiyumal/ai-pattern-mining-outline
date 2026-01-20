# Cluster 45

def example_standalone_usage():
    """Example: Using hedge fund agents standalone"""
    mock_state = {'data': {'tickers': ['AAPL'], 'end_date': '2024-12-31', 'analyst_signals': {}}, 'metadata': {'show_reasoning': True}}
    import bridgewater_hedge_fund
    llm_adapter = HedgeFundLLMAdapter(provider='openrouter', model='anthropic/claude-3.5-sonnet')
    bridgewater_hedge_fund.call_llm = llm_adapter.call_llm
    result = bridgewater_associates_agent(mock_state, 'bridgewater_test')
    print('Bridgewater Analysis Results:')
    print(json.dumps(mock_state['data']['analyst_signals']['bridgewater_test'], indent=2))

def example_multi_hedge_fund_consensus():
    """Example: Get consensus from multiple hedge funds"""
    hedge_funds = [('bridgewater', bridgewater_associates_agent), ('renaissance', renaissance_technologies_agent), ('aqr', aqr_capital_hedge_fund_agent), ('elliott', elliott_management_hedge_fund_agent)]
    ticker = 'MSFT'
    all_signals = {}
    llm_adapter = HedgeFundLLMAdapter(provider='ollama', model='llama3.1:70b')
    for fund_name, fund_agent in hedge_funds:
        fund_module = fund_agent.__module__
        import importlib
        module = importlib.import_module(fund_module)
        module.call_llm = llm_adapter.call_llm
        state = create_mock_state([ticker])
        try:
            fund_agent(state, f'{fund_name}_analysis')
            all_signals[fund_name] = state['data']['analyst_signals'][f'{fund_name}_analysis'][ticker]
        except Exception as e:
            print(f'Error running {fund_name}: {e}')
            continue
    consensus = calculate_hedge_fund_consensus(all_signals)
    return consensus

