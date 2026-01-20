# Cluster 44

@app.route('/analyze/<ticker>')
def analyze_ticker(ticker):
    """Web endpoint to analyze a ticker with all hedge funds"""
    try:
        provider = request.args.get('provider', 'openrouter')
        model = request.args.get('model', 'anthropic/claude-3.5-sonnet')
        llm_adapter = HedgeFundLLMAdapter(provider=provider, model=model)
        selected_funds = request.args.get('funds', 'bridgewater,aqr,renaissance').split(',')
        results = {}
        for fund_name in selected_funds:
            if fund_name in AVAILABLE_FUNDS:
                fund_agent = AVAILABLE_FUNDS[fund_name]
                fund_module = importlib.import_module(fund_agent.__module__)
                fund_module.call_llm = llm_adapter.call_llm
                state = create_mock_state([ticker])
                fund_agent(state, f'{fund_name}_web')
                results[fund_name] = state['data']['analyst_signals'][f'{fund_name}_web'][ticker]
        consensus = calculate_hedge_fund_consensus(results)
        return jsonify({'ticker': ticker, 'consensus': consensus, 'individual_analyses': results, 'timestamp': '2024-12-31T00:00:00Z'})
    except Exception as e:
        return (jsonify({'error': str(e)}), 500)

def create_mock_state(tickers: list):
    """Create mock state for testing"""
    return {'data': {'tickers': tickers, 'end_date': '2024-12-31', 'analyst_signals': {}}, 'metadata': {'show_reasoning': True}}

def calculate_hedge_fund_consensus(signals: dict):
    """Calculate consensus from multiple hedge fund signals"""
    if not signals:
        return {'consensus': 'neutral', 'confidence': 0}
    bullish_count = sum((1 for signal in signals.values() if signal.get('signal') == 'bullish'))
    bearish_count = sum((1 for signal in signals.values() if signal.get('signal') == 'bearish'))
    neutral_count = len(signals) - bullish_count - bearish_count
    total_funds = len(signals)
    if bullish_count > total_funds * 0.6:
        consensus = 'bullish'
        confidence = bullish_count / total_funds
    elif bearish_count > total_funds * 0.6:
        consensus = 'bearish'
        confidence = bearish_count / total_funds
    else:
        consensus = 'neutral'
        confidence = max(bullish_count, bearish_count, neutral_count) / total_funds
    avg_confidence = sum((signal.get('confidence', 0) for signal in signals.values())) / total_funds
    return {'consensus': consensus, 'consensus_strength': confidence, 'average_confidence': avg_confidence, 'fund_breakdown': {'bullish': bullish_count, 'bearish': bearish_count, 'neutral': neutral_count}, 'individual_signals': signals}

