# Cluster 54

def statistical_arbitrage_analysis(metrics: list, financial_line_items: list) -> dict:
    """Renaissance's statistical arbitrage models"""
    score = 0
    details = []
    arbitrage_signals = {}
    if not metrics or not financial_line_items:
        return {'score': 0, 'details': 'No statistical arbitrage data'}
    pairs_signal = calculate_pairs_trading_signals(metrics, financial_line_items)
    arbitrage_signals['pairs'] = pairs_signal
    score += pairs_signal * 2
    mispricing_score = detect_statistical_mispricing(metrics, financial_line_items)
    arbitrage_signals['mispricing'] = mispricing_score
    score += mispricing_score * 2.5
    regime_change = detect_regime_changes(financial_line_items)
    arbitrage_signals['regime_change'] = regime_change
    score += regime_change * 1.5
    microstructure_score = analyze_market_microstructure(metrics)
    arbitrage_signals['microstructure'] = microstructure_score
    score += microstructure_score * 1
    details.append(f'Pairs trading signal: {pairs_signal:.2f}')
    details.append(f'Mispricing detection: {mispricing_score:.2f}')
    details.append(f'Regime change signal: {regime_change:.2f}')
    return {'score': min(score, 10), 'details': '; '.join(details), 'arbitrage_signals': arbitrage_signals}

def analyze_market_microstructure(financial_line_items: list, market_cap: float) -> float:
    """Analyze market microstructure opportunities"""
    liquidity_score = min(market_cap / 10000000000, 1.0)
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if revenues:
        volatility = calculate_coefficient_of_variation(revenues)
        volatility_score = min(volatility, 1.0)
        return (liquidity_score + volatility_score) / 2
    return liquidity_score

