# Cluster 47

def pershing_risk_management_analysis(metrics: list, financial_line_items: list, market_cap: float) -> dict:
    """Pershing Square's risk management for concentrated positions"""
    score = 0
    details = []
    risk_factors = []
    if not metrics or not financial_line_items or (not market_cap):
        return {'score': 0, 'details': 'No risk data'}
    concentration_risk = assess_concentration_risk(market_cap)
    if concentration_risk < 0.3:
        score += 2
        details.append('Low concentration risk')
    elif concentration_risk > 0.7:
        risk_factors.append('High concentration risk')
        details.append('High concentration risk identified')
    liquidity_risk = assess_large_position_liquidity_risk(market_cap)
    if liquidity_risk < 0.2:
        score += 2
        details.append('Low liquidity risk for large positions')
    elif liquidity_risk > 0.6:
        risk_factors.append('Liquidity constraints')
    campaign_risk = assess_activist_campaign_risks(financial_line_items)
    if campaign_risk < 0.4:
        score += 1.5
        details.append('Low activist campaign risk')
    elif campaign_risk > 0.7:
        risk_factors.append('High campaign execution risk')
    regulatory_risk = assess_regulatory_and_legal_risks(market_cap)
    if regulatory_risk < 0.3:
        score += 1.5
        details.append('Low regulatory risk')
    elif regulatory_risk > 0.6:
        risk_factors.append('Regulatory concerns')
    reputational_risk = assess_reputational_risk_exposure(financial_line_items)
    if reputational_risk < 0.4:
        score += 1
        details.append('Low reputational risk')
    elif reputational_risk > 0.7:
        risk_factors.append('Reputational risk exposure')
    downside_protection = calculate_downside_protection(financial_line_items, market_cap)
    if downside_protection > 0.6:
        score += 2
        details.append('Strong downside protection')
    if risk_factors:
        details.append(f'Risk factors: {'; '.join(risk_factors)}')
    return {'score': score, 'details': '; '.join(details), 'risk_factors': risk_factors}

def calculate_downside_protection(financial_line_items: list, market_cap: float) -> dict:
    """Eveillard's downside protection calculation"""
    score = 0
    details = []
    if not financial_line_items or not market_cap:
        return {'score': 0, 'details': 'No downside protection data'}
    latest = financial_line_items[0]
    current_assets = latest.current_assets if latest.current_assets else 0
    cash = latest.cash_and_equivalents if latest.cash_and_equivalents else 0
    conservative_assets = cash + (current_assets - cash) * 0.7
    current_liabilities = latest.current_liabilities if latest.current_liabilities else 0
    total_debt = latest.total_debt if latest.total_debt else 0
    net_liquidation_value = conservative_assets - current_liabilities - total_debt
    if net_liquidation_value > 0:
        liquidation_ratio = net_liquidation_value / market_cap
        if liquidation_ratio > 0.7:
            score += 4
            details.append(f'Excellent liquidation protection: {liquidation_ratio:.1%}')
        elif liquidation_ratio > 0.4:
            score += 2
            details.append(f'Good liquidation protection: {liquidation_ratio:.1%}')
        elif liquidation_ratio > 0.2:
            score += 1
            details.append(f'Some liquidation protection: {liquidation_ratio:.1%}')
    if latest.shareholders_equity:
        book_protection = latest.shareholders_equity / market_cap
        if book_protection > 1.0:
            score += 2
            details.append(f'Trading below book value: {book_protection:.1f}x')
        elif book_protection > 0.8:
            score += 1
            details.append(f'Near book value protection: {book_protection:.1f}x')
    earnings = [item.net_income for item in financial_line_items if item.net_income]
    if earnings:
        worst_earnings = min(earnings)
        if worst_earnings > 0:
            score += 2
            details.append('No historical losses provide earnings floor')
        elif worst_earnings > -market_cap * 0.03:
            score += 1
            details.append('Limited historical downside from worst performance')
    return {'score': score, 'details': '; '.join(details)}

