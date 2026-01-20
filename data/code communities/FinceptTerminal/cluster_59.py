# Cluster 59

def global_macro_department(metrics: list, financial_line_items: list) -> dict:
    """Citadel's global macro department analysis"""
    score = 0
    details = []
    if not metrics or not financial_line_items:
        return {'score': 0, 'details': 'No macro data'}
    latest = financial_line_items[0]
    if latest.total_debt and latest.total_assets:
        debt_ratio = latest.total_debt / latest.total_assets
        if debt_ratio < 0.3:
            score += 2
            details.append(f'Low interest rate risk: {debt_ratio:.1%} debt/assets')
        elif debt_ratio > 0.6:
            score -= 1
            details.append(f'High interest rate sensitivity: {debt_ratio:.1%} debt/assets')
    international_revenue_proxy = analyze_international_exposure(financial_line_items)
    if 0.2 <= international_revenue_proxy <= 0.6:
        score += 1
        details.append('Balanced international exposure')
    pricing_power = analyze_pricing_power(financial_line_items)
    if pricing_power > 0.7:
        score += 2
        details.append('Strong pricing power - inflation hedge')
    recession_resilience = analyze_recession_performance(financial_line_items)
    if recession_resilience > 0.6:
        score += 1
        details.append('Good recession resilience')
    commodity_sensitivity = analyze_commodity_exposure(financial_line_items)
    if commodity_sensitivity < 0.3:
        score += 1
        details.append('Low commodity price sensitivity')
    return {'score': score, 'details': '; '.join(details)}

def analyze_international_exposure(financial_line_items: list) -> float:
    """Estimate international exposure (simplified)"""
    return 0.4

