# Cluster 46

def calculate_quality_factor(metrics: list, financial_line_items: list) -> float:
    """Calculate AQR's quality factor score"""
    if not metrics:
        return 0.5
    latest = metrics[0]
    quality_components = []
    if latest.return_on_equity:
        roe_score = min(latest.return_on_equity / 0.2, 1.0)
        quality_components.append(roe_score)
    if latest.return_on_assets:
        roa_score = min(latest.return_on_assets / 0.1, 1.0)
        quality_components.append(roa_score)
    earnings = [item.net_income for item in financial_line_items if item.net_income]
    if earnings:
        earnings_stability = 1 - min(calculate_coefficient_of_variation(earnings), 1.0)
        quality_components.append(earnings_stability)
    if latest.debt_to_equity is not None:
        debt_quality = max(0, 1 - latest.debt_to_equity)
        quality_components.append(debt_quality)
    return sum(quality_components) / len(quality_components) if quality_components else 0.5

def calculate_coefficient_of_variation(data: list) -> float:
    """Calculate coefficient of variation"""
    if not data or len(data) < 2:
        return 1.0
    mean_val = sum(data) / len(data)
    if mean_val == 0:
        return 1.0
    variance = sum(((x - mean_val) ** 2 for x in data)) / len(data)
    std_dev = variance ** 0.5
    return std_dev / abs(mean_val)

def calculate_low_volatility_factor(financial_line_items: list) -> float:
    """Calculate low volatility factor score"""
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    earnings = [item.net_income for item in financial_line_items if item.net_income]
    if not revenues and (not earnings):
        return 0.5
    volatility_scores = []
    if revenues:
        revenue_vol = calculate_coefficient_of_variation(revenues)
        volatility_scores.append(max(0, 1 - revenue_vol))
    if earnings:
        earnings_vol = calculate_coefficient_of_variation(earnings)
        volatility_scores.append(max(0, 1 - earnings_vol))
    return sum(volatility_scores) / len(volatility_scores) if volatility_scores else 0.5

def analyze_risk_budgeting(financial_line_items: list) -> float:
    """Analyze risk budgeting allocation"""
    earnings = [item.net_income for item in financial_line_items if item.net_income]
    if earnings:
        earnings_cv = calculate_coefficient_of_variation(earnings)
        return max(0, 1 - earnings_cv)
    return 0.5

def analyze_competitive_moat_strength(metrics: list, financial_line_items: list) -> float:
    """Analyze competitive moat strength"""
    if not metrics or not financial_line_items:
        return 0.5
    moat_indicators = 0
    total_indicators = 0
    margins = [item.operating_margin for item in financial_line_items if item.operating_margin]
    if margins:
        avg_margin = sum(margins) / len(margins)
        margin_stability = 1 - (max(margins) - min(margins)) / max(margins) if max(margins) > 0 else 0
        if avg_margin > 0.15 and margin_stability > 0.7:
            moat_indicators += 1
        total_indicators += 1
    if metrics:
        roic_vals = [m.return_on_invested_capital for m in metrics if m.return_on_invested_capital]
        if roic_vals:
            avg_roic = sum(roic_vals) / len(roic_vals)
            if avg_roic > 0.15:
                moat_indicators += 1
            total_indicators += 1
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if len(revenues) >= 5:
        revenue_volatility = calculate_coefficient_of_variation(revenues)
        if revenue_volatility < 0.2:
            moat_indicators += 1
        total_indicators += 1
    return moat_indicators / total_indicators if total_indicators > 0 else 0.5

def assess_public_narrative_strength(financial_line_items: list) -> float:
    """Assess strength of public narrative"""
    narrative_score = 0
    margins = [item.operating_margin for item in financial_line_items if item.operating_margin]
    if margins and len(margins) >= 3:
        margin_decline = margins[-1] - margins[0]
        if margin_decline > 0.03:
            narrative_score += 0.4
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if len(revenues) >= 3:
        revenue_stability = 1 - calculate_coefficient_of_variation(revenues)
        if revenue_stability > 0.7:
            narrative_score += 0.3
    latest = financial_line_items[0]
    if latest.total_assets and latest.total_debt:
        net_assets = latest.total_assets - latest.total_debt
        narrative_score += 0.3
    return narrative_score

def calculate_downside_protection(financial_line_items: list, market_cap: float) -> float:
    """Calculate downside protection"""
    if not financial_line_items or not market_cap:
        return 0.3
    protection_score = 0
    latest = financial_line_items[0]
    if latest.total_assets and latest.total_debt:
        net_assets = latest.total_assets - latest.total_debt
        if net_assets > market_cap * 0.8:
            protection_score += 0.4
    if latest.free_cash_flow and latest.free_cash_flow > 0:
        fcf_yield = latest.free_cash_flow / market_cap
        if fcf_yield > 0.08:
            protection_score += 0.3
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if revenues:
        revenue_stability = 1 - calculate_coefficient_of_variation(revenues)
        protection_score += revenue_stability * 0.3
    return protection_score

def model_volatility_patterns(financial_line_items: list) -> float:
    """Model volatility patterns"""
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if len(revenues) >= 4:
        volatility = calculate_coefficient_of_variation(revenues)
        return max(1 - volatility, 0)
    return 0.5

def engineer_financial_features(metrics: list, financial_line_items: list) -> dict:
    """Engineer features from financial data"""
    features = {}
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if len(revenues) >= 4:
        recent_momentum = revenues[0] / revenues[1] - 1
        historical_momentum = (revenues[1] / revenues[-1]) ** (1 / (len(revenues) - 2)) - 1
        features['revenue_momentum'] = recent_momentum - historical_momentum
    margins = [item.operating_margin for item in financial_line_items if item.operating_margin]
    if margins:
        margin_cv = calculate_coefficient_of_variation(margins)
        features['margin_stability'] = max(0, 1 - margin_cv)
    if metrics and len(revenues) >= 3:
        revenue_growth = (revenues[0] / revenues[-1]) ** (1 / len(revenues)) - 1
        roe = metrics[0].return_on_equity if metrics[0].return_on_equity else 0
        features['growth_quality'] = min(revenue_growth * roe * 10, 1.0)
    return features

def analyze_volatility_clustering(data: list) -> float:
    """Analyze volatility clustering"""
    if len(data) < 4:
        return 0
    volatilities = []
    for i in range(2, len(data)):
        recent_vol = abs(data[i] - data[i - 1])
        volatilities.append(recent_vol)
    if volatilities:
        return min(calculate_coefficient_of_variation(volatilities), 1.0)
    return 0

def estimate_volatility(financial_line_items: list) -> float:
    """Estimate volatility from financial data"""
    earnings = [item.net_income for item in financial_line_items if item.net_income]
    if earnings:
        return calculate_coefficient_of_variation(earnings)
    return 0.3

