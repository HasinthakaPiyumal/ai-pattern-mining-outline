# Cluster 48

def assess_governance_improvement_opportunities(metrics: list, financial_line_items: list) -> float:
    """Assess governance improvement opportunities"""
    governance_score = 0
    management_quality = assess_management_quality(metrics, financial_line_items)
    if management_quality < 0.5:
        governance_score += 0.4
    capital_allocation = analyze_capital_allocation_opportunities(financial_line_items)
    if capital_allocation > 0.6:
        governance_score += 0.3
    if metrics:
        latest = metrics[0]
        if latest.return_on_equity and latest.return_on_equity < 0.1:
            governance_score += 0.3
    return governance_score

def assess_management_quality(metrics: list, financial_line_items: list) -> float:
    """Assess management quality"""
    if not metrics or not financial_line_items:
        return 0.5
    management_score = 0
    shares = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares]
    if len(shares) >= 3:
        share_reduction = (shares[-1] - shares[0]) / shares[-1]
        if share_reduction > 0.1:
            management_score += 0.3
    if metrics and len(metrics) >= 3:
        roe_vals = [m.return_on_equity for m in metrics if m.return_on_equity]
        if len(roe_vals) >= 3:
            recent_roe = sum(roe_vals[:2]) / 2
            historical_roe = sum(roe_vals[2:]) / len(roe_vals[2:])
            if recent_roe > historical_roe:
                management_score += 0.4
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow]
    earnings = [item.net_income for item in financial_line_items if item.net_income]
    if fcf_vals and earnings and (len(fcf_vals) == len(earnings)):
        fcf_conversion = sum(fcf_vals) / sum(earnings) if sum(earnings) > 0 else 0
        if fcf_conversion > 0.8:
            management_score += 0.3
    return management_score

def assess_esg_improvement_opportunities(financial_line_items: list) -> float:
    """Assess ESG improvement opportunities"""
    esg_score = 0
    rd_vals = [item.research_and_development for item in financial_line_items if item.research_and_development]
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if rd_vals and revenues:
        rd_intensity = sum(rd_vals) / sum(revenues[:len(rd_vals)])
        if rd_intensity < 0.03:
            esg_score += 0.3
    governance_opportunity = assess_governance_improvement_opportunities(metrics, financial_line_items)
    esg_score += governance_opportunity * 0.4
    if latest.revenue and latest.total_assets:
        productivity_proxy = latest.revenue / latest.total_assets
        if productivity_proxy < 0.8:
            esg_score += 0.3
    return min(esg_score, 1.0)

