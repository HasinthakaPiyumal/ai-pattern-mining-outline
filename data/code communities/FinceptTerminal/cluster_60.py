# Cluster 60

def assess_management_change_potential(metrics: list, financial_line_items: list) -> float:
    """Assess management change potential"""
    management_score = assess_management_performance(metrics, financial_line_items)
    if management_score < 0.3:
        return 0.8
    elif management_score < 0.5:
        return 0.6
    return 0.2

def assess_management_performance(metrics: list, financial_line_items: list) -> float:
    """Assess management performance"""
    if not metrics:
        return 0.5
    roe_vals = [m.return_on_equity for m in metrics if m.return_on_equity]
    if len(roe_vals) >= 3:
        recent_roe = sum(roe_vals[:2]) / 2
        historical_roe = sum(roe_vals[2:]) / len(roe_vals[2:])
        if recent_roe > historical_roe * 1.2:
            return 0.8
        elif recent_roe < historical_roe * 0.8:
            return 0.3
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if len(revenues) >= 4:
        growth_years = sum((1 for i in range(len(revenues) - 1) if revenues[i] >= revenues[i + 1]))
        consistency = growth_years / (len(revenues) - 1)
        return consistency
    return 0.5

