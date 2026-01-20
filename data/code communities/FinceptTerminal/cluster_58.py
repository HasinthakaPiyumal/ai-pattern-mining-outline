# Cluster 58

def calculate_expected_shortfall(financial_line_items: list) -> float:
    """Calculate Expected Shortfall (Conditional VaR)"""
    var = calculate_value_at_risk(financial_line_items)
    return var * 1.3

def calculate_value_at_risk(financial_line_items: list) -> float:
    """Calculate Value-at-Risk"""
    earnings = [item.net_income for item in financial_line_items if item.net_income]
    if earnings:
        sorted_earnings = sorted(earnings)
        var_index = int(len(sorted_earnings) * 0.05)
        if var_index < len(sorted_earnings):
            worst_case = sorted_earnings[var_index]
            max_earnings = max(earnings)
            if max_earnings > 0:
                return abs(worst_case - max_earnings) / max_earnings
    return 0.3

