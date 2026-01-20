# Cluster 55

def identify_statistical_patterns(financial_line_items: list) -> float:
    """Identify statistical patterns in financial data"""
    earnings = [item.net_income for item in financial_line_items if item.net_income]
    if len(earnings) >= 8:
        autocorr = calculate_autocorrelation(earnings, lag=1)
        pattern_strength = abs(autocorr)
        return min(pattern_strength, 1.0)
    return 0.3

def calculate_autocorrelation(data: list, lag: int) -> float:
    """Calculate autocorrelation with given lag"""
    if len(data) <= lag:
        return 0
    mean_val = sum(data) / len(data)
    numerator = sum(((data[i] - mean_val) * (data[i + lag] - mean_val) for i in range(len(data) - lag)))
    denominator = sum(((x - mean_val) ** 2 for x in data))
    return numerator / denominator if denominator != 0 else 0

