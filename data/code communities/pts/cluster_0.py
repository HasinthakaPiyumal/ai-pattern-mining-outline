# Cluster 0

def calculate_risk_metrics(distribution, name):
    """Calculate risk-adjusted metrics for reasoning quality."""
    dist = np.array(distribution)
    mean_impact = np.mean(dist)
    volatility = np.std(dist)
    downside_risk = np.std(dist[dist < 0]) if len(dist[dist < 0]) > 0 else 0
    upside_potential = np.mean(dist[dist > 0.5]) if len(dist[dist > 0.5]) > 0 else 0
    risk_adjusted_quality = mean_impact / volatility if volatility > 0 else 0
    print(f'  {name}:')
    print(f'    • Mean Impact: {mean_impact:.3f}')
    print(f'    • Volatility: {volatility:.3f}')
    print(f'    • Downside Risk: {downside_risk:.3f}')
    print(f'    • Upside Potential: {upside_potential:.3f}')
    print(f'    • Risk-Adjusted Quality: {risk_adjusted_quality:.3f}')
    return {'mean_impact': mean_impact, 'volatility': volatility, 'downside_risk': downside_risk, 'upside_potential': upside_potential, 'risk_adjusted_quality': risk_adjusted_quality}

