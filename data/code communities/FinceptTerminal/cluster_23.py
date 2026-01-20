# Cluster 23

def standard_normal_cdf(x: Decimal) -> Decimal:
    """Approximate standard normal cumulative distribution function"""
    x_float = float(x)
    if x_float < 0:
        return Decimal('1') - standard_normal_cdf(-x)
    a1 = Decimal('0.254829592')
    a2 = Decimal('-0.284496736')
    a3 = Decimal('1.421413741')
    a4 = Decimal('-1.453152027')
    a5 = Decimal('1.061405429')
    p = Decimal('0.3275911')
    t = Decimal('1') / (Decimal('1') + p * x)
    y = Decimal('1') - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Decimal(str(math.exp(-float(x * x / Decimal('2')))))
    return y

def black_scholes_call_price(S: Decimal, K: Decimal, T: Decimal, r: Decimal, sigma: Decimal) -> Decimal:
    """Black-Scholes call option price (simplified implementation)"""
    if T <= 0 or sigma <= 0:
        return max(S - K, Decimal('0'))
    d1 = (Decimal(str(math.log(float(S / K)))) + (r + sigma * sigma / Decimal('2')) * T) / (sigma * Decimal(str(math.sqrt(float(T)))))
    d2 = d1 - sigma * Decimal(str(math.sqrt(float(T))))
    call_price = S * standard_normal_cdf(d1) - K * Decimal(str(math.exp(-float(r * T)))) * standard_normal_cdf(d2)
    return call_price

