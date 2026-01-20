# Cluster 19

def calculate_all_indicators(df: pd.DataFrame, price_col: str='close', high_col: str='high', low_col: str='low') -> pd.DataFrame:
    """
    Calculate all technical indicators for a given dataframe.
    
    Args:
        df: DataFrame with OHLC data
        price_col: Column name for closing prices
        high_col: Column name for high prices
        low_col: Column name for low prices
        
    Returns:
        pd.DataFrame: Original dataframe with added indicator columns
    """
    try:
        result_df = df.copy()
        result_df['SMA_20'] = TechnicalIndicators.sma(df[price_col], 20)
        result_df['EMA_20'] = TechnicalIndicators.ema(df[price_col], 20)
        result_df['RSI_14'] = TechnicalIndicators.rsi(df[price_col], 14)
        macd, signal, histogram = TechnicalIndicators.macd(df[price_col])
        result_df['MACD'] = macd
        result_df['MACD_Signal'] = signal
        result_df['MACD_Histogram'] = histogram
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df[price_col])
        result_df['BB_Upper'] = bb_upper
        result_df['BB_Middle'] = bb_middle
        result_df['BB_Lower'] = bb_lower
        if high_col in df.columns and low_col in df.columns:
            stoch_k, stoch_d = TechnicalIndicators.stochastic_oscillator(df[high_col], df[low_col], df[price_col])
            result_df['Stoch_K'] = stoch_k
            result_df['Stoch_D'] = stoch_d
            result_df['Williams_R'] = TechnicalIndicators.williams_r(df[high_col], df[low_col], df[price_col])
            result_df['ATR'] = TechnicalIndicators.atr(df[high_col], df[low_col], df[price_col])
            result_df['CCI'] = TechnicalIndicators.cci(df[high_col], df[low_col], df[price_col])
            adx, plus_di, minus_di = TechnicalIndicators.adx(df[high_col], df[low_col], df[price_col])
            result_df['ADX'] = adx
            result_df['Plus_DI'] = plus_di
            result_df['Minus_DI'] = minus_di
        print(f'Successfully calculated all technical indicators for {len(result_df)} data points')
        return result_df
    except Exception as e:
        print(f'Error calculating indicators: {e}')
        return df

