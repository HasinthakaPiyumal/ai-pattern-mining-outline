# Cluster 3

class Alpaca(DataSource):
    """Retrieves stock data from `Alpaca <https://alpaca.markets/>`_."""
    __EST: Final = 'US/Eastern'

    def __init__(self, api_key: str, api_secret: str):
        super().__init__()
        self._api = alpaca_stock.StockHistoricalDataClient(api_key, api_secret)

    def query(self, symbols: Union[str, Iterable[str]], start_date: Union[str, datetime], end_date: Union[str, datetime], timeframe: Optional[str]='1d', adjust: Optional[Any]=None) -> pd.DataFrame:
        _parse_alpaca_timeframe(timeframe)
        return super().query(symbols, start_date, end_date, timeframe, adjust)

    def _fetch_data(self, symbols: frozenset[str], start_date: datetime, end_date: datetime, timeframe: Optional[str], adjust: Optional[Any]) -> pd.DataFrame:
        """:meta private:"""
        amount, unit = _parse_alpaca_timeframe(timeframe)
        adj_enum = None
        if adjust is not None:
            for member in Adjustment:
                if member.value == adjust:
                    adj_enum = member
                    break
            if adj_enum is None:
                raise ValueError(f'Unknown adjustment: {adjust}.')
        request = StockBarsRequest(symbol_or_symbols=list(symbols), start=start_date, end=end_date, timeframe=TimeFrame(amount, unit), limit=None, adjustment=adj_enum, feed=None)
        df = self._api.get_stock_bars(request).df
        if df.columns.empty:
            return pd.DataFrame(columns=[DataCol.SYMBOL.value, DataCol.DATE.value, DataCol.OPEN.value, DataCol.HIGH.value, DataCol.LOW.value, DataCol.CLOSE.value, DataCol.VOLUME.value, DataCol.VWAP.value])
        if df.empty:
            return df
        df = df.reset_index()
        df.rename(columns={'timestamp': DataCol.DATE.value}, inplace=True)
        df = df[[col.value for col in DataCol]]
        df[DataCol.DATE.value] = pd.to_datetime(df[DataCol.DATE.value])
        df[DataCol.DATE.value] = df[DataCol.DATE.value].dt.tz_convert(self.__EST)
        return df

def _parse_alpaca_timeframe(timeframe: Optional[str]) -> tuple[int, TimeFrameUnit]:
    if timeframe is None:
        raise ValueError('Timeframe needs to be specified for Alpaca.')
    parts = parse_timeframe(timeframe)
    if len(parts) != 1:
        raise ValueError(f'Invalid Alpaca timeframe: {timeframe}')
    tf = parts[0]
    if tf[1] == 'min':
        unit = TimeFrameUnit.Minute
    elif tf[1] == 'hour':
        unit = TimeFrameUnit.Hour
    elif tf[1] == 'day':
        unit = TimeFrameUnit.Day
    elif tf[1] == 'week':
        unit = TimeFrameUnit.Week
    else:
        raise ValueError(f'Invalid Alpaca timeframe: {timeframe}')
    return (tf[0], unit)

class AlpacaCrypto(DataSource):
    """Retrieves crypto data from `Alpaca <https://alpaca.markets/>`_.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
    """
    TRADE_COUNT: Final = 'trade_count'
    COLUMNS: Final = (DataCol.SYMBOL.value, DataCol.DATE.value, DataCol.OPEN.value, DataCol.HIGH.value, DataCol.LOW.value, DataCol.CLOSE.value, DataCol.VOLUME.value, DataCol.VWAP.value, TRADE_COUNT)
    __EST: Final = 'US/Eastern'

    def __init__(self, api_key: str, api_secret: str):
        super().__init__()
        self._scope.register_custom_cols(self.TRADE_COUNT)
        self._api = alpaca_crypto.CryptoHistoricalDataClient(api_key, api_secret)

    def query(self, symbols: Union[str, Iterable[str]], start_date: Union[str, datetime], end_date: Union[str, datetime], timeframe: Optional[str]='1d', _adjust: Optional[str]=None) -> pd.DataFrame:
        _parse_alpaca_timeframe(timeframe)
        return super().query(symbols, start_date, end_date, timeframe, _adjust)

    def _fetch_data(self, symbols: frozenset[str], start_date: datetime, end_date: datetime, timeframe: Optional[str], _adjust: Optional[str]) -> pd.DataFrame:
        """:meta private:"""
        amount, unit = _parse_alpaca_timeframe(timeframe)
        request = CryptoBarsRequest(symbol_or_symbols=list(symbols), start=start_date, end=end_date, timeframe=TimeFrame(amount, unit), limit=None)
        df = self._api.get_crypto_bars(request).df
        if df.columns.empty:
            return pd.DataFrame(columns=self.COLUMNS)
        if df.empty:
            return df
        df = df.reset_index()
        df.rename(columns={'timestamp': DataCol.DATE.value}, inplace=True)
        df = df[[col for col in self.COLUMNS]]
        df[DataCol.DATE.value] = pd.to_datetime(df[DataCol.DATE.value])
        df[DataCol.DATE.value] = df[DataCol.DATE.value].dt.tz_convert(self.__EST)
        return df

