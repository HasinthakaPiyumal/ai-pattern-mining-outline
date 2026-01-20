# Cluster 52

def convert_to_serializable(obj):
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, (int, float, bool, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return str(obj)

def show_agent_reasoning(output, agent_name):
    print(f'\n{'=' * 10} {agent_name.center(28)} {'=' * 10}')

    def convert_to_serializable(obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)
    if isinstance(output, (dict, list)):
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2))
    else:
        try:
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            print(output)
    print('=' * 48)

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str=None) -> list[Price]:
    """Fetch price data from cache or API."""
    cache_key = f'{ticker}_{start_date}_{end_date}'
    if (cached_data := _cache.get_prices(cache_key)):
        return [Price(**price) for price in cached_data]
    headers = {}
    financial_api_key = api_key or os.environ.get('FINANCIAL_DATASETS_API_KEY')
    if financial_api_key:
        headers['X-API-KEY'] = financial_api_key
    url = f'https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}'
    response = _make_api_request(url, headers)
    if response.status_code != 200:
        raise Exception(f'Error fetching data: {ticker} - {response.status_code} - {response.text}')
    price_response = PriceResponse(**response.json())
    prices = price_response.prices
    if not prices:
        return []
    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices

def _make_api_request(url: str, headers: dict, method: str='GET', json_data: dict=None, max_retries: int=3) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.
    
    Args:
        url: The URL to request
        headers: Headers to include in the request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests
        max_retries: Maximum number of retries (default: 3)
    
    Returns:
        requests.Response: The response object
    
    Raises:
        Exception: If the request fails with a non-429 error
    """
    for attempt in range(max_retries + 1):
        if method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)
        if response.status_code == 429 and attempt < max_retries:
            delay = 60 + 30 * attempt
            print(f'Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...')
            time.sleep(delay)
            continue
        return response

def get_financial_metrics(ticker: str, end_date: str, period: str='ttm', limit: int=10, api_key: str=None) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    cache_key = f'{ticker}_{period}_{end_date}_{limit}'
    if (cached_data := _cache.get_financial_metrics(cache_key)):
        return [FinancialMetrics(**metric) for metric in cached_data]
    headers = {}
    financial_api_key = api_key or os.environ.get('FINANCIAL_DATASETS_API_KEY')
    if financial_api_key:
        headers['X-API-KEY'] = financial_api_key
    url = f'https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}'
    response = _make_api_request(url, headers)
    if response.status_code != 200:
        raise Exception(f'Error fetching data: {ticker} - {response.status_code} - {response.text}')
    metrics_response = FinancialMetricsResponse(**response.json())
    financial_metrics = metrics_response.financial_metrics
    if not financial_metrics:
        return []
    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
    return financial_metrics

def search_line_items(ticker: str, line_items: list[str], end_date: str, period: str='ttm', limit: int=10, api_key: str=None) -> list[LineItem]:
    """Fetch line items from API."""
    headers = {}
    financial_api_key = api_key or os.environ.get('FINANCIAL_DATASETS_API_KEY')
    if financial_api_key:
        headers['X-API-KEY'] = financial_api_key
    url = 'https://api.financialdatasets.ai/financials/search/line-items'
    body = {'tickers': [ticker], 'line_items': line_items, 'end_date': end_date, 'period': period, 'limit': limit}
    response = _make_api_request(url, headers, method='POST', json_data=body)
    if response.status_code != 200:
        raise Exception(f'Error fetching data: {ticker} - {response.status_code} - {response.text}')
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []
    return search_results[:limit]

def get_insider_trades(ticker: str, end_date: str, start_date: str | None=None, limit: int=1000, api_key: str=None) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    cache_key = f'{ticker}_{start_date or 'none'}_{end_date}_{limit}'
    if (cached_data := _cache.get_insider_trades(cache_key)):
        return [InsiderTrade(**trade) for trade in cached_data]
    headers = {}
    financial_api_key = api_key or os.environ.get('FINANCIAL_DATASETS_API_KEY')
    if financial_api_key:
        headers['X-API-KEY'] = financial_api_key
    all_trades = []
    current_end_date = end_date
    while True:
        url = f'https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}'
        if start_date:
            url += f'&filing_date_gte={start_date}'
        url += f'&limit={limit}'
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            raise Exception(f'Error fetching data: {ticker} - {response.status_code} - {response.text}')
        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        if not insider_trades:
            break
        all_trades.extend(insider_trades)
        if not start_date or len(insider_trades) < limit:
            break
        current_end_date = min((trade.filing_date for trade in insider_trades)).split('T')[0]
        if current_end_date <= start_date:
            break
    if not all_trades:
        return []
    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    return all_trades

def get_company_news(ticker: str, end_date: str, start_date: str | None=None, limit: int=1000, api_key: str=None) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    cache_key = f'{ticker}_{start_date or 'none'}_{end_date}_{limit}'
    if (cached_data := _cache.get_company_news(cache_key)):
        return [CompanyNews(**news) for news in cached_data]
    headers = {}
    financial_api_key = api_key or os.environ.get('FINANCIAL_DATASETS_API_KEY')
    if financial_api_key:
        headers['X-API-KEY'] = financial_api_key
    all_news = []
    current_end_date = end_date
    while True:
        url = f'https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}'
        if start_date:
            url += f'&start_date={start_date}'
        url += f'&limit={limit}'
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            raise Exception(f'Error fetching data: {ticker} - {response.status_code} - {response.text}')
        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news
        if not company_news:
            break
        all_news.extend(company_news)
        if not start_date or len(company_news) < limit:
            break
        current_end_date = min((news.date for news in company_news)).split('T')[0]
        if current_end_date <= start_date:
            break
    if not all_news:
        return []
    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news

def get_market_cap(ticker: str, end_date: str, api_key: str=None) -> float | None:
    """Fetch market cap from the API."""
    if end_date == datetime.datetime.now().strftime('%Y-%m-%d'):
        headers = {}
        financial_api_key = api_key or os.environ.get('FINANCIAL_DATASETS_API_KEY')
        if financial_api_key:
            headers['X-API-KEY'] = financial_api_key
        url = f'https://api.financialdatasets.ai/company/facts/?ticker={ticker}'
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            print(f'Error fetching company facts: {ticker} - {response.status_code}')
            return None
        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap
    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if not financial_metrics:
        return None
    market_cap = financial_metrics[0].market_cap
    if not market_cap:
        return None
    return market_cap

def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str=None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)

def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df['Date'] = pd.to_datetime(df['time'])
    df.set_index('Date', inplace=True)
    numeric_cols = ['open', 'close', 'high', 'low', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.sort_index(inplace=True)
    return df

def bridgewater_associates_agent(state: AgentState, agent_id: str='bridgewater_associates_agent'):
    """
    Bridgewater Associates: All Weather portfolio, economic machine principles
    Structure: Economic Research → Portfolio Construction → Risk Management → Ray Dalio Final Decision
    Philosophy: Diversification across economic environments, systematic risk parity
    """
    data = state['data']
    end_date = data['end_date']
    tickers = data['tickers']
    api_key = get_api_key_from_state(state, 'FINANCIAL_DATASETS_API_KEY')
    analysis_data = {}
    bridgewater_analysis = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, 'Economic research team analyzing macro environment')
        metrics = get_financial_metrics(ticker, end_date, period='annual', limit=10, api_key=api_key)
        financial_line_items = search_line_items(ticker, ['revenue', 'net_income', 'free_cash_flow', 'total_debt', 'shareholders_equity', 'operating_margin', 'total_assets', 'current_assets', 'current_liabilities', 'interest_expense'], end_date, period='annual', limit=10, api_key=api_key)
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
        progress.update_status(agent_id, ticker, 'Economic research team analysis')
        economic_research = economic_research_team_analysis(metrics, financial_line_items)
        progress.update_status(agent_id, ticker, 'Portfolio construction team analysis')
        portfolio_construction = portfolio_construction_team_analysis(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Risk management team analysis')
        risk_management = risk_management_team_analysis(financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'All Weather framework analysis')
        all_weather_analysis = all_weather_framework_analysis(metrics, financial_line_items)
        progress.update_status(agent_id, ticker, "Ray Dalio's synthesis and final decision")
        ray_dalio_synthesis = ray_dalio_final_decision(economic_research, portfolio_construction, risk_management, all_weather_analysis)
        total_score = economic_research['score'] * 0.3 + all_weather_analysis['score'] * 0.25 + risk_management['score'] * 0.25 + portfolio_construction['score'] * 0.2
        if total_score >= 7.5:
            signal = 'bullish'
        elif total_score <= 4.0:
            signal = 'bearish'
        else:
            signal = 'neutral'
        analysis_data[ticker] = {'signal': signal, 'score': total_score, 'economic_research': economic_research, 'portfolio_construction': portfolio_construction, 'risk_management': risk_management, 'all_weather_analysis': all_weather_analysis, 'ray_dalio_synthesis': ray_dalio_synthesis}
        bridgewater_output = generate_bridgewater_output(ticker, analysis_data, state, agent_id)
        bridgewater_analysis[ticker] = {'signal': bridgewater_output.signal, 'confidence': bridgewater_output.confidence, 'reasoning': bridgewater_output.reasoning, 'all_weather_allocation': bridgewater_output.all_weather_allocation}
        progress.update_status(agent_id, ticker, 'Done', analysis=bridgewater_output.reasoning)
    message = HumanMessage(content=json.dumps(bridgewater_analysis), name=agent_id)
    if state['metadata']['show_reasoning']:
        show_agent_reasoning(bridgewater_analysis, 'Bridgewater Associates')
    state['data']['analyst_signals'][agent_id] = bridgewater_analysis
    progress.update_status(agent_id, None, 'Done')
    return {'messages': [message], 'data': state['data']}

def get_api_key_from_state(state: dict, api_key_name: str) -> str:
    """Get an API key from the state object."""
    if state and state.get('metadata', {}).get('request'):
        request = state['metadata']['request']
        if hasattr(request, 'api_keys') and request.api_keys:
            return request.api_keys.get(api_key_name)
    return None

def portfolio_construction_team_analysis(metrics: list, financial_line_items: list, market_cap: float) -> dict:
    """AQR's portfolio construction team analysis"""
    score = 0
    details = []
    if not metrics or not financial_line_items or (not market_cap):
        return {'score': 0, 'details': 'No portfolio construction data'}
    factor_timing_score = analyze_factor_timing(financial_line_items)
    if factor_timing_score > 0.6:
        score += 2
        details.append(f'Good factor timing opportunity: {factor_timing_score:.2f}')
    position_sizing = optimize_position_sizing(metrics, market_cap)
    if position_sizing > 0.7:
        score += 2
        details.append('Optimal position sizing for factor exposure')
    transaction_costs = analyze_transaction_costs_aqr(market_cap)
    if transaction_costs < 0.02:
        score += 1.5
        details.append(f'Low implementation costs: {transaction_costs:.2%}')
    strategy_capacity = assess_aqr_strategy_capacity(market_cap)
    if strategy_capacity > 0.8:
        score += 1.5
        details.append('High strategy capacity')
    risk_budgeting_score = analyze_risk_budgeting(financial_line_items)
    if risk_budgeting_score > 0.7:
        score += 1
        details.append('Efficient risk budgeting allocation')
    rebalancing_analysis = optimize_rebalancing_frequency(financial_line_items)
    if rebalancing_analysis > 0.6:
        score += 1
        details.append('Optimal rebalancing frequency identified')
    return {'score': score, 'details': '; '.join(details)}

def renaissance_technologies_agent(state: AgentState, agent_id: str='renaissance_technologies_agent'):
    """
    Renaissance Technologies: Pure quantitative, statistical arbitrage
    Structure: Signal Generation → Risk Models → Execution Optimization → Jim Simons Systematic Decision
    Philosophy: Mathematical models, statistical edges, high-frequency systematic trading
    """
    data = state['data']
    end_date = data['end_date']
    tickers = data['tickers']
    api_key = get_api_key_from_state(state, 'FINANCIAL_DATASETS_API_KEY')
    analysis_data = {}
    renaissance_analysis = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, 'Signal generation algorithms analyzing patterns')
        metrics = get_financial_metrics(ticker, end_date, period='quarterly', limit=20, api_key=api_key)
        financial_line_items = search_line_items(ticker, ['revenue', 'net_income', 'free_cash_flow', 'total_assets', 'shareholders_equity', 'operating_margin', 'total_debt', 'current_assets', 'current_liabilities', 'outstanding_shares'], end_date, period='quarterly', limit=20, api_key=api_key)
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
        progress.update_status(agent_id, ticker, 'Signal generation team analysis')
        signal_generation = signal_generation_team_analysis(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Risk modeling team analysis')
        risk_modeling = risk_modeling_team_analysis(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Execution optimization analysis')
        execution_optimization = execution_optimization_analysis(financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Statistical arbitrage models')
        statistical_arbitrage = statistical_arbitrage_analysis(metrics, financial_line_items)
        progress.update_status(agent_id, ticker, 'Jim Simons systematic synthesis')
        simons_systematic_decision = simons_systematic_synthesis(signal_generation, risk_modeling, execution_optimization, statistical_arbitrage)
        total_score = signal_generation['score'] * 0.35 + statistical_arbitrage['score'] * 0.3 + risk_modeling['score'] * 0.2 + execution_optimization['score'] * 0.15
        if total_score >= 8.0 and signal_generation.get('statistical_significance', 0) > 0.95:
            signal = 'bullish'
        elif total_score <= 3.0 or signal_generation.get('statistical_significance', 0) < 0.6:
            signal = 'bearish'
        else:
            signal = 'neutral'
        analysis_data[ticker] = {'signal': signal, 'score': total_score, 'signal_generation': signal_generation, 'risk_modeling': risk_modeling, 'execution_optimization': execution_optimization, 'statistical_arbitrage': statistical_arbitrage, 'simons_systematic_decision': simons_systematic_decision}
        renaissance_output = generate_renaissance_output(ticker, analysis_data, state, agent_id)
        renaissance_analysis[ticker] = {'signal': renaissance_output.signal, 'confidence': renaissance_output.confidence, 'reasoning': renaissance_output.reasoning, 'statistical_edge': renaissance_output.statistical_edge, 'signal_strength': renaissance_output.signal_strength}
        progress.update_status(agent_id, ticker, 'Done', analysis=renaissance_output.reasoning)
    message = HumanMessage(content=json.dumps(renaissance_analysis), name=agent_id)
    if state['metadata']['show_reasoning']:
        show_agent_reasoning(renaissance_analysis, 'Renaissance Technologies')
    state['data']['analyst_signals'][agent_id] = renaissance_analysis
    progress.update_status(agent_id, None, 'Done')
    return {'messages': [message], 'data': state['data']}

def risk_modeling_team_analysis(metrics: list, financial_line_items: list, market_cap: float) -> dict:
    """Renaissance's risk modeling team analysis"""
    score = 0
    details = []
    risk_metrics = {}
    if not metrics or not financial_line_items:
        return {'score': 0, 'details': 'No risk modeling data'}
    volatility_score = model_volatility_patterns(financial_line_items)
    risk_metrics['volatility'] = volatility_score
    if volatility_score > 0.7:
        score += 2
        details.append('Low volatility risk profile')
    correlation_risk = analyze_correlation_risks(metrics, financial_line_items)
    risk_metrics['correlation'] = correlation_risk
    if correlation_risk < 0.6:
        score += 2
        details.append('Low correlation risk with market factors')
    liquidity_risk = model_liquidity_risk(financial_line_items, market_cap)
    risk_metrics['liquidity'] = liquidity_risk
    if liquidity_risk > 0.8:
        score += 1
        details.append('Adequate liquidity for position sizing')
    factor_risks = decompose_factor_risks(metrics)
    risk_metrics['factor_risks'] = factor_risks
    if factor_risks.get('idiosyncratic_risk', 0) > 0.6:
        score += 2
        details.append('High idiosyncratic risk component - good for alpha generation')
    drawdown_risk = calculate_maximum_drawdown_risk(financial_line_items)
    risk_metrics['drawdown'] = drawdown_risk
    if drawdown_risk < 0.3:
        score += 1
        details.append('Limited drawdown risk based on historical patterns')
    return {'score': score, 'details': '; '.join(details), 'risk_metrics': risk_metrics}

def decompose_factor_risks(metrics: list, financial_line_items: list) -> dict:
    """Decompose factor risks"""
    return {'market_beta': 0.7, 'value_loading': 0.6, 'momentum_loading': 0.4, 'quality_loading': 0.5, 'size_loading': 0.3, 'diversification_ratio': 0.8}

def two_sigma_hedge_fund_agent(state: AgentState, agent_id: str='two_sigma_hedge_fund_agent'):
    """
    Two Sigma: Machine learning, data science
    Structure: Data Science → ML Engineering → Risk Models → Portfolio Optimization → Scientific Decision
    Philosophy: Data-driven, machine learning, scientific approach, technology focus
    """
    data = state['data']
    end_date = data['end_date']
    tickers = data['tickers']
    api_key = get_api_key_from_state(state, 'FINANCIAL_DATASETS_API_KEY')
    analysis_data = {}
    two_sigma_analysis = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, 'Data science team feature engineering')
        metrics = get_financial_metrics(ticker, end_date, period='quarterly', limit=16, api_key=api_key)
        financial_line_items = search_line_items(ticker, ['revenue', 'net_income', 'free_cash_flow', 'total_debt', 'shareholders_equity', 'operating_margin', 'total_assets', 'current_assets', 'current_liabilities', 'research_and_development'], end_date, period='quarterly', limit=16, api_key=api_key)
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
        progress.update_status(agent_id, ticker, 'Data science team analysis')
        data_science_team = data_science_team_analysis(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'ML engineering team models')
        ml_engineering_team = ml_engineering_team_analysis(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Risk modeling team analysis')
        risk_modeling_team = risk_modeling_team_analysis(metrics, financial_line_items)
        progress.update_status(agent_id, ticker, 'Portfolio optimization analysis')
        portfolio_optimization = portfolio_optimization_analysis(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Scientific method synthesis')
        scientific_synthesis = scientific_method_synthesis(data_science_team, ml_engineering_team, risk_modeling_team, portfolio_optimization)
        total_score = ml_engineering_team['score'] * 0.35 + data_science_team['score'] * 0.25 + portfolio_optimization['score'] * 0.2 + risk_modeling_team['score'] * 0.2
        ml_confidence = ml_engineering_team.get('model_confidence', 0)
        if total_score >= 8.0 and ml_confidence > 0.75:
            signal = 'bullish'
        elif total_score <= 3.5 or ml_confidence < 0.4:
            signal = 'bearish'
        else:
            signal = 'neutral'
        analysis_data[ticker] = {'signal': signal, 'score': total_score, 'data_science_team': data_science_team, 'ml_engineering_team': ml_engineering_team, 'risk_modeling_team': risk_modeling_team, 'portfolio_optimization': portfolio_optimization, 'scientific_synthesis': scientific_synthesis}
        two_sigma_output = generate_two_sigma_output(ticker, analysis_data, state, agent_id)
        two_sigma_analysis[ticker] = {'signal': two_sigma_output.signal, 'confidence': two_sigma_output.confidence, 'reasoning': two_sigma_output.reasoning, 'ml_model_predictions': two_sigma_output.ml_model_predictions}
        progress.update_status(agent_id, ticker, 'Done', analysis=two_sigma_output.reasoning)
    message = HumanMessage(content=json.dumps(two_sigma_analysis), name=agent_id)
    if state['metadata']['show_reasoning']:
        show_agent_reasoning(two_sigma_analysis, 'Two Sigma')
    state['data']['analyst_signals'][agent_id] = two_sigma_analysis
    progress.update_status(agent_id, None, 'Done')
    return {'messages': [message], 'data': state['data']}

def citadel_hedge_fund_agent(state: AgentState, agent_id: str='citadel_hedge_fund_agent'):
    """
    Citadel: Multi-strategy quantitative approach
    Structure: Fundamental Research → Quantitative Research → Global Macro → Trading → Ken Griffin Final Decision
    Philosophy: Multi-strategy approach, technological edge, risk management, market making
    """
    data = state['data']
    end_date = data['end_date']
    tickers = data['tickers']
    api_key = get_api_key_from_state(state, 'FINANCIAL_DATASETS_API_KEY')
    analysis_data = {}
    citadel_analysis = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, 'Fundamental research team analyzing company')
        metrics = get_financial_metrics(ticker, end_date, period='annual', limit=8, api_key=api_key)
        financial_line_items = search_line_items(ticker, ['revenue', 'net_income', 'free_cash_flow', 'total_debt', 'shareholders_equity', 'operating_margin', 'total_assets', 'current_assets', 'current_liabilities', 'research_and_development'], end_date, period='annual', limit=8, api_key=api_key)
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
        progress.update_status(agent_id, ticker, 'Fundamental research department')
        fundamental_research = fundamental_research_department(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Quantitative research department')
        quantitative_research = quantitative_research_department(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Global macro department')
        global_macro = global_macro_department(metrics, financial_line_items)
        progress.update_status(agent_id, ticker, 'Trading department analysis')
        trading_department = trading_department_analysis(financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Risk management oversight')
        risk_management = risk_management_oversight(financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, "Ken Griffin's multi-strategy synthesis")
        ken_griffin_decision = ken_griffin_multi_strategy_synthesis(fundamental_research, quantitative_research, global_macro, trading_department, risk_management)
        total_score = fundamental_research['score'] * 0.25 + quantitative_research['score'] * 0.25 + trading_department['score'] * 0.2 + global_macro['score'] * 0.15 + risk_management['score'] * 0.15
        if total_score >= 8.0 and ken_griffin_decision.get('risk_adjusted_return', 0) > 0.15:
            signal = 'bullish'
        elif total_score <= 4.0 or len(risk_management.get('risk_flags', [])) > 2:
            signal = 'bearish'
        else:
            signal = 'neutral'
        analysis_data[ticker] = {'signal': signal, 'score': total_score, 'fundamental_research': fundamental_research, 'quantitative_research': quantitative_research, 'global_macro': global_macro, 'trading_department': trading_department, 'risk_management': risk_management, 'ken_griffin_decision': ken_griffin_decision}
        citadel_output = generate_citadel_output(ticker, analysis_data, state, agent_id)
        citadel_analysis[ticker] = {'signal': citadel_output.signal, 'confidence': citadel_output.confidence, 'reasoning': citadel_output.reasoning, 'strategy_allocation': citadel_output.strategy_allocation}
        progress.update_status(agent_id, ticker, 'Done', analysis=citadel_output.reasoning)
    message = HumanMessage(content=json.dumps(citadel_analysis), name=agent_id)
    if state['metadata']['show_reasoning']:
        show_agent_reasoning(citadel_analysis, 'Citadel')
    state['data']['analyst_signals'][agent_id] = citadel_analysis
    progress.update_status(agent_id, None, 'Done')
    return {'messages': [message], 'data': state['data']}

def elliott_management_hedge_fund_agent(state: AgentState, agent_id: str='elliott_management_hedge_fund_agent'):
    """
    Elliott Management: Activist investing, event-driven strategies
    Structure: Research → Legal → Activism → Event-Driven → Paul Singer Decision
    Philosophy: Catalyst-driven value creation, shareholder activism, distressed opportunities
    """
    data = state['data']
    end_date = data['end_date']
    tickers = data['tickers']
    api_key = get_api_key_from_state(state, 'FINANCIAL_DATASETS_API_KEY')
    analysis_data = {}
    elliott_analysis = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, 'Research team analyzing activist opportunity')
        metrics = get_financial_metrics(ticker, end_date, period='annual', limit=6, api_key=api_key)
        financial_line_items = search_line_items(ticker, ['revenue', 'net_income', 'free_cash_flow', 'total_debt', 'shareholders_equity', 'operating_margin', 'total_assets', 'current_assets', 'current_liabilities', 'research_and_development'], end_date, period='annual', limit=6, api_key=api_key)
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
        progress.update_status(agent_id, ticker, 'Research team fundamental analysis')
        research_team = research_team_analysis(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Legal team governance analysis')
        legal_team = legal_team_governance_analysis(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Activism team catalyst identification')
        activism_team = activism_team_analysis(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Event-driven team analysis')
        event_driven_team = event_driven_team_analysis(metrics, financial_line_items, market_cap)
        progress.update_status(agent_id, ticker, 'Paul Singer strategic decision')
        paul_singer_decision = paul_singer_strategic_decision(research_team, legal_team, activism_team, event_driven_team)
        total_score = activism_team['score'] * 0.35 + research_team['score'] * 0.25 + event_driven_team['score'] * 0.2 + legal_team['score'] * 0.2
        activist_potential = activism_team.get('activist_score', 0)
        if total_score >= 7.5 and activist_potential > 0.7:
            signal = 'bullish'
        elif total_score <= 4.0 or legal_team.get('governance_score', 0) < 0.3:
            signal = 'bearish'
        else:
            signal = 'neutral'
        analysis_data[ticker] = {'signal': signal, 'score': total_score, 'research_team': research_team, 'legal_team': legal_team, 'activism_team': activism_team, 'event_driven_team': event_driven_team, 'paul_singer_decision': paul_singer_decision}
        elliott_output = generate_elliott_output(ticker, analysis_data, state, agent_id)
        elliott_analysis[ticker] = {'signal': elliott_output.signal, 'confidence': elliott_output.confidence, 'reasoning': elliott_output.reasoning, 'activist_potential': elliott_output.activist_potential}
        progress.update_status(agent_id, ticker, 'Done', analysis=elliott_output.reasoning)
    message = HumanMessage(content=json.dumps(elliott_analysis), name=agent_id)
    if state['metadata']['show_reasoning']:
        show_agent_reasoning(elliott_analysis, 'Elliott Management')
    state['data']['analyst_signals'][agent_id] = elliott_analysis
    progress.update_status(agent_id, None, 'Done')
    return {'messages': [message], 'data': state['data']}

