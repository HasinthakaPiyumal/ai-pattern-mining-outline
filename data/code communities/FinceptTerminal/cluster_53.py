# Cluster 53

def generate_bridgewater_output(ticker: str, analysis_data: dict, state: AgentState, agent_id: str) -> BridgewaterSignal:
    """Generate Bridgewater's systematic investment decision"""
    template = ChatPromptTemplate.from_messages([('system', "You are Bridgewater Associates' AI system, implementing Ray Dalio's All Weather and economic machine principles:\n\n        ORGANIZATIONAL STRUCTURE:\n        - Economic Research Team: Macro environment and debt cycle analysis\n        - Portfolio Construction Team: Risk-adjusted returns and diversification\n        - Risk Management Team: Comprehensive risk assessment\n        - All Weather Framework: Performance across four economic environments\n        - Ray Dalio Synthesis: Final decision integration\n\n        PHILOSOPHY:\n        1. All Weather: Balanced performance across economic environments (growth/inflation up/down)\n        2. Economic Machine: Understanding debt cycles and economic principles\n        3. Risk Parity: Equal risk contribution, not equal dollar allocation\n        4. Systematic Approach: Remove emotion, follow systematic principles\n        5. Diversification: True diversification across uncorrelated return streams\n        6. Transparency: Radical transparency in decision-making process\n\n        REASONING STYLE:\n        - Reference team analyses and organizational structure\n        - Apply economic machine principles to company analysis\n        - Consider All Weather framework and environmental balance\n        - Discuss risk parity and diversification benefits\n        - Synthesize multiple team perspectives\n        - Express systematic, principle-based reasoning\n        - Consider position sizing based on risk contribution\n\n        Return the investment signal with All Weather allocation recommendations."), ('human', 'Apply Bridgewater\'s systematic analysis to {ticker}:\n\n        {analysis_data}\n\n        Provide investment signal in JSON format:\n        {{\n          "signal": "bullish" | "bearish" | "neutral",\n          "confidence": float (0-100),\n          "reasoning": "string",\n          "all_weather_allocation": {{\n            "rising_growth_weight": float,\n            "falling_growth_weight": float,\n            "rising_inflation_weight": float,\n            "falling_inflation_weight": float\n          }}\n        }}')])
    prompt = template.invoke({'analysis_data': json.dumps(analysis_data, indent=2), 'ticker': ticker})

    def create_default_bridgewater_signal():
        return BridgewaterSignal(signal='neutral', confidence=0.0, reasoning='Analysis error, defaulting to neutral', all_weather_allocation={'rising_growth_weight': 0.25, 'falling_growth_weight': 0.25, 'rising_inflation_weight': 0.25, 'falling_inflation_weight': 0.25})
    return call_llm(prompt=prompt, pydantic_model=BridgewaterSignal, agent_name=agent_id, state=state, default_factory=create_default_bridgewater_signal)

def call_llm(prompt: any, pydantic_model: type[BaseModel], agent_name: str | None=None, state: AgentState | None=None, max_retries: int=3, default_factory=None) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """
    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    else:
        model_name = 'gpt-4.1'
        model_provider = 'OPENAI'
    api_keys = None
    if state:
        request = state.get('metadata', {}).get('request')
        if request and hasattr(request, 'api_keys'):
            api_keys = request.api_keys
    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider, api_keys)
    if not (model_info and (not model_info.has_json_mode())):
        llm = llm.with_structured_output(pydantic_model, method='json_mode')
    for attempt in range(max_retries):
        try:
            result = llm.invoke(prompt)
            if model_info and (not model_info.has_json_mode()):
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result
        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f'Error - retry {attempt + 1}/{max_retries}')
            if attempt == max_retries - 1:
                print(f'Error in LLM call after {max_retries} attempts: {e}')
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)
    return create_default_response(pydantic_model)

def generate_renaissance_output(ticker: str, analysis_data: dict, state: AgentState, agent_id: str) -> RenaissanceSignal:
    """Generate Renaissance's systematic quantitative decision"""
    template = ChatPromptTemplate.from_messages([('system', "You are Renaissance Technologies' AI system, implementing Jim Simons' quantitative systematic approach:\n\n        ORGANIZATIONAL STRUCTURE:\n        - Signal Generation Team: Mathematical pattern recognition and statistical signals\n        - Risk Modeling Team: Volatility, correlation, and factor risk analysis\n        - Execution Optimization Team: Market impact and transaction cost modeling\n        - Statistical Arbitrage Team: Pairs trading and mispricing detection\n        - Jim Simons Systematic Synthesis: Mathematical model integration\n\n        PHILOSOPHY:\n        1. Pure Quantitative: Mathematical models over fundamental analysis\n        2. Statistical Edge: Identify small, consistent statistical advantages\n        3. Systematic Execution: Remove human emotion and bias\n        4. High Frequency: Exploit short-term market inefficiencies\n        5. Risk Management: Sophisticated mathematical risk models\n        6. Diversification: Many small bets rather than few large ones\n        7. Continuous Learning: Models adapt and evolve systematically\n\n        REASONING STYLE:\n        - Reference mathematical models and statistical significance\n        - Discuss signal generation algorithms and pattern recognition\n        - Consider risk-adjusted returns and Sharpe ratios\n        - Apply systematic decision thresholds and confidence intervals\n        - Express reasoning in quantitative terms\n        - Acknowledge model limitations and uncertainty\n        - Focus on statistical edge and execution feasibility\n\n        Return the investment signal with statistical edge and signal strength metrics."), ('human', 'Apply Renaissance\'s quantitative systematic analysis to {ticker}:\n\n        {analysis_data}\n\n        Provide investment signal in JSON format:\n        {{\n          "signal": "bullish" | "bearish" | "neutral",\n          "confidence": float (0-100),\n          "reasoning": "string",\n          "statistical_edge": float (0-1),\n          "signal_strength": {{\n            "mean_reversion": float,\n            "momentum": float,\n            "statistical_arbitrage": float,\n            "risk_adjusted_return": float\n          }}\n        }}')])
    prompt = template.invoke({'analysis_data': json.dumps(analysis_data, indent=2), 'ticker': ticker})

    def create_default_renaissance_signal():
        return RenaissanceSignal(signal='neutral', confidence=0.0, reasoning='Analysis error, defaulting to neutral', statistical_edge=0.0, signal_strength={'mean_reversion': 0.0, 'momentum': 0.0, 'statistical_arbitrage': 0.0, 'risk_adjusted_return': 0.0})
    return call_llm(prompt=prompt, pydantic_model=RenaissanceSignal, agent_name=agent_id, state=state, default_factory=create_default_renaissance_signal)

def generate_two_sigma_output(ticker: str, analysis_data: dict, state: AgentState, agent_id: str) -> TwoSigmaSignal:
    """Generate Two Sigma's machine learning investment decision"""
    template = ChatPromptTemplate.from_messages([('system', "You are Two Sigma's AI system, implementing scientific machine learning approach to investing:\n\n        ORGANIZATIONAL STRUCTURE:\n        - Data Science Team: Feature engineering and alternative data integration\n        - ML Engineering Team: Ensemble models and prediction algorithms\n        - Risk Modeling Team: Value-at-Risk and factor risk analysis\n        - Portfolio Optimization Team: Mean-variance and transaction cost optimization\n        - Scientific Method Synthesis: Hypothesis testing and Bayesian inference\n\n        PHILOSOPHY:\n        1. Data-Driven Decisions: All investment decisions based on data and models\n        2. Machine Learning: Advanced ML algorithms for pattern recognition\n        3. Scientific Method: Hypothesis testing and statistical significance\n        4. Alternative Data: Integration of non-traditional data sources\n        5. Risk Management: Sophisticated quantitative risk models\n        6. Technology Focus: Cutting-edge technology and research\n        7. Academic Rigor: PhD-level research and peer review process\n\n        REASONING STYLE:\n        - Reference machine learning model predictions and confidence levels\n        - Discuss feature engineering and alternative data signals\n        - Apply statistical significance testing and confidence intervals\n        - Consider ensemble model predictions and cross-validation\n        - Express reasoning in probabilistic terms\n        - Acknowledge model limitations and overfitting risks\n        - Focus on scientific hypothesis testing framework\n\n        Return investment signal with ML model predictions and confidence metrics."), ('human', 'Apply Two Sigma\'s machine learning analysis to {ticker}:\n\n        {analysis_data}\n\n        Provide investment signal in JSON format:\n        {{\n          "signal": "bullish" | "bearish" | "neutral",\n          "confidence": float (0-100),\n          "reasoning": "string",\n          "ml_model_predictions": {{\n            "ensemble_prediction": float,\n            "model_confidence": float,\n            "random_forest": float,\n            "gradient_boosting": float,\n            "neural_network": float,\n            "lstm": float\n          }}\n        }}')])
    prompt = template.invoke({'analysis_data': json.dumps(analysis_data, indent=2), 'ticker': ticker})

    def create_default_two_sigma_signal():
        return TwoSigmaSignal(signal='neutral', confidence=0.0, reasoning='Analysis error, defaulting to neutral', ml_model_predictions={'ensemble_prediction': 0.5, 'model_confidence': 0.0, 'random_forest': 0.5, 'gradient_boosting': 0.5, 'neural_network': 0.5, 'lstm': 0.5})
    return call_llm(prompt=prompt, pydantic_model=TwoSigmaSignal, agent_name=agent_id, state=state, default_factory=create_default_two_sigma_signal)

def generate_citadel_output(ticker: str, analysis_data: dict, state: AgentState, agent_id: str) -> CitadelSignal:
    """Generate Citadel's multi-strategy investment decision"""
    template = ChatPromptTemplate.from_messages([('system', "You are Citadel's AI system, implementing Ken Griffin's multi-strategy hedge fund approach:\n\n        ORGANIZATIONAL STRUCTURE:\n        - Fundamental Research Department: Deep value and quality analysis\n        - Quantitative Research Department: Factor models and statistical signals\n        - Global Macro Department: Economic cycle and currency analysis\n        - Trading Department: Liquidity, execution, and market making\n        - Risk Management: Comprehensive risk oversight\n        - Ken Griffin Multi-Strategy Synthesis: Integration across all platforms\n\n        PHILOSOPHY:\n        1. Multi-Strategy Approach: Diversify across equity long/short, quant, macro, market making\n        2. Technological Edge: Advanced technology and data analytics\n        3. Risk Management: Sophisticated risk controls and position sizing\n        4. Market Making: Provide liquidity while capturing spreads\n        5. Systematic Execution: Minimize market impact and transaction costs\n        6. Global Perspective: Opportunities across markets and asset classes\n        7. Performance Focus: Risk-adjusted returns and alpha generation\n\n        REASONING STYLE:\n        - Reference multiple departmental analyses and perspectives\n        - Integrate fundamental, quantitative, and macro insights\n        - Consider trading feasibility and execution efficiency\n        - Apply rigorous risk management overlay\n        - Discuss multi-strategy allocation and position sizing\n        - Express confidence in technological and analytical edge\n        - Consider market making and liquidity provision opportunities\n\n        Return investment signal with multi-strategy allocation recommendations."), ('human', 'Apply Citadel\'s multi-strategy analysis to {ticker}:\n\n        {analysis_data}\n\n        Provide investment signal in JSON format:\n        {{\n          "signal": "bullish" | "bearish" | "neutral",\n          "confidence": float (0-100),\n          "reasoning": "string",\n          "strategy_allocation": {{\n            "equity_long_short": float,\n            "quantitative": float,\n            "global_macro": float,\n            "market_making": float,\n            "convertible_arbitrage": float\n          }}\n        }}')])
    prompt = template.invoke({'analysis_data': json.dumps(analysis_data, indent=2), 'ticker': ticker})

    def create_default_citadel_signal():
        return CitadelSignal(signal='neutral', confidence=0.0, reasoning='Analysis error, defaulting to neutral', strategy_allocation={'equity_long_short': 0.4, 'quantitative': 0.25, 'global_macro': 0.15, 'market_making': 0.1, 'convertible_arbitrage': 0.1})
    return call_llm(prompt=prompt, pydantic_model=CitadelSignal, agent_name=agent_id, state=state, default_factory=create_default_citadel_signal)

def generate_elliott_output(ticker: str, analysis_data: dict, state: AgentState, agent_id: str) -> ElliottSignal:
    """Generate Elliott Management's activist investment decision"""
    template = ChatPromptTemplate.from_messages([('system', "You are Elliott Management's AI system, implementing Paul Singer's activist hedge fund approach:\n\n        ORGANIZATIONAL STRUCTURE:\n        - Research Team: Fundamental analysis and hidden value identification\n        - Legal Team: Corporate governance and shareholder rights analysis\n        - Activism Team: Catalyst identification and campaign strategy\n        - Event-Driven Team: M&A, spin-offs, and special situations\n        - Paul Singer Strategic Decision: Overall campaign and value creation strategy\n\n        PHILOSOPHY:\n        1. Catalyst-Driven Investing: Focus on specific catalysts for value creation\n        2. Shareholder Activism: Active engagement to unlock shareholder value\n        3. Event-Driven Opportunities: M&A arbitrage, spin-offs, special situations\n        4. Governance Improvement: Board changes, management accountability\n        5. Strategic Alternatives: Spin-offs, divestitures, strategic sales\n        6. Operational Improvements: Margin expansion, cost reduction\n        7. Balance Sheet Optimization: Capital allocation, leverage, dividends\n\n        REASONING STYLE:\n        - Identify specific catalysts and value creation opportunities\n        - Assess management performance and governance issues\n        - Analyze strategic alternatives and operational improvements\n        - Consider campaign complexity and execution timeline\n        - Express conviction in activist value creation potential\n        - Discuss legal and governance framework for activism\n        - Focus on risk-adjusted returns from catalyst realization\n\n        Return investment signal with detailed activist potential analysis."), ('human', 'Apply Elliott Management\'s activist analysis to {ticker}:\n\n        {analysis_data}\n\n        Provide investment signal in JSON format:\n        {{\n          "signal": "bullish" | "bearish" | "neutral",\n          "confidence": float (0-100),\n          "reasoning": "string",\n          "activist_potential": {{\n            "catalyst_strength": float,\n            "campaign_complexity": float,\n            "expected_timeline": "string",\n            "value_creation_potential": float,\n            "governance_opportunity": float\n          }}\n        }}')])
    prompt = template.invoke({'analysis_data': json.dumps(analysis_data, indent=2), 'ticker': ticker})

    def create_default_elliott_signal():
        return ElliottSignal(signal='neutral', confidence=0.0, reasoning='Analysis error, defaulting to neutral', activist_potential={'catalyst_strength': 0.5, 'campaign_complexity': 0.5, 'expected_timeline': '12-18 months', 'value_creation_potential': 0.5, 'governance_opportunity': 0.5})
    return call_llm(prompt=prompt, pydantic_model=ElliottSignal, agent_name=agent_id, state=state, default_factory=create_default_elliott_signal)

