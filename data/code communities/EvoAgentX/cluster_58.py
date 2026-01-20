# Cluster 58

def quick_fetch_catl_data():
    """
    快速抓取宁德时代数据的便捷函数（向后兼容）
    
    Returns:
        dict: 包含所有数据的字典
    """
    return fetch_stock_data('300750')

def fetch_stock_data(stock_code, output_dir=None):
    """
    快速抓取指定股票的全部数据（主要函数）
    
    Args:
        stock_code (str): 股票代码（如：300750、000001、000858等）
        output_dir (str): 输出目录路径（可选）
        
    Returns:
        dict: 包含所有数据的字典
        
    输出文件夹格式: output_股票代码/ 或指定的output_dir
    包含文件:
    - 10个CSV数据文件
    - 1个数据文件说明.md
    """
    print(f'🚀 开始抓取股票 {stock_code} 的全部数据...')
    fetcher = StockDataFetcher(stock_code=stock_code, auto_create_output_dir=False)
    if output_dir:
        fetcher.output_dir = Path(output_dir)
        fetcher.output_dir.mkdir(exist_ok=True)
    return fetcher.fetch_all_data()

def fetch_single_data_type(stock_code, data_type):
    """
    抓取指定股票的单一类型数据
    
    Args:
        stock_code (str): 股票代码
        data_type (str): 数据类型 ('stock_daily', 'cpi', 'gdp', 'industry_fund', 
                                   'news', 'market_summary', 'indices', 'volatility', 'rating')
        
    Returns:
        pandas.DataFrame: 抓取的数据
    """
    fetcher = StockDataFetcher(stock_code=stock_code)
    data_map = {'stock_daily': fetcher.fetch_stock_daily, 'cpi': fetcher.fetch_china_cpi, 'gdp': fetcher.fetch_china_gdp, 'industry_fund': fetcher.fetch_industry_fund_flow, 'news': fetcher.fetch_stock_news, 'market_summary': fetcher.fetch_market_summary, 'indices': fetcher.fetch_market_indices, 'volatility': fetcher.fetch_option_volatility, 'rating': fetcher.fetch_institution_recommendation}
    if data_type in data_map:
        result = data_map[data_type]()
        if result is not None:
            filename_mapping = {'stock_daily': 'stock_daily_catl', 'cpi': 'china_cpi', 'gdp': 'china_gdp_yearly', 'industry_fund': 'industry_fund_flow', 'news': 'stock_news_catl', 'market_summary': 'market_summary_sse', 'indices': 'market_indices', 'volatility': 'option_volatility_50etf', 'rating': 'institution_recommendation_catl'}
            fetcher.save_data(result, filename_mapping[data_type], f'{data_type}数据')
        return result
    else:
        print(f'❌ 不支持的数据类型: {data_type}')
        print(f'支持的类型: {list(data_map.keys())}')
        return None

def batch_generate_charts(symbols: List[str], output_base_dir: str='charts') -> Dict[str, Dict]:
    """
    批量生成多个股票的图表
    
    Args:
        symbols (List[str]): 股票代码列表
        output_base_dir (str): 基础输出目录
        
    Returns:
        Dict[str, Dict]: 每个股票的生成结果
        
    Example:
        symbols = ["300750", "600519", "000001"]
        results = batch_generate_charts(symbols)
    """
    results = {}
    print(f'🚀 批量生成 {len(symbols)} 个股票的图表')
    print('=' * 60)
    for i, symbol in enumerate(symbols, 1):
        print(f'\n📈 [{i}/{len(symbols)}] 处理股票: {symbol}')
        print('-' * 40)
        try:
            stock_output_dir = os.path.join(output_base_dir, f'stock_{symbol}')
            chart_paths = generate_stock_charts(symbol=symbol, output_dir=stock_output_dir, chart_types=['technical', 'candlestick'])
            results[symbol] = {'status': 'success', 'charts': chart_paths, 'output_dir': stock_output_dir}
        except Exception as e:
            print(f'❌ 生成失败: {e}')
            results[symbol] = {'status': 'failed', 'error': str(e), 'charts': {}, 'output_dir': None}
    print('\n' + '=' * 60)
    print('📋 批量生成结果汇总')
    print('=' * 60)
    success_count = 0
    for symbol, result in results.items():
        if result['status'] == 'success':
            success_count += 1
            print(f'✅ {symbol}: 成功生成 {len(result['charts'])} 个图表')
        else:
            print(f'❌ {symbol}: {result.get('error', '未知错误')}')
    print(f'\n🎉 批量生成完成: {success_count}/{len(symbols)} 成功')
    return results

def generate_stock_charts(symbol: str='300750', output_dir: str='output', chart_types: List[str]=None) -> Dict[str, Optional[str]]:
    """
    生成股票技术分析图表的主函数
    
    Args:
        symbol (str): 股票代码（如：300750、000001、000858等）
        output_dir (str): 输出目录，默认为"output"
        chart_types (List[str]): 图表类型列表，可选 "technical", "candlestick"
                                默认生成所有类型
        
    Returns:
        Dict[str, Optional[str]]: 生成的图表路径字典
        
    Example:
        # 生成宁德时代的所有图表
        charts = generate_stock_charts("300750")
        
        # 只生成K线图
        charts = generate_stock_charts("600519", chart_types=["candlestick"])
        
        # 生成到指定目录
        charts = generate_stock_charts("000001", output_dir="my_charts")
    """
    if chart_types is None:
        chart_types = ['technical', 'candlestick']
    generator = StockChartGenerator(symbol, output_dir)
    if set(chart_types) == {'technical', 'candlestick'}:
        return generator.generate_all_charts()
    print(f'🚀 生成股票 {symbol} 的指定图表类型')
    print('=' * 60)
    chart_paths = {}
    df = generator.get_stock_data()
    if df is None:
        print('❌ 无法获取数据')
        return {}
    generator.calculate_indicators(df)
    if 'technical' in chart_types:
        print('📊 生成技术分析图表...')
        technical_path = generator.create_technical_chart()
        if technical_path:
            chart_paths['technical'] = technical_path
    if 'candlestick' in chart_types:
        print('🕯️ 生成K线图...')
        candlestick_path = generator.create_candlestick_chart()
        if candlestick_path:
            chart_paths['candlestick'] = candlestick_path
    if chart_paths:
        print(f'✅ 图表生成成功:')
        for chart_type, path in chart_paths.items():
            print(f'   {chart_type}: {os.path.abspath(path)}')
    else:
        print('❌ 图表生成失败')
    return chart_paths

def generate_html_from_existing_files(stock_code, timestamp=None):
    """Generate HTML report from existing markdown and chart files"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d')
    base_dir, data_dir, report_dir, graphs_dir = get_directories(stock_code, timestamp)
    print(f'🔍 查找现有文件:')
    print(f'   报告目录: {report_dir}')
    print(f'   图表目录: {graphs_dir}')
    if not report_dir.exists():
        print(f'❌ 报告目录不存在: {report_dir}')
        return False
    if not graphs_dir.exists():
        print(f'⚠️  图表目录不存在: {graphs_dir}')
        graphs_dir = None
    return generate_html_report(stock_code, base_dir, report_dir, graphs_dir, timestamp)

def get_directories(stock_code, timestamp):
    """Get directory paths for a given stock code and timestamp"""
    base_dir = Path(f'./{stock_code}')
    data_dir = base_dir / timestamp / 'data'
    report_dir = base_dir / 'reports'
    graphs_dir = base_dir / timestamp / 'graphs'
    return (base_dir, data_dir, report_dir, graphs_dir)

def generate_html_report(stock_code, base_dir, report_dir, graphs_dir, timestamp):
    """Generate HTML report from markdown and charts"""
    try:
        from html_report_generator import HTMLGenerator
        md_file = report_dir / f'text_report_{stock_code}_{timestamp}.md'
        html_output = base_dir / datetime.now().strftime('%Y%m%d') / 'html_report' / f'report_{stock_code}_{timestamp}.html'
        technical_chart = graphs_dir / f'{stock_code}_technical_charts.png'
        price_volume_chart = graphs_dir / f'{stock_code}_candlestick_chart.png'
        if not md_file.exists():
            print(f'❌ Markdown file not found: {md_file}')
            return False
        if not technical_chart.exists():
            print(f'⚠️  Technical chart not found: {technical_chart}')
            technical_chart = ''
        if not price_volume_chart.exists():
            print(f'⚠️  Price/volume chart not found: {price_volume_chart}')
            price_volume_chart = ''
        print(f'[4] 生成HTML报告: {html_output}')
        generator = HTMLGenerator(str(html_output))
        output_file = generator.generate_report(str(md_file), str(technical_chart) if technical_chart else '', str(price_volume_chart) if price_volume_chart else '')
        print(f'✅ HTML报告生成成功: {output_file}')
        print(f'📁 资源文件夹: {Path(output_file).parent / 'assets'}')
        print(f'🌐 在浏览器中打开HTML文件查看报告')
        return True
    except Exception as e:
        print(f'❌ HTML报告生成失败: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        stock_code = input('请输入股票代码 (如300750): ').strip()
    else:
        stock_code = sys.argv[1].strip()
    if not stock_code.isdigit():
        print('❌ 股票代码应为数字！')
        return
    timestamp = datetime.now().strftime('%Y%m%d')
    base_dir, data_dir, report_dir, graphs_dir = get_directories(stock_code, timestamp)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    if not check_data_exists(data_dir):
        print(f'\n[1] 拉取数据到: {data_dir}')
        fetch_stock_data(stock_code, output_dir=str(data_dir))
    else:
        print(f'\n[1] 跳过数据拉取 (数据已存在)')
    if not check_charts_exist(graphs_dir, stock_code):
        print(f'[2] 生成图表到: {graphs_dir}')
        generate_stock_charts(stock_code, output_dir=str(graphs_dir))
    else:
        print(f'[2] 跳过图表生成 (图表已存在)')
    print(f'[3] 生成报告到: {report_dir}')
    execute_workflow(stock_code, data_dir, report_dir, timestamp)
    print(f'\n[4] 生成HTML报告')
    html_success = generate_html_report(stock_code, base_dir, report_dir, graphs_dir, timestamp)
    if html_success:
        print('\n✅ 全部流程完成！包括HTML报告生成')
    else:
        print('\n✅ 主要流程完成！(HTML报告生成失败)')

def check_data_exists(data_dir):
    """Check if data files already exist in the data directory"""
    if not data_dir.exists():
        return False
    expected_files = ['stock_daily_catl_*.csv', 'china_cpi_*.csv', 'china_gdp_yearly_*.csv', 'industry_fund_flow_*.csv', 'stock_news_catl_*.csv', 'market_summary_sse_*.csv', 'market_indices_*.csv', 'option_volatility_50etf_*.csv', 'institution_recommendation_catl_*.csv']
    existing_files = list(data_dir.glob('*.csv'))
    if len(existing_files) >= 5:
        print(f'✅ 数据文件已存在: {data_dir}')
        print(f'   发现 {len(existing_files)} 个数据文件')
        return True
    return False

def check_charts_exist(graphs_dir, stock_code):
    """Check if chart files already exist"""
    if not graphs_dir.exists():
        return False
    expected_charts = [f'{stock_code}_technical_charts.png', f'{stock_code}_candlestick_chart.png']
    existing_charts = [f.name for f in graphs_dir.glob('*.png')]
    if all((chart in existing_charts for chart in expected_charts)):
        print(f'✅ 图表文件已存在: {graphs_dir}')
        print(f'   发现 {len(existing_charts)} 个图表文件')
        return True
    return False

def quick_fetch_catl_data():
    """
    快速抓取宁德时代数据的便捷函数（向后兼容）
    
    Returns:
        dict: 包含所有数据的字典
    """
    return fetch_stock_data('300750')

def batch_generate_charts(symbols: List[str], output_base_dir: str='charts') -> Dict[str, Dict]:
    """
    批量生成多个股票的图表
    
    Args:
        symbols (List[str]): 股票代码列表
        output_base_dir (str): 基础输出目录
        
    Returns:
        Dict[str, Dict]: 每个股票的生成结果
        
    Example:
        symbols = ["300750", "600519", "000001"]
        results = batch_generate_charts(symbols)
    """
    results = {}
    print(f'🚀 批量生成 {len(symbols)} 个股票的图表')
    print('=' * 60)
    for i, symbol in enumerate(symbols, 1):
        print(f'\n📈 [{i}/{len(symbols)}] 处理股票: {symbol}')
        print('-' * 40)
        try:
            stock_output_dir = os.path.join(output_base_dir, f'stock_{symbol}')
            chart_paths = generate_stock_charts(symbol=symbol, output_dir=stock_output_dir, chart_types=['technical', 'candlestick'])
            results[symbol] = {'status': 'success', 'charts': chart_paths, 'output_dir': stock_output_dir}
        except Exception as e:
            print(f'❌ 生成失败: {e}')
            results[symbol] = {'status': 'failed', 'error': str(e), 'charts': {}, 'output_dir': None}
    print('\n' + '=' * 60)
    print('📋 批量生成结果汇总')
    print('=' * 60)
    success_count = 0
    for symbol, result in results.items():
        if result['status'] == 'success':
            success_count += 1
            print(f'✅ {symbol}: 成功生成 {len(result['charts'])} 个图表')
        else:
            print(f'❌ {symbol}: {result.get('error', '未知错误')}')
    print(f'\n🎉 批量生成完成: {success_count}/{len(symbols)} 成功')
    return results

def generate_html_from_existing_files(stock_code, timestamp=None):
    """Generate HTML report from existing markdown and chart files"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d')
    base_dir, data_dir, report_dir, graphs_dir = get_directories(stock_code, timestamp)
    print(f'🔍 查找现有文件:')
    print(f'   报告目录: {report_dir}')
    print(f'   图表目录: {graphs_dir}')
    if not report_dir.exists():
        print(f'❌ 报告目录不存在: {report_dir}')
        return False
    if not graphs_dir.exists():
        print(f'⚠️  图表目录不存在: {graphs_dir}')
        graphs_dir = None
    return generate_html_report(stock_code, base_dir, report_dir, graphs_dir, timestamp)

def main():
    if len(sys.argv) < 2:
        stock_code = input('请输入股票代码 (如300750): ').strip()
    else:
        stock_code = sys.argv[1].strip()
    if not stock_code.isdigit():
        print('❌ 股票代码应为数字！')
        return
    timestamp = datetime.now().strftime('%Y%m%d')
    base_dir, data_dir, report_dir, graphs_dir = get_directories(stock_code, timestamp)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    if not check_data_exists(data_dir):
        print(f'\n[1] 拉取数据到: {data_dir}')
        fetch_stock_data(stock_code, output_dir=str(data_dir))
    else:
        print(f'\n[1] 跳过数据拉取 (数据已存在)')
    if not check_charts_exist(graphs_dir, stock_code):
        print(f'[2] 生成图表到: {graphs_dir}')
        generate_stock_charts(stock_code, output_dir=str(graphs_dir))
    else:
        print(f'[2] 跳过图表生成 (图表已存在)')
    print(f'[3] 生成报告到: {report_dir}')
    execute_workflow(stock_code, data_dir, report_dir, timestamp)
    print(f'\n[4] 生成HTML报告')
    html_success = generate_html_report(stock_code, base_dir, report_dir, graphs_dir, timestamp)
    if html_success:
        print('\n✅ 全部流程完成！包括HTML报告生成')
    else:
        print('\n✅ 主要流程完成！(HTML报告生成失败)')

