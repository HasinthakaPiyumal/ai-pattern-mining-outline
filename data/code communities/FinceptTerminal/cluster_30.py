# Cluster 30

def execute_stock_data_node(node_id, ticker, period):
    print(f'Fetching data for {ticker}, period: {period}')
    data = fetch_stock_data(ticker, period)
    if data is not None:
        node_outputs[node_id] = data
        dpg.set_value(f'{node_id}_status', '✓ Data loaded')
        print(f'Data loaded successfully for node {node_id}')
    else:
        dpg.set_value(f'{node_id}_status', '✗ Failed to load')
        print(f'Failed to load data for node {node_id}')

def fetch_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data
    except Exception as e:
        print(f'Error fetching data: {e}')
        return None

def execute_advanced_signal_node(node_id, indicator1_data, indicator2_data, condition, entry_rule, exit_rule, risk_reward_ratio=2.0, stop_loss_pct=2.0):
    print(f'Executing Advanced Signal node {node_id}')
    base_data = get_base_stock_data()
    if base_data is None:
        dpg.set_value(f'{node_id}_status', '✗ No base stock data found')
        return
    if indicator1_data is None or indicator2_data is None:
        dpg.set_value(f'{node_id}_status', '✗ Need both indicators')
        return
    try:
        ind1_df = standardize_data_format(indicator1_data, 'ind1')
        ind2_df = standardize_data_format(indicator2_data, 'ind2')
        if ind1_df is None or ind2_df is None:
            dpg.set_value(f'{node_id}_status', '✗ Invalid indicator data')
            return
        ind1_col = ind1_df.columns[0]
        ind2_col = ind2_df.columns[0]
        combined_data = pd.concat([base_data['Open'], base_data['High'], base_data['Low'], base_data['Close'], ind1_df[ind1_col], ind2_df[ind2_col]], axis=1).dropna()
        combined_data.columns = ['Open', 'High', 'Low', 'Close', 'Indicator1', 'Indicator2']
        if combined_data.empty:
            dpg.set_value(f'{node_id}_status', '✗ No overlapping data')
            return
        print(f'Signal generation using {ind1_col} vs {ind2_col}, condition: {condition}')
        print(f'Data shape: {combined_data.shape}, Date range: {combined_data.index[0]} to {combined_data.index[-1]}')
        signals = combined_data.copy()
        signals['signal'] = 0
        signals['position'] = 0
        signals['entry_price'] = 0.0
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        signals['exit_reason'] = ''
        entry_signals = pd.Series(False, index=signals.index)
        if condition == 'Crossover':
            if entry_rule == 'Buy on Signal':
                entry_signals = (signals['Indicator1'] > signals['Indicator2']) & (signals['Indicator1'].shift(1) <= signals['Indicator2'].shift(1))
                print(f'Crossover logic: Fast MA > Slow MA with previous Fast MA <= previous Slow MA')
            elif entry_rule == 'Sell on Signal':
                entry_signals = (signals['Indicator1'] < signals['Indicator2']) & (signals['Indicator1'].shift(1) >= signals['Indicator2'].shift(1))
        elif condition == 'Greater Than':
            if entry_rule == 'Buy on Signal':
                entry_signals = (signals['Indicator1'] > signals['Indicator2']) & (signals['Indicator1'].shift(1) <= signals['Indicator2'].shift(1))
            else:
                entry_signals = (signals['Indicator1'] < signals['Indicator2']) & (signals['Indicator1'].shift(1) >= signals['Indicator2'].shift(1))
        elif condition == 'RSI Levels':
            if entry_rule == 'Buy on Signal':
                entry_signals = (signals['Indicator1'] > 30) & (signals['Indicator1'].shift(1) <= 30)
            else:
                entry_signals = (signals['Indicator1'] < 70) & (signals['Indicator1'].shift(1) >= 70)
        elif condition == 'Bollinger Breakout':
            if entry_rule == 'Buy on Signal':
                entry_signals = signals['Close'] > signals['Indicator2']
            else:
                entry_signals = signals['Close'] < signals['Indicator2']
        potential_entries = entry_signals.sum()
        print(f'Found {potential_entries} potential entry signals')
        in_position = False
        entry_price = 0.0
        stop_loss_price = 0.0
        take_profit_price = 0.0
        position_type = 0
        trade_count = 0
        signal_array = signals['signal'].values
        position_array = signals['position'].values
        entry_price_array = signals['entry_price'].values
        stop_loss_array = signals['stop_loss'].values
        take_profit_array = signals['take_profit'].values
        exit_reason_array = signals['exit_reason'].values.astype(object)
        for i in range(1, len(signals)):
            current_idx = signals.index[i]
            current_price = signals.iloc[i]['Close']
            current_high = signals.iloc[i]['High']
            current_low = signals.iloc[i]['Low']
            if not in_position and entry_signals.iloc[i]:
                entry_price = current_price
                position_type = 1 if entry_rule == 'Buy on Signal' else -1
                trade_count += 1
                if position_type == 1:
                    stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                    take_profit_price = entry_price * (1 + stop_loss_pct * risk_reward_ratio / 100)
                else:
                    stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
                    take_profit_price = entry_price * (1 - stop_loss_pct * risk_reward_ratio / 100)
                signal_array[i] = position_type
                entry_price_array[i] = entry_price
                stop_loss_array[i] = stop_loss_price
                take_profit_array[i] = take_profit_price
                in_position = True
                print(f'ENTRY #{trade_count}: {entry_rule} at ${entry_price:.2f} on {current_idx.date()}')
                print(f'  SL: ${stop_loss_price:.2f} (-{stop_loss_pct}%), TP: ${take_profit_price:.2f} (+{stop_loss_pct * risk_reward_ratio}%)')
            elif in_position:
                should_exit = False
                exit_reason = ''
                exit_price = current_price
                if position_type == 1:
                    if current_low <= stop_loss_price:
                        should_exit = True
                        exit_reason = 'Stop Loss'
                        exit_price = stop_loss_price
                    elif current_high >= take_profit_price:
                        should_exit = True
                        exit_reason = 'Take Profit'
                        exit_price = take_profit_price
                elif position_type == -1:
                    if current_high >= stop_loss_price:
                        should_exit = True
                        exit_reason = 'Stop Loss'
                        exit_price = stop_loss_price
                    elif current_low <= take_profit_price:
                        should_exit = True
                        exit_reason = 'Take Profit'
                        exit_price = take_profit_price
                if not should_exit and exit_rule == 'Opposite Signal':
                    if condition == 'Crossover':
                        if position_type == 1:
                            if signals.iloc[i]['Indicator1'] < signals.iloc[i]['Indicator2'] and signals.iloc[i - 1]['Indicator1'] >= signals.iloc[i - 1]['Indicator2']:
                                should_exit = True
                                exit_reason = 'Opposite Signal'
                        elif signals.iloc[i]['Indicator1'] > signals.iloc[i]['Indicator2'] and signals.iloc[i - 1]['Indicator1'] <= signals.iloc[i - 1]['Indicator2']:
                            should_exit = True
                            exit_reason = 'Opposite Signal'
                if should_exit:
                    signal_array[i] = -position_type
                    exit_reason_array[i] = exit_reason
                    in_position = False
                    pnl = (exit_price - entry_price) * position_type
                    pnl_pct = pnl / entry_price * 100
                    print(f'EXIT #{trade_count}: {exit_reason} at ${exit_price:.2f} on {current_idx.date()}')
                    print(f'  P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)')
                    position_type = 0
        if in_position:
            final_idx = signals.index[-1]
            final_price = signals.iloc[-1]['Close']
            signal_array[-1] = -position_type
            exit_reason_array[-1] = 'End of Data'
            pnl = (final_price - entry_price) * position_type
            pnl_pct = pnl / entry_price * 100
            print(f'FINAL EXIT #{trade_count}: End of data at ${final_price:.2f}')
            print(f'  P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)')
        signals.loc[:, 'signal'] = signal_array
        signals.loc[:, 'entry_price'] = entry_price_array
        signals.loc[:, 'stop_loss'] = stop_loss_array
        signals.loc[:, 'take_profit'] = take_profit_array
        signals.loc[:, 'exit_reason'] = exit_reason_array
        current_pos = 0
        for i in range(len(signals)):
            if signal_array[i] != 0:
                if current_pos == 0:
                    current_pos = signal_array[i]
                else:
                    current_pos = 0
            position_array[i] = current_pos
        signals.loc[:, 'position'] = position_array
        signals['position_change'] = signals['position'].diff().fillna(0)
        signals['strategy'] = f'{ind1_col}_{condition}_{ind2_col}'
        signals['entry_rule'] = entry_rule
        signals['exit_rule'] = exit_rule
        signals['risk_reward'] = risk_reward_ratio
        signals['stop_loss_pct'] = stop_loss_pct
        buy_signals_generated = (signals['signal'] == 1).sum()
        sell_signals_generated = (signals['signal'] == -1).sum()
        print(f'DEBUG: Generated {buy_signals_generated} buy signals and {sell_signals_generated} sell signals')
        signal_rows = signals[signals['signal'] != 0][['Close', 'signal', 'entry_price', 'stop_loss', 'take_profit', 'exit_reason']].head(10)
        print(f'DEBUG: First 10 signal rows:')
        print(signal_rows)
        node_outputs[node_id] = signals
        entries = len(signals[signals['signal'] > 0])
        exits = len(signals[signals['signal'] < 0])
        actual_trades = min(entries, exits)
        dpg.set_value(f'{node_id}_status', f'✓ {actual_trades} complete trades generated')
        print(f'Signal generation complete: {actual_trades} trades from {entries} entries and {exits} exits')
    except Exception as e:
        dpg.set_value(f'{node_id}_status', f'✗ Error: {str(e)[:30]}')
        print(f'Error in advanced signal generation for node {node_id}: {e}')
        import traceback
        traceback.print_exc()

def get_base_stock_data():
    """Get the original OHLCV stock data"""
    for node_id, data in node_outputs.items():
        if isinstance(data, pd.DataFrame) and all((col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])):
            return data
    return None

def standardize_data_format(data, name='data'):
    """Convert any data format to a standardized DataFrame with proper index"""
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, pd.Series):
        df = pd.DataFrame({name: data})
        return df
    else:
        print(f'Unsupported data type: {type(data)}')
        return None

def show_comprehensive_plot(node_id):
    print(f'Showing comprehensive plot for node {node_id}')
    all_data = get_all_input_data(node_id)
    stock_data = find_stock_data()
    if not all_data and stock_data is None:
        dpg.set_value(f'{node_id}_status', '✗ No input data')
        print(f'No input data for plot node {node_id}')
        return
    if dpg.does_item_exist('comprehensive_plot_window'):
        dpg.delete_item('comprehensive_plot_window')
    with dpg.window(label='Comprehensive Chart', width=1000, height=700, pos=[50, 50], tag='comprehensive_plot_window'):
        with dpg.plot(label='Stock Analysis', height=600, width=950):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label='Date')
            dpg.add_plot_axis(dpg.mvYAxis, label='Price', tag='price_y_axis')
            has_rsi = any(('RSI' in str(item['data'].columns).upper() if isinstance(item['data'], pd.DataFrame) else False for item in all_data))
            if has_rsi:
                dpg.add_plot_axis(dpg.mvYAxis, label='RSI (0-100)', tag='rsi_y_axis')
            close_price_plotted = False
            if stock_data is not None:
                dates = list(range(len(stock_data)))
                prices = stock_data['Close'].values.tolist()
                dpg.add_line_series(dates, prices, label='Close Price', parent='price_y_axis')
                close_price_plotted = True
                print('Added Close Price to plot')
            for item in all_data:
                data = item['data']
                source_id = item['source_id']
                if isinstance(data, pd.DataFrame):
                    if 'Close' in data.columns and close_price_plotted:
                        continue
                    for col in data.columns:
                        if col in ['Open', 'High', 'Low', 'Volume']:
                            continue
                        clean_data = data[col].dropna()
                        if len(clean_data) > 0:
                            if stock_data is not None:
                                start_idx = len(stock_data) - len(clean_data)
                                dates = list(range(start_idx, len(stock_data)))
                            else:
                                dates = list(range(len(clean_data)))
                            values = clean_data.values.tolist()
                            if 'RSI' in col.upper() and has_rsi:
                                dpg.add_line_series(dates, values, label=col, parent='rsi_y_axis')
                            else:
                                dpg.add_line_series(dates, values, label=col, parent='price_y_axis')
    total_series = len(all_data) + (1 if close_price_plotted else 0)
    dpg.set_value(f'{node_id}_status', f'✓ Plot with {total_series} series')
    print(f'Comprehensive plot displayed for node {node_id} with {total_series} data series')

def get_all_input_data(node_id):
    """Get all connected input data for a node"""
    print(f'Getting all input data for node {node_id}')
    if node_id not in node_connections:
        print(f'Node {node_id} has no input connections')
        return []
    all_data = []
    connections = node_connections[node_id]
    for input_type, source_node_ids in connections.items():
        for source_node_id in source_node_ids:
            if source_node_id in node_outputs:
                data = node_outputs[source_node_id]
                all_data.append({'data': data, 'source_id': source_node_id, 'input_type': input_type})
                print(f'Retrieved data from node {source_node_id} (type: {input_type})')
    return all_data

def find_stock_data():
    """Find the original stock data DataFrame from any stock data node"""
    for node_id, data in node_outputs.items():
        if isinstance(data, pd.DataFrame) and all((col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])):
            return data
    return None

def create_stock_data_node(sender, app_data, user_data):
    node_id = f'stock_{dpg.generate_uuid()}'
    dpg_node_id = dpg.generate_uuid()
    with dpg.node(label='Stock Data', pos=[50, 50], parent='node_editor', tag=dpg_node_id):
        with dpg.node_attribute(label='Settings', attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(width=150)
            dpg.add_input_text(label='Ticker', default_value='AAPL', width=100, tag=f'{node_id}_ticker')
            dpg.add_combo(['1mo', '3mo', '6mo', '1y', '2y', '5y'], default_value='1y', label='Period', width=100, tag=f'{node_id}_period')
            dpg.add_button(label='Load Data', width=100, callback=lambda: execute_stock_data_node(node_id, dpg.get_value(f'{node_id}_ticker'), dpg.get_value(f'{node_id}_period')))
            dpg.add_text('Status: Waiting', tag=f'{node_id}_status')
            dpg.add_button(label='Delete', width=100, callback=lambda: delete_node(dpg_node_id))
        output_attr = dpg.add_node_attribute(label='OHLCV Data', attribute_type=dpg.mvNode_Attr_Output)
        dpg.add_spacer(width=1, parent=output_attr)
    node_registry[dpg_node_id] = {'node_id': node_id, 'output_attr': output_attr}
    print(f'Created Stock Data node: {node_id} with output attr: {output_attr}')

def delete_node(dpg_node_id):
    """Delete a node and clean up all references"""
    if dpg_node_id not in node_registry:
        print(f'Node {dpg_node_id} not found in registry')
        return
    node_id = node_registry[dpg_node_id]['node_id']
    print(f'Deleting node {node_id} (dpg_id: {dpg_node_id})')
    if node_id in node_outputs:
        del node_outputs[node_id]
        print(f'Removed output data for {node_id}')
    if node_id in node_connections:
        del node_connections[node_id]
        print(f'Removed input connections for {node_id}')
    for target_node, connections in list(node_connections.items()):
        for input_type, source_list in list(connections.items()):
            if node_id in source_list:
                source_list.remove(node_id)
                print(f'Removed {node_id} as source for {target_node}')
            if not source_list:
                del connections[input_type]
        if not connections:
            del node_connections[target_node]
    links_to_remove = []
    for link_id, (source, target, input_type) in link_registry.items():
        if source == node_id or target == node_id:
            links_to_remove.append(link_id)
    for link_id in links_to_remove:
        del link_registry[link_id]
        print(f'Removed link {link_id} from registry')
    del node_registry[dpg_node_id]
    if dpg.does_item_exist(dpg_node_id):
        dpg.delete_item(dpg_node_id)
        print(f'Deleted DPG node {dpg_node_id}')
    status_items = [f'{node_id}_status', f'{node_id}_ticker', f'{node_id}_period', f'{node_id}_window', f'{node_id}_results_text', f'{node_id}_condition', f'{node_id}_entry_rule', f'{node_id}_exit_rule', f'{node_id}_capital', f'{node_id}_commission']
    for item in status_items:
        if dpg.does_item_exist(item):
            dpg.delete_item(item)
    dpg.set_value('connection_status', f'Deleted node: {node_id}')

def create_moving_average_node(sender, app_data, user_data):
    node_id = f'ma_{dpg.generate_uuid()}'
    dpg_node_id = dpg.generate_uuid()
    with dpg.node(label='Moving Average', pos=[250, 50], parent='node_editor', tag=dpg_node_id):
        input_attr = dpg.add_node_attribute(label='Price Data', attribute_type=dpg.mvNode_Attr_Input)
        dpg.add_spacer(width=1, parent=input_attr)
        with dpg.node_attribute(label='Settings', attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(width=150)
            dpg.add_input_int(label='Window', default_value=20, min_value=5, max_value=200, width=100, tag=f'{node_id}_window')
            dpg.add_button(label='Calculate', width=100, callback=lambda: execute_moving_average_node(node_id, get_input_data(node_id), dpg.get_value(f'{node_id}_window')))
            dpg.add_text('Status: Waiting', tag=f'{node_id}_status')
            dpg.add_button(label='Delete', width=100, callback=lambda: delete_node(dpg_node_id))
        output_attr = dpg.add_node_attribute(label='MA Values', attribute_type=dpg.mvNode_Attr_Output)
        dpg.add_spacer(width=1, parent=output_attr)
    node_registry[dpg_node_id] = {'node_id': node_id, 'input_attr': input_attr, 'output_attr': output_attr}
    print(f'Created MA node: {node_id} with input attr: {input_attr}, output attr: {output_attr}')

def execute_moving_average_node(node_id, input_data, window):
    print(f'Executing MA node {node_id} with window {window}')
    if input_data is not None and (not input_data.empty):
        if 'Close' in input_data.columns:
            ma = input_data['Close'].rolling(window=window).mean()
            ma.name = f'MA{window}'
            ma_df = pd.DataFrame({f'MA{window}': ma})
            node_outputs[node_id] = ma_df
            dpg.set_value(f'{node_id}_status', f'✓ MA{window} calculated')
            print(f'MA calculated successfully for node {node_id}')
        else:
            dpg.set_value(f'{node_id}_status', '✗ No Close price data')
            print(f'No Close price data for MA node {node_id}')
    else:
        dpg.set_value(f'{node_id}_status', '✗ No input data')
        print(f'No input data for MA node {node_id}')

def get_input_data(node_id, input_type='default'):
    print(f'Getting input data for node {node_id}, type: {input_type}')
    if node_id not in node_connections:
        print(f'Node {node_id} has no input connections')
        return None
    connections = node_connections[node_id]
    if input_type in connections and connections[input_type]:
        source_node_id = connections[input_type][-1]
        print(f'Found source node: {source_node_id}')
        if source_node_id in node_outputs:
            data = node_outputs[source_node_id]
            print(f'Retrieved data from node {source_node_id}')
            return data
        else:
            print(f'No data available in source node {source_node_id}')
            return None
    else:
        print(f'No source node found for input type {input_type}')
        return None

def create_rsi_node(sender, app_data, user_data):
    node_id = f'rsi_{dpg.generate_uuid()}'
    dpg_node_id = dpg.generate_uuid()
    with dpg.node(label='RSI Indicator', pos=[250, 200], parent='node_editor', tag=dpg_node_id):
        input_attr = dpg.add_node_attribute(label='Price Data', attribute_type=dpg.mvNode_Attr_Input)
        dpg.add_spacer(width=1, parent=input_attr)
        with dpg.node_attribute(label='Settings', attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(width=150)
            dpg.add_input_int(label='Period', default_value=14, min_value=5, max_value=50, width=100, tag=f'{node_id}_period')
            dpg.add_button(label='Calculate', width=100, callback=lambda: execute_rsi_node(node_id, get_input_data(node_id), dpg.get_value(f'{node_id}_period')))
            dpg.add_text('Status: Waiting', tag=f'{node_id}_status')
            dpg.add_button(label='Delete', width=100, callback=lambda: delete_node(dpg_node_id))
        output_attr = dpg.add_node_attribute(label='RSI Values', attribute_type=dpg.mvNode_Attr_Output)
        dpg.add_spacer(width=1, parent=output_attr)
    node_registry[dpg_node_id] = {'node_id': node_id, 'input_attr': input_attr, 'output_attr': output_attr}
    print(f'Created RSI node: {node_id} with input attr: {input_attr}, output attr: {output_attr}')

def execute_rsi_node(node_id, input_data, period=14):
    print(f'Executing RSI node {node_id} with period {period}')
    if input_data is not None and (not input_data.empty):
        if 'Close' in input_data.columns:
            close_prices = input_data['Close']
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - 100 / (1 + rs)
            rsi_df = pd.DataFrame({f'RSI{period}': rsi})
            node_outputs[node_id] = rsi_df
            dpg.set_value(f'{node_id}_status', f'✓ RSI{period} calculated')
            print(f'RSI calculated successfully for node {node_id}')
        else:
            dpg.set_value(f'{node_id}_status', '✗ No Close price data')
            print(f'No Close price data for RSI node {node_id}')
    else:
        dpg.set_value(f'{node_id}_status', '✗ No input data')
        print(f'No input data for RSI node {node_id}')

def create_bollinger_bands_node(sender, app_data, user_data):
    node_id = f'bb_{dpg.generate_uuid()}'
    dpg_node_id = dpg.generate_uuid()
    with dpg.node(label='Bollinger Bands', pos=[250, 350], parent='node_editor', tag=dpg_node_id):
        input_attr = dpg.add_node_attribute(label='Price Data', attribute_type=dpg.mvNode_Attr_Input)
        dpg.add_spacer(width=1, parent=input_attr)
        with dpg.node_attribute(label='Settings', attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(width=150)
            dpg.add_input_int(label='Window', default_value=20, min_value=5, max_value=100, width=100, tag=f'{node_id}_window')
            dpg.add_input_float(label='Std Dev', default_value=2.0, min_value=1.0, max_value=5.0, width=100, tag=f'{node_id}_std_dev')
            dpg.add_button(label='Calculate', width=100, callback=lambda: execute_bollinger_bands_node(node_id, get_input_data(node_id), dpg.get_value(f'{node_id}_window'), dpg.get_value(f'{node_id}_std_dev')))
            dpg.add_text('Status: Waiting', tag=f'{node_id}_status')
            dpg.add_button(label='Delete', width=100, callback=lambda: delete_node(dpg_node_id))
        output_attr = dpg.add_node_attribute(label='BB Values', attribute_type=dpg.mvNode_Attr_Output)
        dpg.add_spacer(width=1, parent=output_attr)
    node_registry[dpg_node_id] = {'node_id': node_id, 'input_attr': input_attr, 'output_attr': output_attr}
    print(f'Created Bollinger Bands node: {node_id} with input attr: {input_attr}, output attr: {output_attr}')

def execute_bollinger_bands_node(node_id, input_data, window=20, std_dev=2):
    print(f'Executing Bollinger Bands node {node_id} with window {window}, std_dev {std_dev}')
    if input_data is not None and (not input_data.empty):
        if 'Close' in input_data.columns:
            close_prices = input_data['Close']
            sma = close_prices.rolling(window=window).mean()
            std = close_prices.rolling(window=window).std()
            upper_band = sma + std * std_dev
            lower_band = sma - std * std_dev
            bb_df = pd.DataFrame({'BB_Upper': upper_band, 'BB_Middle': sma, 'BB_Lower': lower_band})
            node_outputs[node_id] = bb_df
            dpg.set_value(f'{node_id}_status', f'✓ BB({window},{std_dev}) calculated')
            print(f'Bollinger Bands calculated successfully for node {node_id}')
        else:
            dpg.set_value(f'{node_id}_status', '✗ No Close price data')
    else:
        dpg.set_value(f'{node_id}_status', '✗ No input data')

def create_advanced_signal_node(sender, app_data, user_data):
    node_id = f'adv_signal_{dpg.generate_uuid()}'
    dpg_node_id = dpg.generate_uuid()
    with dpg.node(label='Advanced Signal Generator', pos=[500, 50], parent='node_editor', tag=dpg_node_id):
        indicator1_input_attr = dpg.add_node_attribute(label='Indicator 1 (Fast)', attribute_type=dpg.mvNode_Attr_Input)
        dpg.add_spacer(width=1, parent=indicator1_input_attr)
        indicator2_input_attr = dpg.add_node_attribute(label='Indicator 2 (Slow)', attribute_type=dpg.mvNode_Attr_Input)
        dpg.add_spacer(width=1, parent=indicator2_input_attr)
        with dpg.node_attribute(label='Strategy Settings', attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(width=200)
            dpg.add_combo(['Crossover', 'Greater Than', 'RSI Levels', 'Bollinger Breakout'], default_value='Crossover', label='Condition', width=130, tag=f'{node_id}_condition')
            dpg.add_combo(['Buy on Signal', 'Sell on Signal'], default_value='Buy on Signal', label='Entry Rule', width=130, tag=f'{node_id}_entry_rule')
            dpg.add_combo(['Opposite Signal', 'Risk/Reward Only'], default_value='Risk/Reward Only', label='Exit Rule', width=130, tag=f'{node_id}_exit_rule')
            dpg.add_input_float(label='Risk/Reward Ratio', default_value=2.0, min_value=0.5, max_value=10.0, width=130, tag=f'{node_id}_risk_reward', format='%.1f')
            dpg.add_input_float(label='Stop Loss %', default_value=2.0, min_value=0.5, max_value=10.0, width=130, tag=f'{node_id}_stop_loss', format='%.1f')
            dpg.add_button(label='Generate Strategy', width=150, callback=lambda: execute_advanced_signal_node(node_id, get_input_data(node_id, 'indicator1'), get_input_data(node_id, 'indicator2'), dpg.get_value(f'{node_id}_condition'), dpg.get_value(f'{node_id}_entry_rule'), dpg.get_value(f'{node_id}_exit_rule'), dpg.get_value(f'{node_id}_risk_reward'), dpg.get_value(f'{node_id}_stop_loss')))
            dpg.add_text('Status: Waiting', tag=f'{node_id}_status')
            dpg.add_button(label='Delete', width=100, callback=lambda: delete_node(dpg_node_id))
        output_attr = dpg.add_node_attribute(label='Trading Signals', attribute_type=dpg.mvNode_Attr_Output)
        dpg.add_spacer(width=1, parent=output_attr)
    node_registry[dpg_node_id] = {'node_id': node_id, 'indicator1_input_attr': indicator1_input_attr, 'indicator2_input_attr': indicator2_input_attr, 'output_attr': output_attr}
    print(f'Created Advanced Signal node: {node_id} with indicator1: {indicator1_input_attr}, indicator2: {indicator2_input_attr}, output: {output_attr}')

def create_professional_backtest_node(sender, app_data, user_data):
    node_id = f'pro_backtest_{dpg.generate_uuid()}'
    dpg_node_id = dpg.generate_uuid()
    with dpg.node(label='Professional Backtest', pos=[750, 50], parent='node_editor', tag=dpg_node_id):
        signals_input_attr = dpg.add_node_attribute(label='Trading Signals', attribute_type=dpg.mvNode_Attr_Input)
        dpg.add_spacer(width=1, parent=signals_input_attr)
        with dpg.node_attribute(label='Backtest Settings', attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(width=180)
            dpg.add_input_int(label='Initial Capital', default_value=10000, min_value=1000, max_value=1000000, width=120, tag=f'{node_id}_capital')
            dpg.add_input_float(label='Commission %', default_value=0.1, min_value=0.0, max_value=1.0, width=120, tag=f'{node_id}_commission')
            dpg.add_button(label='Run Backtest', width=140, callback=lambda: execute_professional_backtest_node(node_id, get_input_data(node_id, 'signals'), dpg.get_value(f'{node_id}_capital'), dpg.get_value(f'{node_id}_commission') / 100.0))
            dpg.add_text('Status: Waiting', tag=f'{node_id}_status')
            dpg.add_button(label='Delete', width=100, callback=lambda: delete_node(dpg_node_id))
        with dpg.node_attribute(label='Results', attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(width=200)
            dpg.add_text('Results will appear here', tag=f'{node_id}_results_text', wrap=180)
    node_registry[dpg_node_id] = {'node_id': node_id, 'signals_input_attr': signals_input_attr}
    print(f'Created Professional Backtest node: {node_id} with signals input: {signals_input_attr}')

def execute_professional_backtest_node(node_id, signals_data, initial_capital=10000, commission=0.001):
    print(f'Executing Professional Backtest node {node_id}')
    if signals_data is None or not isinstance(signals_data, pd.DataFrame):
        dpg.set_value(f'{node_id}_status', '✗ Need valid signals data')
        return
    required_cols = ['Close', 'signal']
    if not all((col in signals_data.columns for col in required_cols)):
        missing = [col for col in required_cols if col not in signals_data.columns]
        dpg.set_value(f'{node_id}_status', f'✗ Missing: {missing}')
        return
    try:
        signals = signals_data.copy()
        signal_trades = signals[signals['signal'] != 0]
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        print(f'Signal Analysis:')
        print(f'  Total signals: {len(signal_trades)}')
        print(f'  Buy signals (1): {len(buy_signals)}')
        print(f'  Sell signals (-1): {len(sell_signals)}')
        print(f'  Signal range: {signals['signal'].min()} to {signals['signal'].max()}')
        unique_signals = signals['signal'].unique()
        print(f'  Unique signal values: {unique_signals}')
        if signal_trades.empty:
            dpg.set_value(f'{node_id}_status', '✗ No trades found in signals')
            print('No trades found in signal data')
            return
        print(f'Found {len(signal_trades)} signal entries in data')
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['price'] = signals['Close']
        portfolio['signal'] = signals['signal']
        portfolio['shares'] = 0.0
        portfolio['cash'] = float(initial_capital)
        portfolio['commission'] = 0.0
        portfolio['trade_type'] = ''
        portfolio['trade_value'] = 0.0
        portfolio['realized_pnl'] = 0.0
        shares_held = 0.0
        cash = float(initial_capital)
        total_commission = 0.0
        entry_price = 0.0
        trades = []
        total_trades = 0
        print(f'Starting backtest with ${initial_capital:,.2f} initial capital')
        for i, (date, row) in enumerate(portfolio.iterrows()):
            current_price = row['price']
            signal = row['signal']
            if i == 0:
                portfolio.loc[date, 'shares'] = shares_held
                portfolio.loc[date, 'cash'] = cash
                portfolio.loc[date, 'trade_type'] = 'START'
                continue
            commission_cost = 0.0
            trade_value = 0.0
            realized_pnl = 0.0
            trade_type = 'HOLD'
            if signal != 0:
                print(f'Processing signal {signal} on {date.date()} at price ${current_price:.2f}')
            if signal == 1 or signal == 1.0:
                print(f'  -> BUY signal detected')
                if shares_held == 0:
                    available_cash = cash * 0.95
                    effective_price = current_price * (1 + commission)
                    shares_to_buy = available_cash / effective_price
                    trade_value = shares_to_buy * current_price
                    commission_cost = trade_value * commission
                    total_cost = trade_value + commission_cost
                    print(f'  Available cash: ${available_cash:.2f}')
                    print(f'  Trade value: ${trade_value:.2f}')
                    print(f'  Commission: ${commission_cost:.2f}')
                    print(f'  Total cost: ${total_cost:.2f}')
                    if available_cash >= total_cost:
                        shares_held = shares_to_buy
                        cash -= total_cost
                        entry_price = current_price
                        trade_type = 'BUY'
                        total_commission += commission_cost
                        total_trades += 1
                        trades.append({'date': date, 'type': 'BUY', 'shares': shares_to_buy, 'price': current_price, 'value': trade_value, 'commission': commission_cost})
                        print(f'  EXECUTED BUY #{total_trades}: {shares_to_buy:.2f} shares at ${current_price:.2f} = ${trade_value:.2f}')
                        print(f'  Cash remaining: ${cash:.2f}')
                    else:
                        print(f'  INSUFFICIENT CASH: Need ${total_cost:.2f}, have ${available_cash:.2f}')
                else:
                    print(f'  ALREADY HOLDING: {shares_held:.2f} shares')
            elif signal == -1 or signal == -1.0:
                print(f'  -> SELL signal detected')
                if shares_held > 0:
                    trade_value = shares_held * current_price
                    commission_cost = trade_value * commission
                    proceeds = trade_value - commission_cost
                    cost_basis = shares_held * entry_price
                    realized_pnl = trade_value - cost_basis - commission_cost
                    cash += proceeds
                    trade_type = 'SELL'
                    total_commission += commission_cost
                    trades.append({'date': date, 'type': 'SELL', 'shares': shares_held, 'price': current_price, 'value': trade_value, 'commission': commission_cost, 'pnl': realized_pnl, 'entry_price': entry_price})
                    print(f'  EXECUTED SELL: {shares_held:.2f} shares at ${current_price:.2f} = ${trade_value:.2f}, P&L: ${realized_pnl:.2f}')
                    shares_held = 0.0
                    entry_price = 0.0
                else:
                    print(f'  NO SHARES TO SELL')
            portfolio.loc[date, 'shares'] = shares_held
            portfolio.loc[date, 'cash'] = cash
            portfolio.loc[date, 'commission'] = commission_cost
            portfolio.loc[date, 'trade_type'] = trade_type
            portfolio.loc[date, 'trade_value'] = trade_value
            portfolio.loc[date, 'realized_pnl'] = realized_pnl
        if shares_held > 0:
            final_date = portfolio.index[-1]
            final_price = portfolio.loc[final_date, 'price']
            final_value = shares_held * final_price
            final_commission = final_value * commission
            final_proceeds = final_value - final_commission
            cost_basis = shares_held * entry_price
            final_pnl = final_value - cost_basis - final_commission
            cash += final_proceeds
            total_commission += final_commission
            trades.append({'date': final_date, 'type': 'FINAL_SELL', 'shares': shares_held, 'price': final_price, 'value': final_value, 'commission': final_commission, 'pnl': final_pnl, 'entry_price': entry_price})
            portfolio.loc[final_date, 'shares'] = 0.0
            portfolio.loc[final_date, 'cash'] = cash
            portfolio.loc[final_date, 'commission'] = final_commission
            portfolio.loc[final_date, 'trade_type'] = 'FINAL_SELL'
            portfolio.loc[final_date, 'trade_value'] = final_value
            portfolio.loc[final_date, 'realized_pnl'] = final_pnl
            print(f'FINAL SELL: {shares_held:.2f} shares at ${final_price:.2f} = ${final_value:.2f}, P&L: ${final_pnl:.2f}')
        portfolio['stock_value'] = portfolio['shares'] * portfolio['price']
        portfolio['total_value'] = portfolio['cash'] + portfolio['stock_value']
        portfolio['returns'] = portfolio['total_value'].pct_change()
        portfolio['cumulative_returns'] = (portfolio['total_value'] / initial_capital - 1) * 100
        final_value = portfolio['total_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        buy_trades = [t for t in trades if t['type'] in ['BUY']]
        sell_trades = [t for t in trades if t['type'] in ['SELL', 'FINAL_SELL']]
        actual_trade_count = len(buy_trades)
        winning_trades = len([t for t in sell_trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in sell_trades if t.get('pnl', 0) < 0])
        win_rate = winning_trades / len(sell_trades) * 100 if sell_trades else 0
        returns_series = portfolio['returns'].dropna()
        if len(returns_series) > 0:
            avg_return = returns_series.mean()
            volatility = returns_series.std()
            sharpe_ratio = avg_return / volatility * np.sqrt(252) if volatility != 0 else 0
            running_max = portfolio['total_value'].expanding().max()
            drawdown = (portfolio['total_value'] - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            profitable_trades = [t['pnl'] for t in sell_trades if t.get('pnl', 0) > 0]
            losing_trades_pnl = [t['pnl'] for t in sell_trades if t.get('pnl', 0) < 0]
            avg_win = np.mean(profitable_trades) if profitable_trades else 0
            avg_loss = np.mean(losing_trades_pnl) if losing_trades_pnl else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        node_outputs[node_id] = portfolio
        dpg.set_value(f'{node_id}_status', f'✓ Return: {total_return:.2f}%')
        results_text = f'PERFORMANCE METRICS:\nTotal Return: {total_return:.2f}%\nFinal Value: ${final_value:,.2f}\nMax Drawdown: {max_drawdown:.2f}%\nSharpe Ratio: {sharpe_ratio:.2f}\n\nTRADING STATISTICS:\nTotal Trades: {actual_trade_count}\nWinning Trades: {winning_trades}\nLosing Trades: {losing_trades}\nWin Rate: {win_rate:.1f}%\nAvg Win: ${avg_win:.2f}\nAvg Loss: ${avg_loss:.2f}\nProfit Factor: {profit_factor:.2f}\n\nCOSTS:\nTotal Commission: ${total_commission:.2f}\nCommission %: {total_commission / initial_capital * 100:.3f}%\n\nSTRATEGY: {(signals_data.get('strategy', ['N/A']).iloc[0] if 'strategy' in signals_data.columns else 'N/A')}'
        dpg.set_value(f'{node_id}_results_text', results_text)
        print(f'Professional backtest completed for node {node_id}: {total_return:.2f}% return, {actual_trade_count} trades')
        print(f'Trade Summary: {winning_trades} wins, {losing_trades} losses, {win_rate:.1f}% win rate')
    except Exception as e:
        dpg.set_value(f'{node_id}_status', f'✗ Error: {str(e)[:30]}')
        print(f'Error in professional backtest for node {node_id}: {e}')
        import traceback
        traceback.print_exc()

def create_comprehensive_plot_node(sender, app_data, user_data):
    node_id = f'plot_{dpg.generate_uuid()}'
    dpg_node_id = dpg.generate_uuid()
    with dpg.node(label='Comprehensive Plot', pos=[500, 350], parent='node_editor', tag=dpg_node_id):
        input_attr = dpg.add_node_attribute(label='Data Input', attribute_type=dpg.mvNode_Attr_Input)
        dpg.add_spacer(width=1, parent=input_attr)
        with dpg.node_attribute(label='Plot', attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(width=150)
            dpg.add_button(label='Show Plot', width=100, callback=lambda: show_comprehensive_plot(node_id))
            dpg.add_text('Status: Waiting', tag=f'{node_id}_status')
            dpg.add_button(label='Delete', width=100, callback=lambda: delete_node(dpg_node_id))
    node_registry[dpg_node_id] = {'node_id': node_id, 'input_attr': input_attr}
    print(f'Created Comprehensive Plot node: {node_id} with input attr: {input_attr}')

