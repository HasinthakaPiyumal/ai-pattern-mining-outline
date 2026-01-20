# Cluster 18

def load_more_items():
    """Load more items for infinite scroll"""
    global current_batch, loaded_items
    new_items = []
    for _ in range(batch_size):
        new_items.append(generate_random_container())
    loaded_items.extend(new_items)
    current_batch += 1
    print(f'Loaded batch {current_batch}, total items: {len(loaded_items)}')

def generate_random_container():
    """Generate a random container configuration"""
    labels = ['Stock Chart', 'Order Book', 'News Feed', 'Portfolio', 'Watchlist', 'Market Data', 'Analytics', 'Trading Panel', 'Risk Meter', 'Balance', 'Economic Calendar', 'Alerts', 'Research', 'Screener', 'Heat Map', 'Sentiment', 'Volatility', 'Options Flow', 'Crypto', 'Forex', 'Bonds', 'Commodities', 'Earnings', 'IPO Tracker', 'Dividends']
    col_span = random.choice([1, 1, 1, 2, 2, 3, 4])
    row_span = random.choice([1, 1, 1, 2, 2])
    label = random.choice(labels) + f' #{current_batch * batch_size + len(loaded_items) + 1}'
    color = [random.randint(80, 220), random.randint(80, 220), random.randint(80, 220), 255]
    return (col_span, row_span, label, color)

def check_scroll_position():
    """Check if we need to load more content based on scroll position"""
    if not dpg.does_item_exist('grid_container'):
        return
    if len(loaded_items) < 100:
        try:
            scroll_max_y = dpg.get_y_scroll_max('grid_container')
            scroll_y = dpg.get_y_scroll('grid_container')
            viewport_height = dpg.get_viewport_height()
            if scroll_max_y - scroll_y < viewport_height * 2:
                load_more_items()
                create_grid_layout()
        except:
            estimated_rows = len(loaded_items) // COLS + 1
            viewport_height = dpg.get_viewport_height()
            if estimated_rows * ROW_HEIGHT < viewport_height * 3:
                load_more_items()
                create_grid_layout()

def create_grid_layout():
    """Create a responsive grid layout with proper positioning"""
    if dpg.does_item_exist('grid_container'):
        dpg.delete_item('grid_container')
    viewport_width = dpg.get_viewport_width()
    viewport_height = dpg.get_viewport_height()
    total_margin_width = MARGIN * (COLS + 1)
    available_width = viewport_width - total_margin_width
    col_width = available_width // COLS
    with dpg.child_window(tag='grid_container', parent='main_window', width=viewport_width, height=viewport_height, pos=[0, 0], horizontal_scrollbar=False, no_scrollbar=False, border=False):
        occupied_grid = {}
        max_row_used = 0
        for i, (col_span, row_span, label, color) in enumerate(loaded_items):
            col_span = min(col_span, COLS)
            row, col = find_next_available_position(occupied_grid, col_span, row_span)
            mark_cells_occupied(occupied_grid, row, col, row_span, col_span)
            max_row_used = max(max_row_used, row + row_span - 1)
            x_pos = MARGIN + col * (col_width + MARGIN)
            y_pos = MARGIN + row * (ROW_HEIGHT + MARGIN)
            box_width = col_span * col_width + (col_span - 1) * MARGIN
            box_height = row_span * ROW_HEIGHT + (row_span - 1) * MARGIN
            with dpg.theme() as box_theme:
                with dpg.theme_component(dpg.mvChildWindow):
                    dpg.add_theme_color(dpg.mvThemeCol_ChildBg, color)
                    dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 3)
                    dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
                    dpg.add_theme_color(dpg.mvThemeCol_Border, [255, 255, 255, 100])
            with dpg.child_window(width=box_width, height=box_height, pos=[x_pos, y_pos], border=True, tag=f'container_{i}') as box:
                dpg.bind_item_theme(box, box_theme)
                dpg.add_text(label, pos=[8, 8])
                dpg.add_text(f'{col_span}x{row_span} | Item {i + 1}', pos=[8, 28])
                if box_height > 60:
                    content_y = 48
                    content_width = box_width - 16
                    if content_width > 100:
                        if 'Chart' in label:
                            value = i * 0.1 % 1.0
                            dpg.add_progress_bar(default_value=value, pos=[8, content_y], width=min(content_width, 200))
                        elif 'Order' in label or 'Trading' in label:
                            dpg.add_input_float(label='', default_value=100.0 + i, pos=[8, content_y], width=min(content_width, 120))
                        elif 'News' in label or 'Feed' in label:
                            if box_height > 100:
                                items = [f'Item {j + 1}' for j in range(3)]
                                list_height = min(3, (box_height - content_y - 16) // 20)
                                dpg.add_listbox(items, pos=[8, content_y], width=min(content_width, 150), num_items=max(1, list_height))
                        else:
                            dpg.add_button(label=f'Action', pos=[8, content_y], width=min(content_width, 100))
        if loaded_items:
            load_more_y = MARGIN + (max_row_used + 2) * (ROW_HEIGHT + MARGIN)
            with dpg.child_window(width=viewport_width - 2 * MARGIN, height=60, pos=[MARGIN, load_more_y], border=True) as load_more_container:
                with dpg.theme() as load_more_theme:
                    with dpg.theme_component(dpg.mvChildWindow):
                        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [60, 60, 60, 255])
                        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 3)
                dpg.bind_item_theme(load_more_container, load_more_theme)
                dpg.add_text(f'Loaded {len(loaded_items)} items - Scroll down or click to load more', pos=[10, 10])
                dpg.add_button(label='Load More Now', pos=[10, 30], callback=lambda: [load_more_items(), create_grid_layout()])

def find_next_available_position(occupied_grid, col_span, row_span, start_row=0):
    """Find the next available position in the grid that can fit the item"""
    row = start_row
    while True:
        for col in range(COLS - col_span + 1):
            can_place = True
            for r in range(row, row + row_span):
                for c in range(col, col + col_span):
                    if occupied_grid.get((r, c), False):
                        can_place = False
                        break
                if not can_place:
                    break
            if can_place:
                return (row, col)
        row += 1

def mark_cells_occupied(occupied_grid, row, col, row_span, col_span):
    """Mark cells as occupied in the grid"""
    for r in range(row, row + row_span):
        for c in range(col, col + col_span):
            occupied_grid[r, c] = True

def resize_callback():
    """Recreate grid and resize main window when viewport is resized"""
    viewport_width = dpg.get_viewport_width()
    viewport_height = dpg.get_viewport_height()
    dpg.set_item_width('main_window', viewport_width)
    dpg.set_item_height('main_window', viewport_height)
    dpg.set_item_pos('main_window', [0, 0])
    create_grid_layout()

