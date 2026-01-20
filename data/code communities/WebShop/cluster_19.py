# Cluster 19

@app.route('/', methods=['GET', 'POST'])
def search_results(data):
    path = os.path.join(TEMPLATE_DIR, 'results_page.html')
    html = render_template_string(read_html_template(path=path), session_id=SESSION_ID, products=data, keywords=KEYWORDS, page=1, total=len(data), instruction_text=QUERY)
    return html

def read_html_template(path):
    with open(path) as f:
        template = f.read()
    return template

@app.route('/', methods=['GET', 'POST'])
def item_page(session_id, asin, keywords, page, options):
    path = os.path.join(TEMPLATE_DIR, 'item_page.html')
    html = render_template_string(read_html_template(path=path), session_id=session_id, product_info=product_map[asin], keywords=keywords, page=page, asin=asin, options=options, instruction_text=QUERY)
    return html

@app.route('/', methods=['GET', 'POST'])
def item_sub_page(session_id, asin, keywords, page, sub_page, options):
    path = os.path.join(TEMPLATE_DIR, sub_page.value.lower() + '_page.html')
    html = render_template_string(read_html_template(path), session_id=session_id, product_info=product_map[asin], keywords=keywords, page=page, asin=asin, options=options, instruction_text=QUERY)
    return html

@app.route('/', methods=['GET', 'POST'])
def done(asin, options, session_id, **kwargs):
    path = os.path.join(TEMPLATE_DIR, 'done_page.html')
    html = render_template_string(read_html_template(path), session_id=session_id, reward=1, asin=asin, options=product_map[asin]['options'], reward_info=kwargs.get('reward_info'), goal_attrs=kwargs.get('goal_attrs'), purchased_attrs=kwargs.get('purchased_attrs'), goal=kwargs.get('goal'), mturk_code=kwargs.get('mturk_code'), query=kwargs.get('query'), category=kwargs.get('category'), product_category=kwargs.get('product_category'))
    return html

