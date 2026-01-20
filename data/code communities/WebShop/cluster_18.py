# Cluster 18

def bert_predict(obs, info, softmax=True):
    valid_acts = info['valid']
    assert valid_acts[0].startswith('click[')
    state_encodings = bert_tokenizer(process_str(obs), max_length=512, truncation=True, padding='max_length')
    action_encodings = bert_tokenizer(list(map(process_str, valid_acts)), max_length=512, truncation=True, padding='max_length')
    batch = {'state_input_ids': state_encodings['input_ids'], 'state_attention_mask': state_encodings['attention_mask'], 'action_input_ids': action_encodings['input_ids'], 'action_attention_mask': action_encodings['attention_mask'], 'sizes': len(valid_acts), 'images': info['image_feat'].tolist(), 'labels': 0}
    batch = data_collator([batch])
    outputs = bert_model(**batch)
    if softmax:
        idx = torch.multinomial(torch.nn.functional.softmax(outputs.logits[0], dim=0), 1)[0].item()
    else:
        idx = outputs.logits[0].argmax(0).item()
    return valid_acts[idx]

def process_str(s):
    s = s.lower().replace('"', '').replace("'", '').strip()
    s = s.replace('[sep]', '[SEP]')
    return s

def data_collator(batch):
    state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, labels, images = ([], [], [], [], [], [], [])
    for sample in batch:
        state_input_ids.append(sample['state_input_ids'])
        state_attention_mask.append(sample['state_attention_mask'])
        action_input_ids.extend(sample['action_input_ids'])
        action_attention_mask.extend(sample['action_attention_mask'])
        sizes.append(sample['sizes'])
        labels.append(sample['labels'])
        images.append(sample['images'])
    max_state_len = max((sum(x) for x in state_attention_mask))
    max_action_len = max((sum(x) for x in action_attention_mask))
    return {'state_input_ids': torch.tensor(state_input_ids)[:, :max_state_len], 'state_attention_mask': torch.tensor(state_attention_mask)[:, :max_state_len], 'action_input_ids': torch.tensor(action_input_ids)[:, :max_action_len], 'action_attention_mask': torch.tensor(action_attention_mask)[:, :max_action_len], 'sizes': torch.tensor(sizes), 'images': torch.tensor(images), 'labels': torch.tensor(labels)}

def predict(obs, info):
    """
    Given WebShop environment observation and info, predict an action.
    """
    valid_acts = info['valid']
    if valid_acts[0].startswith('click['):
        return bert_predict(obs, info)
    else:
        return 'search[' + bart_predict(process_goal(obs)) + ']'

def bart_predict(input):
    input_ids = bart_tokenizer(input)['input_ids']
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    output = bart_model.generate(input_ids, max_length=512, num_return_sequences=5, num_beams=5)
    return bart_tokenizer.batch_decode(output.tolist(), skip_special_tokens=True)[0]

def process_goal(state):
    state = state.lower().replace('"', '').replace("'", '')
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    state = state.replace('\n[button] search [button_]', '').strip()
    if ', and price lower than' in state:
        state = state.split(', and price lower than')[0]
    return state

def run_episode(goal, env, verbose=True):
    """
    Interact with amazon to find a product given input goal.
    Input: text goal
    Output: a url of found item on amazon.
    """
    env = env.lower()
    if env not in ENVIRONMENTS:
        print(f'[ERROR] Environment {env} not recognized')
    obs = 'Amazon Shopping Game\nInstruction:' + goal + '\n[button] search [button]'
    info = {'valid': ['search[stuff]'], 'image_feat': torch.zeros(512)}
    product_map = {}
    title_to_asin_map = {}
    search_results_cache = {}
    visited_asins, clicked_options = (set(), set())
    sub_page_type, page_type, page_num = (None, None, None)
    search_terms, prod_title, asin = (None, None, None)
    options = {}
    for i in range(100):
        action = predict(obs, info)
        if verbose:
            print('====')
            print(action)
        action_content = action[action.find('[') + 1:action.find(']')]
        prev_page_type = page_type
        if action.startswith('search['):
            page_type = Page.RESULTS
            search_terms = action_content
            page_num = 1
        elif action.startswith('click['):
            if action.startswith('click[item -'):
                prod_title = action_content[len('item -'):].strip()
                found = False
                for key in title_to_asin_map:
                    if prod_title == key:
                        asin = title_to_asin_map[key]
                        page_type = Page.ITEM_PAGE
                        visited_asins.add(asin)
                        found = True
                        break
                if not found:
                    raise Exception('Product to click not found')
            elif any((x.value in action for x in [Page.DESC, Page.FEATURES, Page.REVIEWS])):
                page_type = Page.SUB_PAGE
                sub_page_type = Page(action_content.lower())
            elif action == 'click[< prev]':
                if sub_page_type is not None:
                    page_type, sub_page_type = (Page.ITEM_PAGE, None)
                elif prev_page_type == Page.ITEM_PAGE:
                    page_type = Page.RESULTS
                    options, clicked_options = ({}, set())
                elif prev_page_type == Page.RESULTS and page_num > 1:
                    page_type = Page.RESULTS
                    page_num -= 1
            elif action == 'click[next >]':
                page_type = Page.RESULTS
                page_num += 1
            elif action.lower() == 'click[back to search]':
                page_type = Page.SEARCH
            elif action == 'click[buy now]':
                return get_return_value(env, asin, options, search_terms, page_num, product_map[asin])
            elif prev_page_type == Page.ITEM_PAGE:
                found = False
                for opt_name, opt_values in product_map[asin]['options'].items():
                    if action_content in opt_values:
                        options[opt_name] = action_content
                        page_type = Page.ITEM_PAGE
                        clicked_options.add(action_content)
                        found = True
                        break
                if not found:
                    raise Exception('Unrecognized action: ' + action)
        else:
            raise Exception('Unrecognized action:' + action)
        if verbose:
            print(f'Parsing {page_type.value} page...')
        if page_type == Page.RESULTS:
            if search_terms in search_results_cache:
                data = search_results_cache[search_terms]
                if verbose:
                    print(f'Loading cached results page for "{search_terms}"')
            else:
                begin = time.time()
                if env == 'amazon':
                    data = parse_results_amz(search_terms, page_num, verbose)
                if env == 'webshop':
                    data = parse_results_ws(search_terms, page_num, verbose)
                if env == 'ebay':
                    data = parse_results_ebay(search_terms, page_num, verbose)
                end = time.time()
                if verbose:
                    print(f'Parsing search results took {end - begin} seconds')
                search_results_cache[search_terms] = data
                for d in data:
                    title_to_asin_map[d['Title']] = d['asin']
        elif page_type == Page.ITEM_PAGE or page_type == Page.SUB_PAGE:
            if asin in product_map:
                if verbose:
                    print('Loading cached item page for', asin)
                data = product_map[asin]
            else:
                begin = time.time()
                if env == 'amazon':
                    data = parse_item_page_amz(asin, verbose)
                if env == 'webshop':
                    data = parse_item_page_ws(asin, search_terms, page_num, options, verbose)
                if env == 'ebay':
                    data = parse_item_page_ebay(asin, verbose)
                end = time.time()
                if verbose:
                    print('Parsing item page took', end - begin, 'seconds')
                product_map[asin] = data
        elif page_type == Page.SEARCH:
            if verbose:
                print('Executing search')
            obs = 'Amazon Shopping Game\nInstruction:' + goal + '\n[button] search [button]'
            info = {'valid': ['search[stuff]'], 'image_feat': torch.zeros(512)}
            continue
        else:
            raise Exception('Page of type `', page_type, '` not found')
        begin = time.time()
        html_str = dict_to_fake_html(data, page_type, asin, sub_page_type, options, product_map, goal)
        obs = convert_html_to_text(html_str, simple=False, clicked_options=clicked_options, visited_asins=visited_asins)
        end = time.time()
        if verbose:
            print('[Page Info -> WebShop HTML -> Observation] took', end - begin, 'seconds')
        begin = time.time()
        prod_arg = product_map if page_type == Page.ITEM_PAGE else data
        info = convert_dict_to_actions(page_type, prod_arg, asin, page_num)
        end = time.time()
        if verbose:
            print('Extracting available actions took', end - begin, 'seconds')
        if i == 50:
            return get_return_value(env, asin, options, search_terms, page_num, product_map[asin])

def get_return_value(env, asin, options, search_terms, page_num, product):
    asin_url = None
    if env == 'webshop':
        query_str = '+'.join(search_terms.split())
        options_str = json.dumps(options)
        asin_url = f'{WEBSHOP_URL}/item_page/{WEBSHOP_SESSION}/{asin}/{query_str}/{page_num}/{options_str}'
    else:
        asin_url = f'https://www.ebay.com/itm/{asin}' if env == 'ebay' else f'https://www.amazon.com/dp/{asin}'
    product_reduced = {k: v for k, v in product.items() if k in ['asin', 'Title', 'Description', 'BulletPoints']}
    product_reduced['Description'] = product_reduced['Description'][:100] + '...'
    product_reduced['Features'] = product_reduced.pop('BulletPoints')
    product_reduced['Features'] = product_reduced['Features'][:100] + '...'
    html = '<!DOCTYPE html><html><head><title>Chosen Product</title></head><body>'
    html += f'Product Image:<img src="{product['MainImage']}" height="50px" /><br>' if len(product['MainImage']) > 0 else ''
    html += f'Link to Product:\n        <a href="{asin_url}" style="color:blue;text-decoration:underline;" target="_blank">{asin_url}</a>\n        </body></html>'
    return (product_reduced, options if len(options) > 0 else 'None Selected', html)

def predict(obs, info, model, softmax=False, rule=False, bart_model=None):
    valid_acts = info['valid']
    if valid_acts[0].startswith('search['):
        if bart_model is None:
            return valid_acts[-1]
        else:
            goal = process_goal(obs)
            query = bart_predict(goal, bart_model, num_return_sequences=5, num_beams=5)
            query = query[0]
            return f'search[{query}]'
    if rule:
        item_acts = [act for act in valid_acts if act.startswith('click[item - ')]
        if item_acts:
            return item_acts[0]
        else:
            assert 'click[buy now]' in valid_acts
            return 'click[buy now]'
    state_encodings = tokenizer(process(obs), max_length=512, truncation=True, padding='max_length')
    action_encodings = tokenizer(list(map(process, valid_acts)), max_length=512, truncation=True, padding='max_length')
    batch = {'state_input_ids': state_encodings['input_ids'], 'state_attention_mask': state_encodings['attention_mask'], 'action_input_ids': action_encodings['input_ids'], 'action_attention_mask': action_encodings['attention_mask'], 'sizes': len(valid_acts), 'images': info['image_feat'].tolist(), 'labels': 0}
    batch = data_collator([batch])
    batch = {k: v.cuda() for k, v in batch.items()}
    outputs = model(**batch)
    if softmax:
        idx = torch.multinomial(F.softmax(outputs.logits[0], dim=0), 1)[0].item()
    else:
        idx = outputs.logits[0].argmax(0).item()
    return valid_acts[idx]

def episode(model, idx=None, verbose=False, softmax=False, rule=False, bart_model=None):
    obs, info = env.reset(idx)
    if verbose:
        print(info['goal'])
    for i in range(100):
        action = predict(obs, info, model, softmax=softmax, rule=rule, bart_model=bart_model)
        if verbose:
            print(action)
        obs, reward, done, info = env.step(action)
        if done:
            return reward
    return 0

def get_data(split):
    data = json.load(open(PATH))
    goals, searches = ([], [])
    for goal, search_list in data.items():
        goal = process_goal(goal)
        for search in search_list:
            search = process_str(search)
            goals.append(goal)
            searches.append(search)
    n = len(goals)
    human_goals = json.load(open(HUMAN_GOAL_PATH, 'r'))
    goal_range = range(len(human_goals))
    if split == 'train':
        goal_range = range(500, len(human_goals))
    elif split == 'validation':
        goal_range = range(500, 1500)
    elif split == 'test':
        goal_range = range(0, 500)
    elif split == 'all':
        all_data = json.load(open(GOAL_PATH))
        all_goals = []
        all_goals_processed = []
        for ins_list in all_data.values():
            for ins in ins_list:
                ins = ins['instruction']
                all_goals.append(ins)
                all_goals_processed.append(process_str(ins))
        return (all_goals_processed, all_goals)
    goals_, searches_ = ([], [])
    for goal, search in zip(goals, searches):
        if goal in human_goals and human_goals.index(goal) in goal_range:
            goals_.append(goal)
            searches_.append(search)
    return (goals_, searches_)

