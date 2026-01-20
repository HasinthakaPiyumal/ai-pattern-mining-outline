# Cluster 45

def get_updated_static_items(pre_production_flows, post_production_flows):
    if isinstance(pre_production_flows['crafted'], dict):
        pre_production_flows['crafted'] = [item for item in pre_production_flows['crafted'].values()]
    if isinstance(post_production_flows['crafted'], dict):
        post_production_flows['crafted'] = [item for item in post_production_flows['crafted'].values()]
    new_production_flows = get_new_production_flows(pre_production_flows, post_production_flows)
    static_items = new_production_flows['harvested']
    for item in new_production_flows['crafted']:
        output = item['outputs']
        for key, value in output.items():
            if key in static_items:
                static_items[key] += value
            else:
                static_items[key] = value
    return static_items

def get_new_production_flows(pre_production_flows, post_production_flows):
    new_production_flows = {'input': {}, 'output': {}, 'crafted': [], 'harvested': {}}
    for flow_key in ['input', 'output', 'harvested']:
        for item, value in post_production_flows[flow_key].items():
            pre_item_value = pre_production_flows[flow_key][item] if item in pre_production_flows[flow_key] else 0
            diff = value - pre_item_value
            if diff > 0:
                new_production_flows[flow_key][item] = diff
    for item in post_production_flows['crafted']:
        if item in pre_production_flows['crafted']:
            pre_production_flows['crafted'].remove(item)
        else:
            new_production_flows['crafted'].append(item)
    return new_production_flows

