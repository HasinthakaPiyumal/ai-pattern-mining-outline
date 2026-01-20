# Cluster 75

def merge_dict_of_dicts(dict1, dict2):
    """In case of conflict override with dict2"""
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict.keys():
            if value is not None:
                if isinstance(merged_dict[key], dict) and isinstance(value, dict):
                    merged_dict[key] = merge_dict_of_dicts(merged_dict[key], value)
                else:
                    merged_dict[key] = value
        else:
            merged_dict[key] = value
    return merged_dict

