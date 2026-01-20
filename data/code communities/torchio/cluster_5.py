# Cluster 5

def guess_type(string: str) -> Any:
    string = string.replace(' ', '')
    result_type: Any
    try:
        value = ast.literal_eval(string)
    except ValueError:
        result_type = str
    else:
        result_type = type(value)
    if result_type in (list, tuple):
        string = string[1:-1]
        split = string.split(',')
        list_result = [guess_type(n) for n in split]
        value = tuple(list_result) if result_type is tuple else list_result
        return value
    try:
        value = result_type(string)
    except TypeError:
        value = None
    return value

def get_params_dict_from_kwargs(kwargs):
    from torchio.utils import guess_type
    params_dict = {}
    if kwargs is not None:
        for substring in kwargs.split():
            try:
                key, value_string = substring.split('=')
            except ValueError as error:
                message = f'Arguments string "{kwargs}" not valid'
                raise ValueError(message) from error
            value = guess_type(value_string)
            params_dict[key] = value
    return params_dict

