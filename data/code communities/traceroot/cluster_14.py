# Cluster 14

def separate_filter_categories(categories: list[str], values: list[str], operations: list[str]) -> FilterCategories:
    """Separate special filter categories from regular ones.

    Processes the parallel arrays of categories, values, and operations,
    separating out special categories (service_name, service_environment, log)
    that need different handling. This enables:
    - AWS X-Ray optimized queries for service filters
    - CloudWatch Logs search for log content filters
    - Standard trace filtering for everything else

    Args:
        categories: List of category names (e.g., ['service_name', 'status'])
        values: List of filter values (e.g., ['api-service', '500'])
        operations: List of operation strings (e.g., ['=', '!='])

    Returns:
        FilterCategories with separated special and remaining categories

    Note:
        Categories without corresponding values/operations are added to
        remaining_categories with empty values. Operations are converted
        to Operation enum types.

    Example:
        >>> filter_cats = separate_filter_categories(
        ...     categories=['service_name', 'status', 'log'],
        ...     values=['api-service', '500', 'error'],
        ...     operations=['=', '=', 'contains']
        ... )
        >>> filter_cats.service_name_values
        ['api-service']
        >>> filter_cats.log_search_values
        ['error']
        >>> filter_cats.remaining_categories
        ['status']
        >>> filter_cats.remaining_values
        ['500']

    Raises:
        ValueError: If operation string is not a valid Operation enum value
    """
    service_name_values = []
    service_name_operations = []
    service_environment_values = []
    service_environment_operations = []
    log_search_values = []
    log_search_operations = []
    remaining_categories = []
    remaining_values = []
    remaining_operations = []
    for i, category in enumerate(categories):
        if i >= len(values) or i >= len(operations):
            remaining_categories.append(category)
            continue
        value = values[i]
        operation = operations[i]
        if category == 'service_name':
            service_name_values.append(value)
            service_name_operations.append(Operation(operation))
        elif category == 'service_environment':
            service_environment_values.append(value)
            service_environment_operations.append(Operation(operation))
        elif category == 'log':
            log_search_values.append(value)
            log_search_operations.append(Operation(operation))
        else:
            remaining_categories.append(category)
            remaining_values.append(value)
            remaining_operations.append(Operation(operation))
    return FilterCategories(service_name_values=service_name_values, service_name_operations=service_name_operations, service_environment_values=service_environment_values, service_environment_operations=service_environment_operations, log_search_values=log_search_values, log_search_operations=log_search_operations, remaining_categories=remaining_categories, remaining_values=remaining_values, remaining_operations=remaining_operations)

def hash_user_sub(user_sub: str) -> str:
    hash_object = hashlib.sha256(user_sub.encode('utf-8'))
    hashed = hash_object.hexdigest()
    return hashed

