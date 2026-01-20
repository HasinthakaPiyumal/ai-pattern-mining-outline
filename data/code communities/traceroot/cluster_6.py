# Cluster 6

def matches_filters(log: LogNode) -> bool:
    """Check if a log matches all filter criteria."""
    for feature_type, feature_value, feature_op in zip(feature_types, feature_values, feature_ops):
        log_value = get_log_feature_value(log, feature_type)
        if not apply_operation(str(log_value), str(feature_value), feature_op):
            return False
    return True

def get_log_feature_value(log: LogNode, feature: LogFeature, is_github_pr: bool=False) -> str | int | datetime:
    """Get the feature value from a LogNode."""
    feature_mapping = {LogFeature.LOG_UTC_TIMESTAMP: log.log_utc_timestamp, LogFeature.LOG_LEVEL: log.log_level, LogFeature.LOG_FILE_NAME: log.log_file_name, LogFeature.LOG_FUNC_NAME: log.log_func_name, LogFeature.LOG_MESSAGE_VALUE: log.log_message, LogFeature.LOG_LINE_NUMBER: str(log.log_line_number), LogFeature.LOG_SOURCE_CODE_LINE: log.log_source_code_line}
    if is_github_pr:
        feature_mapping[LogFeature.LOG_SOURCE_CODE_LINES_ABOVE] = '\n'.join(log.log_source_code_lines_above)
        feature_mapping[LogFeature.LOG_SOURCE_CODE_LINES_BELOW] = '\n'.join(log.log_source_code_lines_below)
    return feature_mapping.get(feature, '')

def apply_operation(log_value: str, filter_value: str, operation: FeatureOps) -> bool:
    """Apply the filtering operation between log value
    and filter value.
    """
    log_value_lower = log_value.lower()
    filter_value_lower = filter_value.lower()
    if operation == FeatureOps.EQUAL:
        return log_value_lower == filter_value_lower
    elif operation == FeatureOps.NOT_EQUAL:
        return log_value_lower != filter_value_lower
    elif operation == FeatureOps.CONTAINS:
        return filter_value_lower in log_value_lower
    elif operation == FeatureOps.NOT_CONTAINS:
        return filter_value_lower not in log_value_lower
    else:
        return False

