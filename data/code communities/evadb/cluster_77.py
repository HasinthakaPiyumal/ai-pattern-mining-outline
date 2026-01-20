# Cluster 77

class CacheFunctionExpressionInProject(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALPROJECT)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.CACHE_FUNCTION_EXPRESISON_IN_PROJECT, pattern)

    def promise(self):
        return Promise.CACHE_FUNCTION_EXPRESISON_IN_PROJECT

    def check(self, before: LogicalProject, context: OptimizerContext):
        valid_exprs = []
        for expr in before.target_list:
            if isinstance(expr, FunctionExpression):
                func_exprs = list(expr.find_all(FunctionExpression))
                valid_exprs.extend(filter(lambda expr: check_expr_validity_for_cache(expr), func_exprs))
        if len(valid_exprs) > 0:
            return True
        return False

    def apply(self, before: LogicalProject, context: OptimizerContext):
        new_target_list = [expr.copy() for expr in before.target_list]
        for expr in new_target_list:
            enable_cache_on_expression_tree(context, expr)
        after = LogicalProject(target_list=new_target_list, children=before.children)
        yield after

def check_expr_validity_for_cache(expr: FunctionExpression):
    valid = expr.name in CACHEABLE_FUNCTIONS and (not expr.has_cache())
    if len(expr.children) == 1:
        valid &= isinstance(expr.children[0], TupleValueExpression)
    elif len(expr.children) == 2:
        valid &= isinstance(expr.children[0], ConstantValueExpression) and isinstance(expr.children[1], TupleValueExpression)
    return valid

class CacheFunctionExpressionInFilter(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.CACHE_FUNCTION_EXPRESISON_IN_FILTER, pattern)

    def promise(self):
        return Promise.CACHE_FUNCTION_EXPRESISON_IN_FILTER

    def check(self, before: LogicalFilter, context: OptimizerContext):
        func_exprs = list(before.predicate.find_all(FunctionExpression))
        valid_exprs = list(filter(lambda expr: check_expr_validity_for_cache(expr), func_exprs))
        if len(valid_exprs) > 0:
            return True
        return False

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        after_predicate = before.predicate.copy()
        enable_cache_on_expression_tree(context, after_predicate)
        after_operator = LogicalFilter(predicate=after_predicate, children=before.children)
        yield after_operator

