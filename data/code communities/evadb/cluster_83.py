# Cluster 83

class LogicalJoinToPhysicalNestedLoopJoin(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALJOIN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_JOIN_TO_PHYSICAL_NESTED_LOOP_JOIN, pattern)

    def promise(self):
        return Promise.LOGICAL_JOIN_TO_PHYSICAL_NESTED_LOOP_JOIN

    def check(self, before: LogicalJoin, context: OptimizerContext):
        """
        We want to apply this rule to the join when FuzzDistance
        is being used, which implies that the join is a FuzzyJoin
        """
        if before.join_predicate is None:
            return False
        j_child: FunctionExpression = before.join_predicate.children[0]
        if not isinstance(j_child, FunctionExpression):
            return False
        return before.join_type == JoinType.INNER_JOIN and j_child.name.startswith('FuzzDistance')

    def apply(self, join_node: LogicalJoin, context: OptimizerContext):
        nested_loop_join_plan = NestedLoopJoinPlan(join_node.join_type, join_node.join_predicate)
        nested_loop_join_plan.append_child(join_node.lhs())
        nested_loop_join_plan.append_child(join_node.rhs())
        yield nested_loop_join_plan

