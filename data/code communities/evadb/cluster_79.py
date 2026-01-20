# Cluster 79

class LogicalDeleteToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALDELETE)
        super().__init__(RuleType.LOGICAL_DELETE_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_DELETE_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalDelete, context: OptimizerContext):
        after = DeletePlan(before.table_ref, before.where_clause)
        yield after

