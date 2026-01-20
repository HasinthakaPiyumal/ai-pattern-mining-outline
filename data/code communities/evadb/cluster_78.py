# Cluster 78

class LogicalCreateFromSelectToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALCREATE)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_CREATE_FROM_SELECT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_CREATE_FROM_SELECT_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalCreate, context: OptimizerContext):
        after = CreateFromSelectPlan(before.video, before.column_list, before.if_not_exists)
        for child in before.children:
            after.append_child(child)
        yield after

