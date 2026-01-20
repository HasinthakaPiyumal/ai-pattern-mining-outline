# Cluster 88

class LogicalVectorIndexScanToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICAL_VECTOR_INDEX_SCAN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_VECTOR_INDEX_SCAN_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_VECTOR_INDEX_SCAN_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalVectorIndexScan, context: OptimizerContext):
        after = VectorIndexScanPlan(before.index, before.limit_count, before.search_query_expr)
        for child in before.children:
            after.append_child(child)
        yield after

