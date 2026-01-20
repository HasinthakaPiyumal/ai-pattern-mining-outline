# Cluster 100

class ComparisonExpression(AbstractExpression):

    def __init__(self, exp_type: ExpressionType, left: AbstractExpression, right: AbstractExpression):
        children = []
        if left is not None:
            children.append(left)
        if right is not None:
            children.append(right)
        super().__init__(exp_type, rtype=ExpressionReturnType.BOOLEAN, children=children)

    def evaluate(self, *args, **kwargs):
        lbatch = self.get_child(0).evaluate(*args, **kwargs)
        rbatch = self.get_child(1).evaluate(*args, **kwargs)
        assert len(lbatch) == len(rbatch), f'Left and Right batch does not have equal elements: left: {len(lbatch)} right: {len(rbatch)}'
        assert self.etype in [ExpressionType.COMPARE_EQUAL, ExpressionType.COMPARE_GREATER, ExpressionType.COMPARE_LESSER, ExpressionType.COMPARE_GEQ, ExpressionType.COMPARE_LEQ, ExpressionType.COMPARE_NEQ, ExpressionType.COMPARE_CONTAINS, ExpressionType.COMPARE_IS_CONTAINED, ExpressionType.COMPARE_LIKE], f'Expression type not supported {self.etype}'
        if self.etype == ExpressionType.COMPARE_EQUAL:
            return Batch.from_eq(lbatch, rbatch)
        elif self.etype == ExpressionType.COMPARE_GREATER:
            return Batch.from_greater(lbatch, rbatch)
        elif self.etype == ExpressionType.COMPARE_LESSER:
            return Batch.from_lesser(lbatch, rbatch)
        elif self.etype == ExpressionType.COMPARE_GEQ:
            return Batch.from_greater_eq(lbatch, rbatch)
        elif self.etype == ExpressionType.COMPARE_LEQ:
            return Batch.from_lesser_eq(lbatch, rbatch)
        elif self.etype == ExpressionType.COMPARE_NEQ:
            return Batch.from_not_eq(lbatch, rbatch)
        elif self.etype == ExpressionType.COMPARE_CONTAINS:
            return Batch.compare_contains(lbatch, rbatch)
        elif self.etype == ExpressionType.COMPARE_IS_CONTAINED:
            return Batch.compare_is_contained(lbatch, rbatch)
        elif self.etype == ExpressionType.COMPARE_LIKE:
            return Batch.compare_like(lbatch, rbatch)

    def get_symbol(self) -> str:
        if self.etype == ExpressionType.COMPARE_EQUAL:
            return '='
        elif self.etype == ExpressionType.COMPARE_GREATER:
            return '>'
        elif self.etype == ExpressionType.COMPARE_LESSER:
            return '<'
        elif self.etype == ExpressionType.COMPARE_GEQ:
            return '>='
        elif self.etype == ExpressionType.COMPARE_LEQ:
            return '<='
        elif self.etype == ExpressionType.COMPARE_NEQ:
            return '!='
        elif self.etype == ExpressionType.COMPARE_CONTAINS:
            return '@>'
        elif self.etype == ExpressionType.COMPARE_IS_CONTAINED:
            return '<@'

    def __str__(self) -> str:
        expr_str = '('
        if self.get_child(0):
            expr_str += f'{self.get_child(0)}'
        if self.etype:
            expr_str += f' {self.get_symbol()} '
        if self.get_child(1):
            expr_str += f'{self.get_child(1)}'
        expr_str += ')'
        return expr_str

    def __eq__(self, other):
        is_subtree_equal = super().__eq__(other)
        if not isinstance(other, ComparisonExpression):
            return False
        return is_subtree_equal and self.etype == other.etype

    def __hash__(self) -> int:
        return super().__hash__()

