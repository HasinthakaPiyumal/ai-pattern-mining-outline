# Cluster 99

class ArithmeticExpression(AbstractExpression):

    def __init__(self, exp_type: ExpressionType, left: AbstractExpression, right: AbstractExpression):
        children = []
        if left is not None:
            children.append(left)
        if right is not None:
            children.append(right)
        super().__init__(exp_type, rtype=ExpressionReturnType.FLOAT, children=children)

    def evaluate(self, *args, **kwargs):
        vl = self.get_child(0).evaluate(*args, **kwargs)
        vr = self.get_child(1).evaluate(*args, **kwargs)
        return Batch.combine_batches(vl, vr, self.etype)

    def __eq__(self, other):
        is_subtree_equal = super().__eq__(other)
        if not isinstance(other, ArithmeticExpression):
            return False
        return is_subtree_equal and self.etype == other.etype

