# Cluster 63

class AbstractExpressionsTest(unittest.TestCase):

    def test_walk(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(1)
        const_exp3 = ConstantValueExpression(0)
        const_exp4 = ConstantValueExpression(5)
        cmpr_exp1 = ComparisonExpression(ExpressionType.COMPARE_GEQ, const_exp1, const_exp2)
        cmpr_exp2 = ComparisonExpression(ExpressionType.COMPARE_GEQ, const_exp3, const_exp4)
        expr = LogicalExpression(ExpressionType.LOGICAL_AND, cmpr_exp1, cmpr_exp2)
        self.assertEqual(len(list(expr.walk())), 7)
        self.assertEqual(len(list(expr.walk(bfs=False))), 7)
        bfs = [expr, cmpr_exp1, cmpr_exp2, const_exp1, const_exp2, const_exp3, const_exp4]
        dfs = [expr, cmpr_exp1, const_exp1, const_exp2, cmpr_exp2, const_exp3, const_exp4]
        self.assertTrue(all((isinstance(exp, type(bfs[idx])) for idx, exp in enumerate(list(expr.walk())))))
        self.assertTrue(all((isinstance(exp, type(dfs[idx])) for idx, exp in enumerate(list(expr.walk(bfs=False))))))

    def test_find_all(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(1)
        const_exp3 = ConstantValueExpression(0)
        const_exp4 = ConstantValueExpression(5)
        cmpr_exp1 = ComparisonExpression(ExpressionType.COMPARE_GEQ, const_exp1, const_exp2)
        cmpr_exp2 = ComparisonExpression(ExpressionType.COMPARE_GEQ, const_exp3, const_exp4)
        expr = LogicalExpression(ExpressionType.LOGICAL_AND, cmpr_exp1, cmpr_exp2)
        self.assertEqual([cmpr_exp1, cmpr_exp2], [exp for exp in list(expr.find_all(ComparisonExpression))])
        self.assertNotEqual([cmpr_exp2, cmpr_exp1], [exp for exp in list(expr.find_all(ComparisonExpression))])
        self.assertNotEqual([None], [exp for exp in list(expr.find_all(TupleValueExpression))])

    def test_not_implemented_functions(self):
        with self.assertRaises(TypeError):
            x = AbstractExpression(exp_type=ExpressionType.LOGICAL_AND)
        with patch.object(AbstractExpression, '__abstractmethods__', set()):
            x = AbstractExpression(exp_type=ExpressionType.LOGICAL_AND)
            x.evaluate()

