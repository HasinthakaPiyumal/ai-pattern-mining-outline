# Cluster 57

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

class ExpressionEvaluationTest(unittest.TestCase):

    def test_if_expr_tree_is_equal(self):
        const_exp1 = ConstantValueExpression(0)
        const_exp2 = ConstantValueExpression(0)
        columnName1 = TupleValueExpression(name='DATA')
        columnName2 = TupleValueExpression(name='DATA')
        aggr_expr1 = AggregationExpression(ExpressionType.AGGREGATION_AVG, None, columnName1)
        aggr_expr2 = AggregationExpression(ExpressionType.AGGREGATION_AVG, None, columnName2)
        cmpr_exp1 = ComparisonExpression(ExpressionType.COMPARE_NEQ, aggr_expr1, const_exp1)
        cmpr_exp2 = ComparisonExpression(ExpressionType.COMPARE_NEQ, aggr_expr2, const_exp2)
        self.assertEqual(cmpr_exp1, cmpr_exp2)

    def test_should_return_false_for_unequal_expressions(self):
        const_exp1 = ConstantValueExpression(0)
        const_exp2 = ConstantValueExpression(1)
        func_expr = FunctionExpression(lambda x: x + 1, name='test')
        cmpr_exp = ComparisonExpression(ExpressionType.COMPARE_NEQ, const_exp1, const_exp2)
        tuple_expr = TupleValueExpression(name='id')
        aggr_expr = AggregationExpression(ExpressionType.AGGREGATION_MAX, None, tuple_expr)
        logical_expr = LogicalExpression(ExpressionType.LOGICAL_OR, cmpr_exp, cmpr_exp)
        self.assertNotEqual(const_exp1, const_exp2)
        self.assertNotEqual(cmpr_exp, const_exp1)
        self.assertNotEqual(func_expr, cmpr_exp)
        self.assertNotEqual(tuple_expr, aggr_expr)
        self.assertNotEqual(aggr_expr, tuple_expr)
        self.assertNotEqual(tuple_expr, cmpr_exp)
        self.assertNotEqual(logical_expr, cmpr_exp)

class ConstantExpressionsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_constant_with_input_relationship(self):
        const_expr = ConstantValueExpression(1)
        input_size = 10
        self.assertEqual([1] * input_size, const_expr.evaluate(Batch(pd.DataFrame([0] * input_size))).frames[0].tolist())

class LogicalExpressionsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = Batch(pd.DataFrame([0]))

    def test_logical_and(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(1)
        comparison_expression_left = ComparisonExpression(ExpressionType.COMPARE_EQUAL, const_exp1, const_exp2)
        const_exp1 = ConstantValueExpression(2)
        const_exp2 = ConstantValueExpression(1)
        comparison_expression_right = ComparisonExpression(ExpressionType.COMPARE_GREATER, const_exp1, const_exp2)
        logical_expr = LogicalExpression(ExpressionType.LOGICAL_AND, comparison_expression_left, comparison_expression_right)
        self.assertEqual([True], logical_expr.evaluate(self.batch).frames[0].tolist())

    def test_logical_or(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(1)
        comparison_expression_left = ComparisonExpression(ExpressionType.COMPARE_EQUAL, const_exp1, const_exp2)
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(2)
        comparison_expression_right = ComparisonExpression(ExpressionType.COMPARE_GREATER, const_exp1, const_exp2)
        logical_expr = LogicalExpression(ExpressionType.LOGICAL_OR, comparison_expression_left, comparison_expression_right)
        self.assertEqual([True], logical_expr.evaluate(self.batch).frames[0].tolist())

    def test_logical_not(self):
        const_exp1 = ConstantValueExpression(0)
        const_exp2 = ConstantValueExpression(1)
        comparison_expression_right = ComparisonExpression(ExpressionType.COMPARE_GREATER, const_exp1, const_exp2)
        logical_expr = LogicalExpression(ExpressionType.LOGICAL_NOT, None, comparison_expression_right)
        self.assertEqual([True], logical_expr.evaluate(self.batch).frames[0].tolist())

    def test_short_circuiting_and_complete(self):
        tup_val_exp_l = TupleValueExpression(name=0)
        tup_val_exp_l.col_alias = 0
        tup_val_exp_r = TupleValueExpression(name=1)
        tup_val_exp_r.col_alias = 1
        comp_exp_l = ComparisonExpression(ExpressionType.COMPARE_EQUAL, tup_val_exp_l, tup_val_exp_r)
        comp_exp_r = Mock(spec=ComparisonExpression)
        logical_exp = LogicalExpression(ExpressionType.LOGICAL_AND, comp_exp_l, comp_exp_r)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6]}))
        self.assertEqual([False, False, False], logical_exp.evaluate(tuples).frames[0].tolist())
        comp_exp_r.evaluate.assert_not_called()

    def test_short_circuiting_or_complete(self):
        tup_val_exp_l = TupleValueExpression(name=0)
        tup_val_exp_l.col_alias = 0
        tup_val_exp_r = TupleValueExpression(name=1)
        tup_val_exp_r.col_alias = 1
        comp_exp_l = ComparisonExpression(ExpressionType.COMPARE_EQUAL, tup_val_exp_l, tup_val_exp_r)
        comp_exp_r = Mock(spec=ComparisonExpression)
        logical_exp = LogicalExpression(ExpressionType.LOGICAL_OR, comp_exp_l, comp_exp_r)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [1, 2, 3]}))
        self.assertEqual([True, True, True], logical_exp.evaluate(tuples).frames[0].tolist())
        comp_exp_r.evaluate.assert_not_called()

    def test_short_circuiting_and_partial(self):
        tup_val_exp_l = TupleValueExpression(name=0)
        tup_val_exp_l.col_alias = 0
        tup_val_exp_r = TupleValueExpression(name=1)
        tup_val_exp_r.col_alias = 1
        comp_exp_l = ComparisonExpression(ExpressionType.COMPARE_EQUAL, tup_val_exp_l, tup_val_exp_r)
        comp_exp_r = Mock(spec=ComparisonExpression)
        comp_exp_r.evaluate = Mock(return_value=Mock(_frames=[[True], [False]]))
        logical_exp = LogicalExpression(ExpressionType.LOGICAL_AND, comp_exp_l, comp_exp_r)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3, 4], 1: [1, 2, 5, 6]}))
        self.assertEqual([True, False, False, False], logical_exp.evaluate(tuples).frames[0].tolist())
        comp_exp_r.evaluate.assert_called_once_with(tuples[[0, 1]])

    def test_short_circuiting_or_partial(self):
        tup_val_exp_l = TupleValueExpression(name=0)
        tup_val_exp_l.col_alias = 0
        tup_val_exp_r = TupleValueExpression(name=1)
        tup_val_exp_r.col_alias = 1
        comp_exp_l = ComparisonExpression(ExpressionType.COMPARE_EQUAL, tup_val_exp_l, tup_val_exp_r)
        comp_exp_r = Mock(spec=ComparisonExpression)
        comp_exp_r.evaluate = Mock(return_value=Mock(_frames=[[True], [False]]))
        logical_exp = LogicalExpression(ExpressionType.LOGICAL_OR, comp_exp_l, comp_exp_r)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3, 4], 1: [5, 6, 3, 4]}))
        self.assertEqual([True, False, True, True], logical_exp.evaluate(tuples).frames[0].tolist())
        comp_exp_r.evaluate.assert_called_once_with(tuples[[0, 1]])

    def test_multiple_logical(self):
        batch = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        comp_left = ComparisonExpression(ExpressionType.COMPARE_GREATER, TupleValueExpression(col_alias='col'), ConstantValueExpression(1))
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[list(range(2, 10))]
        batch_copy.drop_zero(comp_left.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        comp_right = ComparisonExpression(ExpressionType.COMPARE_LESSER, TupleValueExpression(col_alias='col'), ConstantValueExpression(8))
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[list(range(0, 8))]
        batch_copy.drop_zero(comp_right.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        comp_expr = ComparisonExpression(ExpressionType.COMPARE_GEQ, TupleValueExpression(col_alias='col'), ConstantValueExpression(5))
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[list(range(5, 10))]
        batch_copy.drop_zero(comp_expr.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        l_expr = LogicalExpression(ExpressionType.LOGICAL_AND, comp_left, comp_right)
        root_l_expr = LogicalExpression(ExpressionType.LOGICAL_AND, comp_expr, l_expr)
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[[5, 6, 7]]
        batch_copy.drop_zero(root_l_expr.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        root_l_expr = LogicalExpression(ExpressionType.LOGICAL_AND, l_expr, comp_expr)
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[[5, 6, 7]]
        batch_copy.drop_zero(root_l_expr.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        between_4_7 = LogicalExpression(ExpressionType.LOGICAL_AND, ComparisonExpression(ExpressionType.COMPARE_GEQ, TupleValueExpression(col_alias='col'), ConstantValueExpression(4)), ComparisonExpression(ExpressionType.COMPARE_LEQ, TupleValueExpression(col_alias='col'), ConstantValueExpression(7)))
        test_expr = LogicalExpression(ExpressionType.LOGICAL_AND, between_4_7, l_expr)
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[[4, 5, 6, 7]]
        batch_copy.drop_zero(test_expr.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        test_expr = LogicalExpression(ExpressionType.LOGICAL_OR, between_4_7, l_expr)
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[[2, 3, 4, 5, 6, 7]]
        batch_copy.drop_zero(test_expr.evaluate(batch))
        self.assertEqual(batch_copy, expected)

class ExpressionUtilsTest(unittest.TestCase):

    def gen_cmp_expr(self, val, expr_type=ExpressionType.COMPARE_GREATER, name='id', const_first=False):
        constexpr = ConstantValueExpression(val)
        colname = TupleValueExpression(name=name, col_alias=f'T.{name}')
        if const_first:
            return ComparisonExpression(expr_type, constexpr, colname)
        return ComparisonExpression(expr_type, colname, constexpr)

    def test_extract_range_list_from_comparison_expr(self):
        expr_types = [ExpressionType.COMPARE_NEQ, ExpressionType.COMPARE_EQUAL, ExpressionType.COMPARE_GREATER, ExpressionType.COMPARE_LESSER, ExpressionType.COMPARE_GEQ, ExpressionType.COMPARE_LEQ]
        results = []
        for expr_type in expr_types:
            cmpr_exp = self.gen_cmp_expr(10, expr_type, const_first=True)
            results.append(extract_range_list_from_comparison_expr(cmpr_exp, 0, 100))
        expected = [[(0, 9), (11, 100)], [(10, 10)], [(0, 9)], [(11, 100)], [(0, 10)], [(10, 100)]]
        self.assertEqual(results, expected)
        results = []
        for expr_type in expr_types:
            cmpr_exp = self.gen_cmp_expr(10, expr_type)
            results.append(extract_range_list_from_comparison_expr(cmpr_exp, 0, 100))
        expected = [[(0, 9), (11, 100)], [(10, 10)], [(11, 100)], [(0, 9)], [(10, 100)], [(0, 10)]]
        self.assertEqual(results, expected)
        with self.assertRaises(RuntimeError):
            cmpr_exp = LogicalExpression(ExpressionType.LOGICAL_AND, Mock(), Mock())
            extract_range_list_from_comparison_expr(cmpr_exp, 0, 100)
        with self.assertRaises(RuntimeError):
            cmpr_exp = self.gen_cmp_expr(10, ExpressionType.COMPARE_CONTAINS)
            extract_range_list_from_comparison_expr(cmpr_exp, 0, 100)
        with self.assertRaises(RuntimeError):
            cmpr_exp = self.gen_cmp_expr(10, ExpressionType.COMPARE_IS_CONTAINED)
            extract_range_list_from_comparison_expr(cmpr_exp, 0, 100)

    def test_extract_range_list_from_predicate(self):
        expr = LogicalExpression(ExpressionType.LOGICAL_AND, self.gen_cmp_expr(10), self.gen_cmp_expr(20))
        self.assertEqual(extract_range_list_from_predicate(expr, 0, 100), [(21, 100)])
        expr = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10), self.gen_cmp_expr(20))
        self.assertEqual(extract_range_list_from_predicate(expr, 0, 100), [(11, 100)])
        expr1 = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10), self.gen_cmp_expr(20))
        expr2 = LogicalExpression(ExpressionType.LOGICAL_AND, self.gen_cmp_expr(10), self.gen_cmp_expr(5, ExpressionType.COMPARE_LESSER))
        expr = LogicalExpression(ExpressionType.LOGICAL_OR, expr1, expr2)
        self.assertEqual(extract_range_list_from_predicate(expr, 0, 100), [(11, 100)])
        expr = LogicalExpression(ExpressionType.LOGICAL_AND, expr1, expr2)
        self.assertEqual(extract_range_list_from_predicate(expr, 0, 100), [])
        expr1 = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10, ExpressionType.COMPARE_LESSER), self.gen_cmp_expr(20))
        expr2 = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(25), self.gen_cmp_expr(5, ExpressionType.COMPARE_LESSER))
        expr = LogicalExpression(ExpressionType.LOGICAL_OR, expr1, expr2)
        self.assertEqual(extract_range_list_from_predicate(expr, 0, 100), [(0, 9), (21, 100)])
        with self.assertRaises(RuntimeError):
            expr = ArithmeticExpression(ExpressionType.AGGREGATION_COUNT, Mock(), Mock())
            extract_range_list_from_predicate(expr, 0, 100)

    def test_predicate_contains_single_column(self):
        self.assertTrue(contains_single_column(self.gen_cmp_expr(10)))
        expr1 = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'x'), self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'x'))
        self.assertTrue(contains_single_column(expr1))
        expr2 = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'x'), self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'y'))
        self.assertFalse(contains_single_column(expr2))
        expr = LogicalExpression(ExpressionType.LOGICAL_OR, expr1, expr2)
        self.assertFalse(contains_single_column(expr))

    def test_is_simple_predicate(self):
        self.assertTrue(is_simple_predicate(self.gen_cmp_expr(10)))
        expr = ArithmeticExpression(ExpressionType.AGGREGATION_COUNT, Mock(), Mock())
        self.assertFalse(is_simple_predicate(expr))
        expr = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'x'), self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'y'))
        self.assertFalse(is_simple_predicate(expr))

    def test_and_(self):
        expr1 = self.gen_cmp_expr(10)
        expr2 = self.gen_cmp_expr(20)
        new_expr = conjunction_list_to_expression_tree([expr1, expr2])
        self.assertEqual(new_expr.etype, ExpressionType.LOGICAL_AND)
        self.assertEqual(new_expr.children[0], expr1)
        self.assertEqual(new_expr.children[1], expr2)

def extract_range_list_from_comparison_expr(expr: ComparisonExpression, lower_bound: int, upper_bound: int) -> List:
    """Extracts the valid range from the comparison expression.

    The expression needs to be amongst <, >, <=, >=, =, !=.

    Args:
        expr (ComparisonExpression): comparison expression with two children
            that are leaf expression nodes. If the input does not match,
            the function return False
        lower_bound (int): lower bound of the comparison predicate
        upper_bound (int): upper bound of the comparison predicate

    Returns:
        List[Tuple(int)]: list of valid ranges

    Raises:
        RuntimeError: Invalid expression

    Example:
        extract_range_from_comparison_expr(id < 10, 0, inf): True, [(0,9)]
    """
    if not isinstance(expr, ComparisonExpression):
        raise RuntimeError(f'Expected Comparison Expression, got {type(expr)}')
    left = expr.children[0]
    right = expr.children[1]
    expr_type = expr.etype
    val = None
    const_first = False
    if isinstance(left, TupleValueExpression) and isinstance(right, ConstantValueExpression):
        val = right.value
    elif isinstance(left, ConstantValueExpression) and isinstance(right, TupleValueExpression):
        val = left.value
        const_first = True
    else:
        raise RuntimeError(f'Only supports extracting range from Comparison Expression                 with two children TupleValueExpression and                 ConstantValueExpression, got {left} and {right}')
    if const_first:
        if expr_type is ExpressionType.COMPARE_GREATER:
            expr_type = ExpressionType.COMPARE_LESSER
        elif expr_type is ExpressionType.COMPARE_LESSER:
            expr_type = ExpressionType.COMPARE_GREATER
        elif expr_type is ExpressionType.COMPARE_GEQ:
            expr_type = ExpressionType.COMPARE_LEQ
        elif expr_type is ExpressionType.COMPARE_LEQ:
            expr_type = ExpressionType.COMPARE_GEQ
    valid_ranges = []
    if expr_type == ExpressionType.COMPARE_EQUAL:
        valid_ranges.append((val, val))
    elif expr_type == ExpressionType.COMPARE_NEQ:
        valid_ranges.append((lower_bound, val - 1))
        valid_ranges.append((val + 1, upper_bound))
    elif expr_type == ExpressionType.COMPARE_GREATER:
        valid_ranges.append((val + 1, upper_bound))
    elif expr_type == ExpressionType.COMPARE_GEQ:
        valid_ranges.append((val, upper_bound))
    elif expr_type == ExpressionType.COMPARE_LESSER:
        valid_ranges.append((lower_bound, val - 1))
    elif expr_type == ExpressionType.COMPARE_LEQ:
        valid_ranges.append((lower_bound, val))
    else:
        raise RuntimeError(f'Unsupported Expression Type {expr_type}')
    return valid_ranges

def extract_range_list_from_predicate(predicate: AbstractExpression, lower_bound: int, upper_bound: int) -> List:
    """The function converts the range predicate on the column in the
        `predicate` to a list of [(start_1, end_1), ... ] pairs.

        It assumes the predicate contains conditions on only one column.
        It is the responsibility of the caller that `predicate` does not contains conditions on multiple columns.

    Args:
        predicate (AbstractExpression): Input predicate to extract
            valid ranges. The predicate should contain conditions on
            only one columns, else it raise error.
        lower_bound (int): lower bound of the comparison predicate
        upper_bound (int): upper bound of the comparison predicate

    Returns:
        List[Tuple]: list of (start, end) pairs of valid ranges

    Example:
            id < 10 : [(0, 9)]
            id > 5 AND id < 10 : [(6, 9)]
            id < 10 OR id >20 : [(0, 9), (21, Inf)]
    """

    def overlap(x, y):
        overlap = (max(x[0], y[0]), min(x[1], y[1]))
        if overlap[0] <= overlap[1]:
            return overlap

    def union(ranges: List):
        reduced_list = []
        for begin, end in sorted(ranges):
            if reduced_list and reduced_list[-1][1] >= begin - 1:
                reduced_list[-1] = (reduced_list[-1][0], max(reduced_list[-1][1], end))
            else:
                reduced_list.append((begin, end))
        return reduced_list
    if predicate.etype == ExpressionType.LOGICAL_AND:
        left_ranges = extract_range_list_from_predicate(predicate.children[0], lower_bound, upper_bound)
        right_ranges = extract_range_list_from_predicate(predicate.children[1], lower_bound, upper_bound)
        valid_overlaps = []
        for left_range in left_ranges:
            for right_range in right_ranges:
                over = overlap(left_range, right_range)
                if over:
                    valid_overlaps.append(over)
        return union(valid_overlaps)
    elif predicate.etype == ExpressionType.LOGICAL_OR:
        left_ranges = extract_range_list_from_predicate(predicate.children[0], lower_bound, upper_bound)
        right_ranges = extract_range_list_from_predicate(predicate.children[1], lower_bound, upper_bound)
        return union(left_ranges + right_ranges)
    elif isinstance(predicate, ComparisonExpression):
        return union(extract_range_list_from_comparison_expr(predicate, lower_bound, upper_bound))
    else:
        raise RuntimeError(f'Contains unsupported expression {type(predicate)}')

class ComparisonExpressionsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = Batch(pd.DataFrame([0]))

    def test_comparison_compare_equal(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(1)
        cmpr_exp = ComparisonExpression(ExpressionType.COMPARE_EQUAL, const_exp1, const_exp2)
        self.assertEqual([True], cmpr_exp.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp), None)

    def test_comparison_compare_greater(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(0)
        cmpr_exp = ComparisonExpression(ExpressionType.COMPARE_GREATER, const_exp1, const_exp2)
        self.assertEqual([True], cmpr_exp.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp), None)

    def test_comparison_compare_lesser(self):
        const_exp1 = ConstantValueExpression(0)
        const_exp2 = ConstantValueExpression(2)
        cmpr_exp = ComparisonExpression(ExpressionType.COMPARE_LESSER, const_exp1, const_exp2)
        self.assertEqual([True], cmpr_exp.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp), None)

    def test_comparison_compare_geq(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(1)
        const_exp3 = ConstantValueExpression(0)
        cmpr_exp1 = ComparisonExpression(ExpressionType.COMPARE_GEQ, const_exp1, const_exp2)
        cmpr_exp2 = ComparisonExpression(ExpressionType.COMPARE_GEQ, const_exp1, const_exp3)
        self.assertEqual([True], cmpr_exp1.evaluate(self.batch).frames[0].tolist())
        self.assertEqual([True], cmpr_exp2.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp1), None)

    def test_comparison_compare_leq(self):
        const_exp1 = ConstantValueExpression(0)
        const_exp2 = ConstantValueExpression(2)
        const_exp3 = ConstantValueExpression(2)
        cmpr_exp1 = ComparisonExpression(ExpressionType.COMPARE_LEQ, const_exp1, const_exp2)
        cmpr_exp2 = ComparisonExpression(ExpressionType.COMPARE_LEQ, const_exp2, const_exp3)
        self.assertEqual([True], cmpr_exp1.evaluate(self.batch).frames[0].tolist())
        self.assertEqual([True], cmpr_exp2.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp1), None)

    def test_comparison_compare_neq(self):
        const_exp1 = ConstantValueExpression(0)
        const_exp2 = ConstantValueExpression(1)
        cmpr_exp = ComparisonExpression(ExpressionType.COMPARE_NEQ, const_exp1, const_exp2)
        self.assertEqual([True], cmpr_exp.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp), None)

    def test_comparison_compare_contains(self):
        const_exp1 = ConstantValueExpression([1, 2], ColumnType.NDARRAY)
        const_exp2 = ConstantValueExpression([1, 5], ColumnType.NDARRAY)
        const_exp3 = ConstantValueExpression([1, 2, 3, 4], ColumnType.NDARRAY)
        cmpr_exp1 = ComparisonExpression(ExpressionType.COMPARE_CONTAINS, const_exp3, const_exp1)
        self.assertEqual([True], cmpr_exp1.evaluate(self.batch).frames[0].tolist())
        cmpr_exp2 = ComparisonExpression(ExpressionType.COMPARE_CONTAINS, const_exp3, const_exp2)
        self.assertEqual([False], cmpr_exp2.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp1), None)

    def test_comparison_compare_is_contained(self):
        const_exp1 = ConstantValueExpression([1, 2], ColumnType.NDARRAY)
        const_exp2 = ConstantValueExpression([1, 5], ColumnType.NDARRAY)
        const_exp3 = ConstantValueExpression([1, 2, 3, 4], ColumnType.NDARRAY)
        cmpr_exp1 = ComparisonExpression(ExpressionType.COMPARE_IS_CONTAINED, const_exp1, const_exp3)
        self.assertEqual([True], cmpr_exp1.evaluate(self.batch).frames[0].tolist())
        cmpr_exp2 = ComparisonExpression(ExpressionType.COMPARE_IS_CONTAINED, const_exp2, const_exp3)
        self.assertEqual([False], cmpr_exp2.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp1), None)

class ArithmeticExpressionsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = Batch(pd.DataFrame([None]))

    def test_addition(self):
        const_exp1 = ConstantValueExpression(2)
        const_exp2 = ConstantValueExpression(5)
        cmpr_exp = ArithmeticExpression(ExpressionType.ARITHMETIC_ADD, const_exp1, const_exp2)
        self.assertEqual([7], cmpr_exp.evaluate(self.batch).frames[0].tolist())

    def test_subtraction(self):
        const_exp1 = ConstantValueExpression(5)
        const_exp2 = ConstantValueExpression(2)
        cmpr_exp = ArithmeticExpression(ExpressionType.ARITHMETIC_SUBTRACT, const_exp1, const_exp2)
        self.assertEqual([3], cmpr_exp.evaluate(self.batch).frames[0].tolist())

    def test_multiply(self):
        const_exp1 = ConstantValueExpression(3)
        const_exp2 = ConstantValueExpression(5)
        cmpr_exp = ArithmeticExpression(ExpressionType.ARITHMETIC_MULTIPLY, const_exp1, const_exp2)
        self.assertEqual([15], cmpr_exp.evaluate(self.batch).frames[0].tolist())

    def test_divide(self):
        const_exp1 = ConstantValueExpression(5)
        const_exp2 = ConstantValueExpression(5)
        cmpr_exp = ArithmeticExpression(ExpressionType.ARITHMETIC_DIVIDE, const_exp1, const_exp2)
        self.assertEqual([1], cmpr_exp.evaluate(self.batch).frames[0].tolist())

    def test_aaequality(self):
        const_exp1 = ConstantValueExpression(5)
        const_exp2 = ConstantValueExpression(15)
        cmpr_exp = ArithmeticExpression(ExpressionType.ARITHMETIC_DIVIDE, const_exp1, const_exp2)
        cmpr_exp2 = ArithmeticExpression(ExpressionType.ARITHMETIC_MULTIPLY, const_exp1, const_exp2)
        cmpr_exp3 = ArithmeticExpression(ExpressionType.ARITHMETIC_MULTIPLY, const_exp2, const_exp1)
        self.assertNotEqual(cmpr_exp, cmpr_exp2)
        self.assertNotEqual(cmpr_exp2, cmpr_exp3)

class AggregationExpressionsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_aggregation_first(self):
        columnName = TupleValueExpression(name=0)
        columnName.col_alias = 0
        aggr_expr = AggregationExpression(ExpressionType.AGGREGATION_FIRST, None, columnName)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [2, 3, 4], 2: [3, 4, 5]}))
        batch = aggr_expr.evaluate(tuples, None)
        self.assertEqual(1, batch.frames.iloc[0][0])
        self.assertNotEqual(str(aggr_expr), None)

    def test_aggregation_last(self):
        columnName = TupleValueExpression(name=0)
        columnName.col_alias = 0
        aggr_expr = AggregationExpression(ExpressionType.AGGREGATION_LAST, None, columnName)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [2, 3, 4], 2: [3, 4, 5]}))
        batch = aggr_expr.evaluate(tuples, None)
        self.assertEqual(3, batch.frames.iloc[0][0])
        self.assertNotEqual(str(aggr_expr), None)

    def test_aggregation_segment(self):
        columnName = TupleValueExpression(name=0)
        columnName.col_alias = 0
        aggr_expr = AggregationExpression(ExpressionType.AGGREGATION_SEGMENT, None, columnName)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [2, 3, 4], 2: [3, 4, 5]}))
        batch = aggr_expr.evaluate(tuples, None)
        self.assertTrue((np.array([1, 2, 3]) == batch.frames.iloc[0][0]).all())
        self.assertNotEqual(str(aggr_expr), None)

    def test_aggregation_sum(self):
        columnName = TupleValueExpression(name=0)
        columnName.col_alias = 0
        aggr_expr = AggregationExpression(ExpressionType.AGGREGATION_SUM, None, columnName)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [2, 3, 4], 2: [3, 4, 5]}))
        batch = aggr_expr.evaluate(tuples, None)
        self.assertEqual(6, batch.frames.iloc[0][0])
        self.assertNotEqual(str(aggr_expr), None)

    def test_aggregation_count(self):
        columnName = TupleValueExpression(name=0)
        columnName.col_alias = 0
        aggr_expr = AggregationExpression(ExpressionType.AGGREGATION_COUNT, None, columnName)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [2, 3, 4], 2: [3, 4, 5]}))
        batch = aggr_expr.evaluate(tuples, None)
        self.assertEqual(3, batch.frames.iloc[0][0])
        self.assertNotEqual(str(aggr_expr), None)

    def test_aggregation_avg(self):
        columnName = TupleValueExpression(name=0)
        columnName.col_alias = 0
        aggr_expr = AggregationExpression(ExpressionType.AGGREGATION_AVG, None, columnName)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [2, 3, 4], 2: [3, 4, 5]}))
        batch = aggr_expr.evaluate(tuples, None)
        self.assertEqual(2, batch.frames.iloc[0][0])
        self.assertNotEqual(str(aggr_expr), None)

    def test_aggregation_min(self):
        columnName = TupleValueExpression(name=0)
        columnName.col_alias = 0
        aggr_expr = AggregationExpression(ExpressionType.AGGREGATION_MIN, None, columnName)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [2, 3, 4], 2: [3, 4, 5]}))
        batch = aggr_expr.evaluate(tuples, None)
        self.assertEqual(1, batch.frames.iloc[0][0])
        self.assertNotEqual(str(aggr_expr), None)

    def test_aggregation_max(self):
        columnName = TupleValueExpression(name=0)
        columnName.col_alias = 0
        aggr_expr = AggregationExpression(ExpressionType.AGGREGATION_MAX, None, columnName)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [2, 3, 4], 2: [3, 4, 5]}))
        batch = aggr_expr.evaluate(tuples, None)
        self.assertEqual(3, batch.frames.iloc[0][0])
        self.assertNotEqual(str(aggr_expr), None)

    def test_aggregation_incorrect_etype(self):
        incorrect_etype = 100
        columnName = TupleValueExpression(name=0)
        aggr_expr = AggregationExpression(incorrect_etype, columnName, columnName)
        with pytest.raises(NotImplementedError):
            str(aggr_expr)

class ParserTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_select_from_data_source(self):
        parser = Parser()
        query = 'SELECT * FROM DemoDB.DemoTable'
        evadb_stmt_list = parser.parse(query)
        self.assertIsInstance(evadb_stmt_list, list)
        self.assertEqual(len(evadb_stmt_list), 1)
        self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_stmt_list[0]
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'DemoTable')
        self.assertEqual(select_stmt.from_table.table.database_name, 'DemoDB')

    def test_use_statement(self):
        parser = Parser()
        query_list = ['SELECT * FROM DemoTable', 'SELECT * FROM DemoTable WHERE col == "xxx"\n            ', "SELECT * FROM DemoTable WHERE col == 'xxx'\n            "]
        for query in query_list:
            use_query = f'USE DemoDB {{{query}}};'
            evadb_stmt_list = parser.parse(use_query)
            self.assertIsInstance(evadb_stmt_list, list)
            self.assertEqual(len(evadb_stmt_list), 1)
            self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.USE)
            expected_stmt = UseStatement('DemoDB', query)
            actual_stmt = evadb_stmt_list[0]
            self.assertEqual(actual_stmt, expected_stmt)

    def test_create_index_statement(self):
        parser = Parser()
        create_index_query = 'CREATE INDEX testindex ON MyVideo (featCol) USING FAISS;'
        evadb_stmt_list = parser.parse(create_index_query)
        self.assertIsInstance(evadb_stmt_list, list)
        self.assertEqual(len(evadb_stmt_list), 1)
        self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.CREATE_INDEX)
        expected_stmt = CreateIndexStatement('testindex', False, TableRef(TableInfo('MyVideo')), [ColumnDefinition('featCol', None, None, None)], VectorStoreType.FAISS, [TupleValueExpression(name='featCol')])
        actual_stmt = evadb_stmt_list[0]
        self.assertEqual(actual_stmt, expected_stmt)
        self.assertEqual(actual_stmt.index_def, create_index_query)
        expected_stmt = CreateIndexStatement('testindex', True, TableRef(TableInfo('MyVideo')), [ColumnDefinition('featCol', None, None, None)], VectorStoreType.FAISS, [TupleValueExpression(name='featCol')])
        create_index_query = 'CREATE INDEX IF NOT EXISTS testindex ON MyVideo (featCol) USING FAISS;'
        evadb_stmt_list = parser.parse(create_index_query)
        actual_stmt = evadb_stmt_list[0]
        expected_stmt._if_not_exists = True
        self.assertEqual(actual_stmt, expected_stmt)
        self.assertEqual(actual_stmt.index_def, create_index_query)
        create_index_query = 'CREATE INDEX testindex ON MyVideo (FeatureExtractor(featCol)) USING FAISS;'
        evadb_stmt_list = parser.parse(create_index_query)
        self.assertIsInstance(evadb_stmt_list, list)
        self.assertEqual(len(evadb_stmt_list), 1)
        self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.CREATE_INDEX)
        func_expr = FunctionExpression(None, 'FeatureExtractor')
        func_expr.append_child(TupleValueExpression('featCol'))
        expected_stmt = CreateIndexStatement('testindex', False, TableRef(TableInfo('MyVideo')), [ColumnDefinition('featCol', None, None, None)], VectorStoreType.FAISS, [func_expr])
        actual_stmt = evadb_stmt_list[0]
        self.assertEqual(actual_stmt, expected_stmt)
        self.assertEqual(actual_stmt.index_def, create_index_query)

    @unittest.skip('Skip parser exception handling testcase, moved to binder')
    def test_create_index_exception_statement(self):
        parser = Parser()
        create_index_query = 'CREATE INDEX testindex USING FAISS ON MyVideo (featCol1, featCol2);'
        with self.assertRaises(Exception):
            parser.parse(create_index_query)

    def test_explain_dml_statement(self):
        parser = Parser()
        explain_query = 'EXPLAIN SELECT CLASS FROM TAIPAI;'
        evadb_statement_list = parser.parse(explain_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.EXPLAIN)
        inner_stmt = evadb_statement_list[0].explainable_stmt
        self.assertEqual(inner_stmt.stmt_type, StatementType.SELECT)
        self.assertIsNotNone(inner_stmt.from_table)
        self.assertIsInstance(inner_stmt.from_table, TableRef)
        self.assertEqual(inner_stmt.from_table.table.table_name, 'TAIPAI')

    def test_explain_ddl_statement(self):
        parser = Parser()
        select_query = 'SELECT id, Yolo(frame).labels FROM MyVideo\n                        WHERE id<5; '
        explain_query = 'EXPLAIN CREATE TABLE uadtrac_fastRCNN AS {}'.format(select_query)
        evadb_statement_list = parser.parse(explain_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.EXPLAIN)
        inner_stmt = evadb_statement_list[0].explainable_stmt
        self.assertEqual(inner_stmt.stmt_type, StatementType.CREATE)
        self.assertIsNotNone(inner_stmt.table_info, TableRef(TableInfo('uadetrac_fastRCNN')))

    def test_create_table_statement(self):
        parser = Parser()
        single_queries = []
        single_queries.append('CREATE TABLE IF NOT EXISTS Persons (\n                  Frame_ID INTEGER UNIQUE,\n                  Frame_Data TEXT,\n                  Frame_Value FLOAT,\n                  Frame_Array NDARRAY UINT8(5, 100, 2432, 4324, 100)\n            );')
        expected_cci = ColConstraintInfo()
        expected_cci.nullable = True
        unique_cci = ColConstraintInfo()
        unique_cci.unique = True
        unique_cci.nullable = False
        expected_stmt = CreateTableStatement(TableInfo('Persons'), True, [ColumnDefinition('Frame_ID', ColumnType.INTEGER, None, (), unique_cci), ColumnDefinition('Frame_Data', ColumnType.TEXT, None, (), expected_cci), ColumnDefinition('Frame_Value', ColumnType.FLOAT, None, (), expected_cci), ColumnDefinition('Frame_Array', ColumnType.NDARRAY, NdArrayType.UINT8, (5, 100, 2432, 4324, 100), expected_cci)])
        for query in single_queries:
            evadb_statement_list = parser.parse(query)
            self.assertIsInstance(evadb_statement_list, list)
            self.assertEqual(len(evadb_statement_list), 1)
            self.assertIsInstance(evadb_statement_list[0], AbstractStatement)
            self.assertEqual(evadb_statement_list[0], expected_stmt)

    def test_create_table_with_dimension_statement(self):
        parser = Parser()
        single_queries = []
        single_queries.append('CREATE TABLE IF NOT EXISTS Persons (\n                  Frame_ID INTEGER UNIQUE,\n                  Frame_Data TEXT(10),\n                  Frame_Value FLOAT(1000, 201),\n                  Frame_Array NDARRAY UINT8(5, 100, 2432, 4324, 100)\n            );')
        expected_cci = ColConstraintInfo()
        expected_cci.nullable = True
        unique_cci = ColConstraintInfo()
        unique_cci.unique = True
        unique_cci.nullable = False
        expected_stmt = CreateTableStatement(TableInfo('Persons'), True, [ColumnDefinition('Frame_ID', ColumnType.INTEGER, None, (), unique_cci), ColumnDefinition('Frame_Data', ColumnType.TEXT, None, (10,), expected_cci), ColumnDefinition('Frame_Value', ColumnType.FLOAT, None, (1000, 201), expected_cci), ColumnDefinition('Frame_Array', ColumnType.NDARRAY, NdArrayType.UINT8, (5, 100, 2432, 4324, 100), expected_cci)])
        for query in single_queries:
            evadb_statement_list = parser.parse(query)
            self.assertIsInstance(evadb_statement_list, list)
            self.assertEqual(len(evadb_statement_list), 1)
            self.assertIsInstance(evadb_statement_list[0], AbstractStatement)
            self.assertEqual(evadb_statement_list[0], expected_stmt)

    def test_create_table_statement_with_rare_datatypes(self):
        parser = Parser()
        query = 'CREATE TABLE IF NOT EXISTS Dummy (\n                  C NDARRAY UINT8(5),\n                  D NDARRAY INT16(5),\n                  E NDARRAY INT32(5),\n                  F NDARRAY INT64(5),\n                  G NDARRAY UNICODE(5),\n                  H NDARRAY BOOLEAN(5),\n                  I NDARRAY FLOAT64(5),\n                  J NDARRAY DECIMAL(5),\n                  K NDARRAY DATETIME(5)\n            );'
        evadb_statement_list = parser.parse(query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertIsInstance(evadb_statement_list[0], AbstractStatement)

    def test_create_table_statement_without_proper_datatype(self):
        parser = Parser()
        query = 'CREATE TABLE IF NOT EXISTS Dummy (\n                  C NDARRAY INT(5)\n                );'
        with self.assertRaises(Exception):
            parser.parse(query)

    def test_create_table_exception_statement(self):
        parser = Parser()
        create_table_query = 'CREATE TABLE ();'
        with self.assertRaises(Exception):
            parser.parse(create_table_query)

    def test_rename_table_statement(self):
        parser = Parser()
        rename_queries = 'RENAME TABLE student TO student_info'
        expected_stmt = RenameTableStatement(TableRef(TableInfo('student')), TableInfo('student_info'))
        evadb_statement_list = parser.parse(rename_queries)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.RENAME)
        rename_stmt = evadb_statement_list[0]
        self.assertEqual(rename_stmt, expected_stmt)

    def test_drop_table_statement(self):
        parser = Parser()
        drop_queries = 'DROP TABLE student_info'
        expected_stmt = DropObjectStatement(ObjectType.TABLE, 'student_info', False)
        evadb_statement_list = parser.parse(drop_queries)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.DROP_OBJECT)
        drop_stmt = evadb_statement_list[0]
        self.assertEqual(drop_stmt, expected_stmt)

    def test_drop_function_statement_str(self):
        drop_func_query1 = 'DROP FUNCTION MyFunc;'
        drop_func_query2 = 'DROP FUNCTION IF EXISTS MyFunc;'
        expected_stmt1 = DropObjectStatement(ObjectType.FUNCTION, 'MyFunc', False)
        expected_stmt2 = DropObjectStatement(ObjectType.FUNCTION, 'MyFunc', True)
        self.assertEqual(str(expected_stmt1), drop_func_query1)
        self.assertEqual(str(expected_stmt2), drop_func_query2)

    def test_single_statement_queries(self):
        parser = Parser()
        single_queries = []
        single_queries.append('SELECT CLASS FROM TAIPAI;')
        single_queries.append("SELECT CLASS FROM TAIPAI WHERE CLASS = 'VAN';")
        single_queries.append("SELECT CLASS,REDNESS FROM TAIPAI             WHERE CLASS = 'VAN' AND REDNESS > 20.5;")
        single_queries.append("SELECT CLASS FROM TAIPAI             WHERE (CLASS = 'VAN' AND REDNESS < 300 ) OR REDNESS > 500;")
        single_queries.append("SELECT CLASS FROM TAIPAI             WHERE (CLASS = 'VAN' AND REDNESS < 300 ) OR REDNESS > 500;")
        for query in single_queries:
            evadb_statement_list = parser.parse(query)
            self.assertIsInstance(evadb_statement_list, list)
            self.assertEqual(len(evadb_statement_list), 1)
            self.assertIsInstance(evadb_statement_list[0], AbstractStatement)

    def test_multiple_statement_queries(self):
        parser = Parser()
        multiple_queries = []
        multiple_queries.append("SELECT CLASS FROM TAIPAI                 WHERE (CLASS != 'VAN' AND REDNESS < 300)  OR REDNESS > 500;                 SELECT REDNESS FROM TAIPAI                 WHERE (CLASS = 'VAN' AND REDNESS = 300)")
        for query in multiple_queries:
            evadb_statement_list = parser.parse(query)
            self.assertIsInstance(evadb_statement_list, list)
            self.assertEqual(len(evadb_statement_list), 2)
            self.assertIsInstance(evadb_statement_list[0], AbstractStatement)
            self.assertIsInstance(evadb_statement_list[1], AbstractStatement)

    def test_select_statement(self):
        parser = Parser()
        select_query = "SELECT CLASS, REDNESS FROM TAIPAI                 WHERE (CLASS = 'VAN' AND REDNESS < 300 ) OR REDNESS > 500;"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 2)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(select_stmt.target_list[1].etype, ExpressionType.TUPLE_VALUE)
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
        self.assertIsNotNone(select_stmt.where_clause)

    def test_select_with_empty_string_literal(self):
        parser = Parser()
        select_query = "SELECT '' FROM TAIPAI;"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)

    def test_string_literal_with_escaped_single_quote(self):
        parser = Parser()
        select_query = "SELECT ChatGPT('Here\\'s a question', 'This is the context') FROM TAIPAI;"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)

    def test_string_literal_with_semi_colon(self):
        parser = Parser()
        select_query = 'SELECT ChatGPT("Here\'s a; question", "This is the context") FROM TAIPAI;'
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)

    def test_string_literal_with_single_quotes_from_variable(self):
        parser = Parser()
        question = json.dumps("Here's a question")
        answer = json.dumps('This is "the" context')
        select_query = f'SELECT ChatGPT({question}, {answer}) FROM TAIPAI;'
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)

    def test_select_union_statement(self):
        parser = Parser()
        select_union_query = 'SELECT CLASS, REDNESS FROM TAIPAI             UNION ALL SELECT CLASS, REDNESS FROM SHANGHAI;'
        evadb_statement_list = parser.parse(select_union_query)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.union_link)
        self.assertEqual(select_stmt.union_all, True)
        second_select_stmt = select_stmt.union_link
        self.assertIsNone(second_select_stmt.union_link)

    def test_select_statement_class(self):
        """Testing setting different clauses for Select
        Statement class
        Class: SelectStatement"""
        select_stmt_new = SelectStatement()
        parser = Parser()
        select_query_new = "SELECT CLASS, REDNESS FROM TAIPAI             WHERE (CLASS = 'VAN' AND REDNESS < 400 ) OR REDNESS > 700;"
        evadb_statement_list = parser.parse(select_query_new)
        select_stmt = evadb_statement_list[0]
        select_stmt_new.where_clause = select_stmt.where_clause
        select_stmt_new.target_list = select_stmt.target_list
        select_stmt_new.from_table = select_stmt.from_table
        self.assertEqual(select_stmt_new.where_clause, select_stmt.where_clause)
        self.assertEqual(select_stmt_new.target_list, select_stmt.target_list)
        self.assertEqual(select_stmt_new.from_table, select_stmt.from_table)
        self.assertEqual(str(select_stmt_new), str(select_stmt))

    def test_select_statement_where_class(self):
        """
        Unit test for logical operators in the where clause.
        """

        def _verify_select_statement(evadb_statement_list):
            self.assertIsInstance(evadb_statement_list, list)
            self.assertEqual(len(evadb_statement_list), 1)
            self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
            select_stmt = evadb_statement_list[0]
            self.assertIsNotNone(select_stmt.target_list)
            self.assertEqual(len(select_stmt.target_list), 2)
            self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
            self.assertEqual(select_stmt.target_list[0].name, 'CLASS')
            self.assertEqual(select_stmt.target_list[1].etype, ExpressionType.TUPLE_VALUE)
            self.assertEqual(select_stmt.target_list[1].name, 'REDNESS')
            self.assertIsNotNone(select_stmt.from_table)
            self.assertIsInstance(select_stmt.from_table, TableRef)
            self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
            self.assertIsNotNone(select_stmt.where_clause)
            self.assertIsInstance(select_stmt.where_clause, LogicalExpression)
            self.assertEqual(select_stmt.where_clause.etype, ExpressionType.LOGICAL_AND)
            self.assertEqual(len(select_stmt.where_clause.children), 2)
            left = select_stmt.where_clause.children[0]
            right = select_stmt.where_clause.children[1]
            self.assertEqual(left.etype, ExpressionType.COMPARE_EQUAL)
            self.assertEqual(right.etype, ExpressionType.COMPARE_LESSER)
            self.assertEqual(len(left.children), 2)
            self.assertEqual(left.children[0].etype, ExpressionType.TUPLE_VALUE)
            self.assertEqual(left.children[0].name, 'CLASS')
            self.assertEqual(left.children[1].etype, ExpressionType.CONSTANT_VALUE)
            self.assertEqual(left.children[1].value, 'VAN')
            self.assertEqual(len(right.children), 2)
            self.assertEqual(right.children[0].etype, ExpressionType.TUPLE_VALUE)
            self.assertEqual(right.children[0].name, 'REDNESS')
            self.assertEqual(right.children[1].etype, ExpressionType.CONSTANT_VALUE)
            self.assertEqual(right.children[1].value, 400)
        parser = Parser()
        select_query = "SELECT CLASS, REDNESS FROM TAIPAI WHERE CLASS = 'VAN' AND REDNESS < 400;"
        _verify_select_statement(parser.parse(select_query))
        select_query = "select CLASS, REDNESS from TAIPAI where CLASS = 'VAN' and REDNESS < 400;"
        _verify_select_statement(parser.parse(select_query))
        select_query = "SELECT CLASS, REDNESS FROM TAIPAI WHERE CLASS = 'VAN' XOR REDNESS < 400;"
        with self.assertRaises(NotImplementedError) as cm:
            parser.parse(select_query)
        self.assertEqual(str(cm.exception), 'Unsupported logical operator: XOR')

    def test_select_statement_groupby_class(self):
        """Testing sample frequency"""
        parser = Parser()
        select_query = "SELECT FIRST(id) FROM TAIPAI GROUP BY '8 frames';"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 1)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.AGGREGATION_FIRST)
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
        self.assertEqual(select_stmt.groupby_clause, ConstantValueExpression('8 frames', v_type=ColumnType.TEXT))

    def test_select_statement_orderby_class(self):
        """Testing order by clause in select statement
        Class: SelectStatement"""
        parser = Parser()
        select_query = "SELECT CLASS, REDNESS FROM TAIPAI                     WHERE (CLASS = 'VAN' AND REDNESS < 400 ) OR REDNESS > 700                     ORDER BY CLASS, REDNESS DESC;"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 2)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(select_stmt.target_list[1].etype, ExpressionType.TUPLE_VALUE)
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
        self.assertIsNotNone(select_stmt.where_clause)
        self.assertIsNotNone(select_stmt.orderby_list)
        self.assertEqual(len(select_stmt.orderby_list), 2)
        self.assertEqual(select_stmt.orderby_list[0][0].name, 'CLASS')
        self.assertEqual(select_stmt.orderby_list[0][1], ParserOrderBySortType.ASC)
        self.assertEqual(select_stmt.orderby_list[1][0].name, 'REDNESS')
        self.assertEqual(select_stmt.orderby_list[1][1], ParserOrderBySortType.DESC)

    def test_select_statement_limit_class(self):
        """Testing limit clause in select statement
        Class: SelectStatement"""
        parser = Parser()
        select_query = "SELECT CLASS, REDNESS FROM TAIPAI                     WHERE (CLASS = 'VAN' AND REDNESS < 400 ) OR REDNESS > 700                     ORDER BY CLASS, REDNESS DESC LIMIT 3;"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 2)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(select_stmt.target_list[1].etype, ExpressionType.TUPLE_VALUE)
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
        self.assertIsNotNone(select_stmt.where_clause)
        self.assertIsNotNone(select_stmt.orderby_list)
        self.assertEqual(len(select_stmt.orderby_list), 2)
        self.assertEqual(select_stmt.orderby_list[0][0].name, 'CLASS')
        self.assertEqual(select_stmt.orderby_list[0][1], ParserOrderBySortType.ASC)
        self.assertEqual(select_stmt.orderby_list[1][0].name, 'REDNESS')
        self.assertEqual(select_stmt.orderby_list[1][1], ParserOrderBySortType.DESC)
        self.assertIsNotNone(select_stmt.limit_count)
        self.assertEqual(select_stmt.limit_count, ConstantValueExpression(3))

    def test_select_statement_sample_class(self):
        """Testing sample frequency"""
        parser = Parser()
        select_query = 'SELECT CLASS, REDNESS FROM TAIPAI SAMPLE 5;'
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 2)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(select_stmt.target_list[1].etype, ExpressionType.TUPLE_VALUE)
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
        self.assertEqual(select_stmt.from_table.sample_freq, ConstantValueExpression(5))

    def test_select_function_star(self):
        parser = Parser()
        query = 'SELECT DemoFunc(*) FROM DemoDB.DemoTable;'
        evadb_stmt_list = parser.parse(query)
        self.assertIsInstance(evadb_stmt_list, list)
        self.assertEqual(len(evadb_stmt_list), 1)
        self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_stmt_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 1)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.FUNCTION_EXPRESSION)
        self.assertEqual(len(select_stmt.target_list[0].children), 1)
        self.assertEqual(select_stmt.target_list[0].children[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(select_stmt.target_list[0].children[0].name, '*')
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'DemoTable')
        self.assertEqual(select_stmt.from_table.table.database_name, 'DemoDB')

    def test_select_without_table_source(self):
        parser = Parser()
        query = 'SELECT DemoFunc(12);'
        evadb_stmt_list = parser.parse(query)
        self.assertIsInstance(evadb_stmt_list, list)
        self.assertEqual(len(evadb_stmt_list), 1)
        self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_stmt_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 1)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.FUNCTION_EXPRESSION)
        self.assertEqual(len(select_stmt.target_list[0].children), 1)
        self.assertEqual(select_stmt.target_list[0].children[0].etype, ExpressionType.CONSTANT_VALUE)
        self.assertEqual(select_stmt.target_list[0].children[0].value, 12)
        self.assertIsNone(select_stmt.from_table)

    def test_table_ref(self):
        """Testing table info in TableRef
        Class: TableInfo
        """
        table_info = TableInfo('TAIPAI', 'Schema', 'Database')
        table_ref_obj = TableRef(table_info)
        select_stmt_new = SelectStatement()
        select_stmt_new.from_table = table_ref_obj
        self.assertEqual(select_stmt_new.from_table.table.table_name, 'TAIPAI')
        self.assertEqual(select_stmt_new.from_table.table.schema_name, 'Schema')
        self.assertEqual(select_stmt_new.from_table.table.database_name, 'Database')

    def test_insert_statement(self):
        parser = Parser()
        insert_query = "INSERT INTO MyVideo (Frame_ID, Frame_Path)\n                                    VALUES    (1, '/mnt/frames/1.png');\n                        "
        expected_stmt = InsertTableStatement(TableRef(TableInfo('MyVideo')), [TupleValueExpression('Frame_ID'), TupleValueExpression('Frame_Path')], [[ConstantValueExpression(1), ConstantValueExpression('/mnt/frames/1.png', ColumnType.TEXT)]])
        evadb_statement_list = parser.parse(insert_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.INSERT)
        insert_stmt = evadb_statement_list[0]
        self.assertEqual(insert_stmt, expected_stmt)

    def test_delete_statement(self):
        parser = Parser()
        delete_statement = 'DELETE FROM Foo WHERE id > 5'
        evadb_statement_list = parser.parse(delete_statement)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.DELETE)
        delete_stmt = evadb_statement_list[0]
        expected_stmt = DeleteTableStatement(TableRef(TableInfo('Foo')), ComparisonExpression(ExpressionType.COMPARE_GREATER, TupleValueExpression('id'), ConstantValueExpression(5)))
        self.assertEqual(delete_stmt, expected_stmt)

    def test_set_statement(self):
        parser = Parser()
        set_statement = "SET OPENAIKEY = 'ABCD'"
        evadb_statement_list = parser.parse(set_statement)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SET)
        set_stmt = evadb_statement_list[0]
        expected_stmt = SetStatement('OPENAIKEY', ConstantValueExpression('ABCD', ColumnType.TEXT))
        self.assertEqual(set_stmt, expected_stmt)
        set_statement = "SET OPENAIKEY TO 'ABCD'"
        evadb_statement_list = parser.parse(set_statement)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SET)
        set_stmt = evadb_statement_list[0]
        expected_stmt = SetStatement('OPENAIKEY', ConstantValueExpression('ABCD', ColumnType.TEXT))
        self.assertEqual(set_stmt, expected_stmt)

    def test_show_config_statement(self):
        parser = Parser()
        show_config_statement = 'SHOW OPENAIKEY'
        evadb_statement_list = parser.parse(show_config_statement)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SHOW)
        show_config_stmt = evadb_statement_list[0]
        expected_stmt = ShowStatement(show_type=ShowType.CONFIGS, show_val='OPENAIKEY')
        self.assertEqual(show_config_stmt, expected_stmt)

    def test_create_predict_function_statement(self):
        parser = Parser()
        create_func_query = "\n            CREATE OR REPLACE FUNCTION HomeSalesForecast FROM\n            ( SELECT * FROM postgres_data.home_sales )\n            TYPE Forecasting\n            PREDICT 'price';\n        "
        evadb_statement_list = parser.parse(create_func_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.CREATE_FUNCTION)
        create_func_stmt = evadb_statement_list[0]
        self.assertEqual(create_func_stmt.name, 'HomeSalesForecast')
        self.assertEqual(create_func_stmt.or_replace, True)
        self.assertEqual(create_func_stmt.if_not_exists, False)
        self.assertEqual(create_func_stmt.impl_path, None)
        self.assertEqual(create_func_stmt.inputs, [])
        self.assertEqual(create_func_stmt.outputs, [])
        self.assertEqual(create_func_stmt.function_type, 'Forecasting')
        self.assertEqual(create_func_stmt.metadata, [('predict', 'price')])
        nested_select_stmt = create_func_stmt.query
        self.assertEqual(nested_select_stmt.stmt_type, StatementType.SELECT)
        self.assertEqual(len(nested_select_stmt.target_list), 1)
        self.assertEqual(nested_select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(nested_select_stmt.target_list[0].name, '*')
        self.assertIsInstance(nested_select_stmt.from_table, TableRef)
        self.assertIsInstance(nested_select_stmt.from_table.table, TableInfo)
        self.assertEqual(nested_select_stmt.from_table.table.table_name, 'home_sales')
        self.assertEqual(nested_select_stmt.from_table.table.database_name, 'postgres_data')

    def test_create_function_statement(self):
        parser = Parser()
        create_func_query = 'CREATE FUNCTION IF NOT EXISTS FastRCNN\n                  INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n                  OUTPUT (Labels NDARRAY STR(10), Bbox NDARRAY UINT8(10, 4))\n                  TYPE  Classification\n                  IMPL  \'data/fastrcnn.py\'\n                  PREDICT "VALUE";\n        '
        expected_cci = ColConstraintInfo()
        expected_cci.nullable = True
        expected_stmt = CreateFunctionStatement('FastRCNN', False, True, Path('data/fastrcnn.py'), [ColumnDefinition('Frame_Array', ColumnType.NDARRAY, NdArrayType.UINT8, (3, 256, 256), expected_cci)], [ColumnDefinition('Labels', ColumnType.NDARRAY, NdArrayType.STR, (10,), expected_cci), ColumnDefinition('Bbox', ColumnType.NDARRAY, NdArrayType.UINT8, (10, 4), expected_cci)], 'Classification', None, [('predict', 'VALUE')])
        evadb_statement_list = parser.parse(create_func_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.CREATE_FUNCTION)
        self.assertEqual(str(evadb_statement_list[0]), str(expected_stmt))
        create_func_stmt = evadb_statement_list[0]
        self.assertEqual(create_func_stmt, expected_stmt)

    def test_load_video_data_statement(self):
        parser = Parser()
        load_data_query = "LOAD VIDEO 'data/video.mp4'\n                             INTO MyVideo"
        file_options = {}
        file_options['file_format'] = FileFormatType.VIDEO
        column_list = None
        expected_stmt = LoadDataStatement(TableInfo('MyVideo'), Path('data/video.mp4'), column_list, file_options)
        evadb_statement_list = parser.parse(load_data_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.LOAD_DATA)
        load_data_stmt = evadb_statement_list[0]
        self.assertEqual(load_data_stmt, expected_stmt)

    def test_load_csv_data_statement(self):
        parser = Parser()
        load_data_query = "LOAD CSV 'data/meta.csv'\n                             INTO\n                             MyMeta (id, frame_id, video_id, label);"
        file_options = {}
        file_options['file_format'] = FileFormatType.CSV
        expected_stmt = LoadDataStatement(TableInfo('MyMeta'), Path('data/meta.csv'), [TupleValueExpression('id'), TupleValueExpression('frame_id'), TupleValueExpression('video_id'), TupleValueExpression('label')], file_options)
        evadb_statement_list = parser.parse(load_data_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.LOAD_DATA)
        load_data_stmt = evadb_statement_list[0]
        self.assertEqual(load_data_stmt, expected_stmt)

    def test_nested_select_statement(self):
        parser = Parser()
        sub_query = "SELECT CLASS FROM TAIPAI WHERE CLASS = 'VAN'"
        nested_query = 'SELECT ID FROM ({}) AS T;'.format(sub_query)
        parsed_sub_query = parser.parse(sub_query)[0]
        actual_stmt = parser.parse(nested_query)[0]
        self.assertEqual(actual_stmt.stmt_type, StatementType.SELECT)
        self.assertEqual(actual_stmt.target_list[0].name, 'ID')
        self.assertEqual(actual_stmt.from_table, TableRef(parsed_sub_query, alias=Alias('T')))
        sub_query = "SELECT Yolo(frame).bbox FROM autonomous_vehicle_1\n                              WHERE Yolo(frame).label = 'vehicle'"
        nested_query = "SELECT Licence_plate(bbox) FROM\n                            ({}) AS T\n                          WHERE Is_suspicious(bbox) = 1 AND\n                                Licence_plate(bbox) = '12345';\n                      ".format(sub_query)
        query = "SELECT Licence_plate(bbox) FROM TAIPAI\n                    WHERE Is_suspicious(bbox) = 1 AND\n                        Licence_plate(bbox) = '12345';\n                "
        query_stmt = parser.parse(query)[0]
        actual_stmt = parser.parse(nested_query)[0]
        sub_query_stmt = parser.parse(sub_query)[0]
        self.assertEqual(actual_stmt.from_table, TableRef(sub_query_stmt, alias=Alias('T')))
        self.assertEqual(actual_stmt.where_clause, query_stmt.where_clause)
        self.assertEqual(actual_stmt.target_list, query_stmt.target_list)

    def test_should_return_false_for_unequal_expression(self):
        table = TableRef(TableInfo('MyVideo'))
        load_stmt = LoadDataStatement(table, Path('data/video.mp4'), FileFormatType.VIDEO)
        insert_stmt = InsertTableStatement(table)
        create_func = CreateFunctionStatement('func', False, False, Path('data/fastrcnn.py'), [ColumnDefinition('frame', ColumnType.NDARRAY, NdArrayType.UINT8, (3, 256, 256))], [ColumnDefinition('labels', ColumnType.NDARRAY, NdArrayType.STR, 10)], 'Classification')
        select_stmt = SelectStatement()
        self.assertNotEqual(load_stmt, insert_stmt)
        self.assertNotEqual(insert_stmt, load_stmt)
        self.assertNotEqual(create_func, insert_stmt)
        self.assertNotEqual(select_stmt, create_func)

    def test_create_table_from_select(self):
        select_query = 'SELECT id, Yolo(frame).labels FROM MyVideo\n                        WHERE id<5; '
        query = 'CREATE TABLE uadtrac_fastRCNN AS {}'.format(select_query)
        parser = Parser()
        mat_view_stmt = parser.parse(query)
        select_stmt = parser.parse(select_query)
        expected_stmt = CreateTableStatement(TableInfo('uadtrac_fastRCNN'), False, [], select_stmt[0])
        self.assertEqual(mat_view_stmt[0], expected_stmt)

    def test_join(self):
        select_query = 'SELECT table1.a FROM table1 JOIN table2\n                    ON table1.a = table2.a; '
        parser = Parser()
        select_stmt = parser.parse(select_query)[0]
        table1_col_a = TupleValueExpression('a', 'table1')
        table2_col_a = TupleValueExpression('a', 'table2')
        select_list = [table1_col_a]
        from_table = TableRef(JoinNode(TableRef(TableInfo('table1')), TableRef(TableInfo('table2')), predicate=ComparisonExpression(ExpressionType.COMPARE_EQUAL, table1_col_a, table2_col_a), join_type=JoinType.INNER_JOIN))
        expected_stmt = SelectStatement(select_list, from_table)
        self.assertEqual(select_stmt, expected_stmt)

    def test_join_with_where(self):
        select_query = 'SELECT table1.a FROM table1 JOIN table2\n            ON table1.a = table2.a WHERE table1.a <= 5'
        parser = Parser()
        select_stmt = parser.parse(select_query)[0]
        table1_col_a = TupleValueExpression('a', 'table1')
        table2_col_a = TupleValueExpression('a', 'table2')
        select_list = [table1_col_a]
        from_table = TableRef(JoinNode(TableRef(TableInfo('table1')), TableRef(TableInfo('table2')), predicate=ComparisonExpression(ExpressionType.COMPARE_EQUAL, table1_col_a, table2_col_a), join_type=JoinType.INNER_JOIN))
        where_clause = ComparisonExpression(ExpressionType.COMPARE_LEQ, table1_col_a, ConstantValueExpression(5))
        expected_stmt = SelectStatement(select_list, from_table, where_clause)
        self.assertEqual(select_stmt, expected_stmt)

    def test_multiple_join_with_multiple_ON(self):
        select_query = 'SELECT table1.a FROM table1 JOIN table2\n            ON table1.a = table2.a JOIN table3\n            ON table3.a = table1.a WHERE table1.a <= 5'
        parser = Parser()
        select_stmt = parser.parse(select_query)[0]
        table1_col_a = TupleValueExpression('a', 'table1')
        table2_col_a = TupleValueExpression('a', 'table2')
        table3_col_a = TupleValueExpression('a', 'table3')
        select_list = [table1_col_a]
        child_join = TableRef(JoinNode(TableRef(TableInfo('table1')), TableRef(TableInfo('table2')), predicate=ComparisonExpression(ExpressionType.COMPARE_EQUAL, table1_col_a, table2_col_a), join_type=JoinType.INNER_JOIN))
        from_table = TableRef(JoinNode(child_join, TableRef(TableInfo('table3')), predicate=ComparisonExpression(ExpressionType.COMPARE_EQUAL, table3_col_a, table1_col_a), join_type=JoinType.INNER_JOIN))
        where_clause = ComparisonExpression(ExpressionType.COMPARE_LEQ, table1_col_a, ConstantValueExpression(5))
        expected_stmt = SelectStatement(select_list, from_table, where_clause)
        self.assertEqual(select_stmt, expected_stmt)

    def test_lateral_join(self):
        select_query = 'SELECT frame FROM MyVideo JOIN LATERAL\n                            ObjectDet(frame) AS OD;'
        parser = Parser()
        select_stmt = parser.parse(select_query)[0]
        tuple_frame = TupleValueExpression('frame')
        func_expr = FunctionExpression(func=None, name='ObjectDet', children=[tuple_frame])
        from_table = TableRef(JoinNode(TableRef(TableInfo('MyVideo')), TableRef(TableValuedExpression(func_expr), alias=Alias('OD')), join_type=JoinType.LATERAL_JOIN))
        expected_stmt = SelectStatement([tuple_frame], from_table)
        self.assertEqual(select_stmt, expected_stmt)

    def test_class_equality(self):
        table_info = TableInfo('MyVideo')
        table_ref = TableRef(TableInfo('MyVideo'))
        tuple_frame = TupleValueExpression('frame')
        func_expr = FunctionExpression(func=None, name='ObjectDet', children=[tuple_frame])
        join_node = JoinNode(TableRef(TableInfo('MyVideo')), TableRef(TableValuedExpression(func_expr), alias=Alias('OD')), join_type=JoinType.LATERAL_JOIN)
        self.assertNotEqual(table_info, table_ref)
        self.assertNotEqual(tuple_frame, table_ref)
        self.assertNotEqual(join_node, table_ref)
        self.assertNotEqual(table_ref, table_info)

    def test_create_job(self):
        queries = ["CREATE OR REPLACE FUNCTION HomeSalesForecast FROM\n                ( SELECT * FROM postgres_data.home_sales )\n                TYPE Forecasting\n                PREDICT 'price';", 'Select HomeSalesForecast(10);']
        job_query = f"CREATE JOB my_job AS {{\n            {''.join(queries)}\n        }}\n        START '2023-04-01'\n        END '2023-05-01'\n        EVERY 2 hour\n        "
        parser = Parser()
        job_stmt = parser.parse(job_query)[0]
        self.assertEqual(job_stmt.job_name, 'my_job')
        self.assertEqual(len(job_stmt.queries), 2)
        self.assertTrue(queries[0].rstrip(';') == str(job_stmt.queries[0]))
        self.assertTrue(queries[1].rstrip(';') == str(job_stmt.queries[1]))
        self.assertEqual(job_stmt.start_time, '2023-04-01')
        self.assertEqual(job_stmt.end_time, '2023-05-01')
        self.assertEqual(job_stmt.repeat_interval, 2)
        self.assertEqual(job_stmt.repeat_period, 'hour')

def overlap(x, y):
    overlap = (max(x[0], y[0]), min(x[1], y[1]))
    if overlap[0] <= overlap[1]:
        return overlap

def union(ranges: List):
    reduced_list = []
    for begin, end in sorted(ranges):
        if reduced_list and reduced_list[-1][1] >= begin - 1:
            reduced_list[-1] = (reduced_list[-1][0], max(reduced_list[-1][1], end))
        else:
            reduced_list.append((begin, end))
    return reduced_list

class Functions:

    def function(self, tree):
        function_name = None
        function_output = None
        function_args = []
        for child in tree.children:
            if isinstance(child, Token):
                if child.value == '*':
                    function_args = [TupleValueExpression(name='*')]
            if isinstance(child, Tree):
                if child.data == 'simple_id':
                    function_name = self.visit(child)
                elif child.data == 'dotted_id':
                    function_output = self.visit(child)
                elif child.data == 'function_args':
                    function_args = self.visit(child)
        func_expr = FunctionExpression(None, name=function_name, output=function_output)
        for arg in function_args:
            func_expr.append_child(arg)
        return func_expr

    def function_args(self, tree):
        args = []
        for child in tree.children:
            if isinstance(child, Tree):
                args.append(self.visit(child))
        return args

    def create_function(self, tree):
        function_name = None
        or_replace = False
        if_not_exists = False
        input_definitions = []
        output_definitions = []
        impl_path = None
        function_type = None
        query = None
        metadata = []
        create_definitions_index = 0
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'function_name':
                    function_name = self.visit(child)
                elif child.data == 'or_replace':
                    or_replace = True
                elif child.data == 'if_not_exists':
                    if_not_exists = True
                elif child.data == 'create_definitions':
                    if create_definitions_index == 0:
                        input_definitions = self.visit(child)
                        create_definitions_index += 1
                    elif create_definitions_index == 1:
                        output_definitions = self.visit(child)
                elif child.data == 'function_type':
                    function_type = self.visit(child)
                elif child.data == 'function_impl':
                    impl_path = self.visit(child).value
                elif child.data == 'simple_select':
                    query = self.visit(child)
                elif child.data == 'function_metadata':
                    key_value_pair = self.visit(child)
                    value = key_value_pair[1]
                    if isinstance(value, ConstantValueExpression):
                        value = value.value
                    (metadata.append((key_value_pair[0].lower(), value)),)
        return CreateFunctionStatement(function_name, or_replace, if_not_exists, impl_path, input_definitions, output_definitions, function_type, query, metadata)

    def get_aggregate_function_type(self, agg_func_name):
        agg_func_type = None
        if agg_func_name == 'COUNT':
            agg_func_type = ExpressionType.AGGREGATION_COUNT
        elif agg_func_name == 'MIN':
            agg_func_type = ExpressionType.AGGREGATION_MIN
        elif agg_func_name == 'MAX':
            agg_func_type = ExpressionType.AGGREGATION_MAX
        elif agg_func_name == 'SUM':
            agg_func_type = ExpressionType.AGGREGATION_SUM
        elif agg_func_name == 'AVG':
            agg_func_type = ExpressionType.AGGREGATION_AVG
        elif agg_func_name == 'FIRST':
            agg_func_type = ExpressionType.AGGREGATION_FIRST
        elif agg_func_name == 'LAST':
            agg_func_type = ExpressionType.AGGREGATION_LAST
        elif agg_func_name == 'SEGMENT':
            agg_func_type = ExpressionType.AGGREGATION_SEGMENT
        return agg_func_type

    def aggregate_windowed_function(self, tree):
        agg_func_arg = None
        agg_func_name = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'function_arg':
                    agg_func_arg = self.visit(child)
                elif child.data == 'aggregate_function_name':
                    agg_func_name = self.visit(child).value
            elif isinstance(child, Token):
                token = child.value
                if token != '*':
                    agg_func_name = token
                elif token == '*':
                    agg_func_arg = TupleValueExpression(name='_row_id')
                else:
                    agg_func_arg = TupleValueExpression(name='id')
        agg_func_type = self.get_aggregate_function_type(agg_func_name)
        agg_expr = AggregationExpression(agg_func_type, None, agg_func_arg)
        return agg_expr

class Expressions:

    def string_literal(self, tree):
        text = tree.children[0]
        assert text is not None
        return ConstantValueExpression(text[1:-1], ColumnType.TEXT)

    def array_literal(self, tree):
        array_elements = []
        for child in tree.children:
            if isinstance(child, Tree):
                array_element = self.visit(child).value
                array_elements.append(array_element)
        res = ConstantValueExpression(np.array(array_elements), ColumnType.NDARRAY)
        return res

    def boolean_literal(self, tree):
        text = tree.children[0]
        if text == 'TRUE':
            return ConstantValueExpression(True, ColumnType.BOOLEAN)
        return ConstantValueExpression(False, ColumnType.BOOLEAN)

    def constant(self, tree):
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'real_literal':
                    real_literal = self.visit(child)
                    return ConstantValueExpression(real_literal, ColumnType.FLOAT)
                elif child.data == 'decimal_literal':
                    decimal_literal = self.visit(child)
                    return ConstantValueExpression(decimal_literal, ColumnType.INTEGER)
        return self.visit_children(tree)

    def logical_expression(self, tree):
        left = self.visit(tree.children[0])
        op = self.visit(tree.children[1])
        right = self.visit(tree.children[2])
        return LogicalExpression(op, left, right)

    def binary_comparison_predicate(self, tree):
        left = self.visit(tree.children[0])
        op = self.visit(tree.children[1])
        right = self.visit(tree.children[2])
        return ComparisonExpression(op, left, right)

    def nested_expression_atom(self, tree):
        expr = tree.children[0]
        return self.visit(expr)

    def comparison_operator(self, tree):
        op = str(tree.children[0])
        if op == '=':
            return ExpressionType.COMPARE_EQUAL
        elif op == '<':
            return ExpressionType.COMPARE_LESSER
        elif op == '>':
            return ExpressionType.COMPARE_GREATER
        elif op == '>=':
            return ExpressionType.COMPARE_GEQ
        elif op == '<=':
            return ExpressionType.COMPARE_LEQ
        elif op == '!=':
            return ExpressionType.COMPARE_NEQ
        elif op == '@>':
            return ExpressionType.COMPARE_CONTAINS
        elif op == '<@':
            return ExpressionType.COMPARE_IS_CONTAINED
        elif op == 'LIKE':
            return ExpressionType.COMPARE_LIKE

    def logical_operator(self, tree):
        op = str(tree.children[0])
        if string_comparison_case_insensitive(op, 'OR'):
            return ExpressionType.LOGICAL_OR
        elif string_comparison_case_insensitive(op, 'AND'):
            return ExpressionType.LOGICAL_AND
        else:
            raise NotImplementedError('Unsupported logical operator: {}'.format(op))

    def expressions_with_defaults(self, tree):
        expr_list = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'expression_or_default':
                    expression = self.visit(child)
                    expr_list.append(expression)
        return expr_list

    def sample_params(self, tree):
        sample_type = None
        sample_freq = None
        for child in tree.children:
            if child.data == 'sample_clause':
                sample_freq = self.visit(child)
            elif child.data == 'sample_clause_with_type':
                sample_type, sample_freq = self.visit(child)
        return (sample_type, sample_freq)

    def sample_clause(self, tree):
        sample_list = self.visit_children(tree)
        assert len(sample_list) == 2
        return ConstantValueExpression(sample_list[1])

    def sample_clause_with_type(self, tree):
        sample_list = self.visit_children(tree)
        assert len(sample_list) == 3 or len(sample_list) == 2
        if len(sample_list) == 3:
            return (ConstantValueExpression(sample_list[1]), ConstantValueExpression(sample_list[2]))
        else:
            return (ConstantValueExpression(sample_list[1]), ConstantValueExpression(1))

    def chunk_params(self, tree):
        chunk_params = self.visit_children(tree)
        assert len(chunk_params) == 2 or len(chunk_params) == 4
        if len(chunk_params) == 4:
            return {'chunk_size': chunk_params[1], 'chunk_overlap': chunk_params[3]}
        elif len(chunk_params) == 2:
            if chunk_params[0] == 'CHUNK_SIZE':
                return {'chunk_size': chunk_params[1]}
            elif chunk_params[0] == 'CHUNK_OVERLAP':
                return {'chunk_overlap': chunk_params[1]}
            else:
                assert f'incorrect keyword found {chunk_params[0]}'

    def colon_param_dict(self, tree):
        param_dict = {}
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'colon_param':
                    param = self.visit(child)
                    key = param[0].value
                    value = param[1].value
                    param_dict[key] = value
        return param_dict

class TableSources:

    def select_elements(self, tree):
        kind = tree.children[0]
        if kind == '*':
            select_list = [TupleValueExpression(name='*')]
        else:
            select_list = []
            for child in tree.children:
                element = self.visit(child)
                select_list.append(element)
        return select_list

    def table_sources(self, tree):
        return self.visit(tree.children[0])

    def table_source(self, tree):
        left_node = None
        join_nodes = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_source_item_with_param':
                    left_node = self.visit(child)
                    join_nodes = [left_node]
                elif child.data.endswith('join'):
                    table = self.visit(child)
                    join_nodes.append(table)
        num_table_joins = len(join_nodes)
        if num_table_joins > 1:
            for i in range(num_table_joins - 1):
                join_nodes[i + 1].join_node.left = join_nodes[i]
            return join_nodes[-1]
        else:
            return join_nodes[0]

    def table_source_item_with_param(self, tree):
        sample_freq = None
        sample_type = None
        alias = None
        table = None
        chunk_params = {}
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_source_item':
                    table = self.visit(child)
                elif child.data == 'sample_params':
                    sample_type, sample_freq = self.visit(child)
                elif child.data == 'chunk_params':
                    chunk_params = self.visit(child)
                elif child.data == 'alias_clause':
                    alias = self.visit(child)
        return TableRef(table=table, alias=alias, sample_freq=sample_freq, sample_type=sample_type, chunk_params=chunk_params)

    def table_source_item(self, tree):
        return self.visit(tree.children[0])

    def query_specification(self, tree):
        target_list = None
        from_clause = None
        where_clause = None
        groupby_clause = None
        orderby_clause = None
        limit_count = None
        for child in tree.children[1:]:
            try:
                if child.data == 'select_elements':
                    target_list = self.visit(child)
                elif child.data == 'from_clause':
                    clause = self.visit(child)
                    from_clause = clause.get('from', None)
                    where_clause = clause.get('where', None)
                    groupby_clause = clause.get('groupby', None)
                elif child.data == 'order_by_clause':
                    orderby_clause = self.visit(child)
                elif child.data == 'limit_clause':
                    limit_count = self.visit(child)
            except BaseException as e:
                logger.error('Error while parsing                                 QuerySpecification')
                raise e
        select_stmt = SelectStatement(target_list, from_clause, where_clause, groupby_clause=groupby_clause, orderby_list=orderby_clause, limit_count=limit_count)
        return select_stmt

    def from_clause(self, tree):
        from_table = None
        where_clause = None
        groupby_clause = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_sources':
                    from_table = self.visit(child)
                elif child.data == 'where_expr':
                    where_clause = self.visit(child)
                elif child.data == 'group_by_item':
                    groupby_item = self.visit(child)
                    groupby_clause = groupby_item
        return {'from': from_table, 'where': where_clause, 'groupby': groupby_clause}

    def inner_join(self, tree):
        table = None
        join_predicate = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_source_item_with_param':
                    table = self.visit(child)
                elif child.data.endswith('expression'):
                    join_predicate = self.visit(child)
        return TableRef(JoinNode(None, table, predicate=join_predicate, join_type=JoinType.INNER_JOIN))

    def lateral_join(self, tree):
        tve = None
        alias = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_valued_function':
                    tve = self.visit(child)
                elif child.data == 'alias_clause':
                    alias = self.visit(child)
        if alias is None:
            err_msg = f'TableValuedFunction {tve.func_expr.name} should have alias.'
            logger.error(err_msg)
            raise SyntaxError(err_msg)
        join_type = JoinType.LATERAL_JOIN
        return TableRef(JoinNode(None, TableRef(tve, alias=alias), join_type=join_type))

    def table_valued_function(self, tree):
        func_expr = None
        has_unnest = False
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data.endswith('function_call'):
                    func_expr = self.visit(child)
            elif child.lower() == 'unnest':
                has_unnest = True
        return TableValuedExpression(func_expr, do_unnest=has_unnest)

    def subquery_table_source_item(self, tree):
        subquery_table_source_item = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'simple_select':
                    subquery_table_source_item = self.visit(child)
        return subquery_table_source_item

    def union_select(self, tree):
        right_select_statement = None
        union_all = False
        statement_id = 0
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data.endswith('select'):
                    if statement_id == 0:
                        left_select_statement = self.visit(child)
                    elif statement_id == 1:
                        right_select_statement = self.visit(child)
                    statement_id += 1
            elif isinstance(child, Token):
                if child.value == 'ALL':
                    union_all = True
        if left_select_statement is not None:
            assert left_select_statement.union_link is None, 'Checking for the correctness of the operator'
            left_select_statement.union_link = right_select_statement
            if union_all is False:
                left_select_statement.union_all = False
            else:
                left_select_statement.union_all = True
        return left_select_statement

    def group_by_item(self, tree):
        expr = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data.endswith('expression'):
                    expr = self.visit(child)
        return expr

    def alias_clause(self, tree):
        alias_name = None
        column_list = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'uid':
                    alias_name = self.visit(child)
                elif child.data == 'uid_list':
                    column_list = self.visit(child)
                    column_list = [col.name for col in column_list]
        return Alias(alias_name, column_list)

class Select:

    def simple_select(self, tree):
        select_stmt = self.visit_children(tree)
        return select_stmt

    def order_by_clause(self, tree):
        orderby_clause_data = []
        for child in tree.children:
            if isinstance(child, Tree):
                orderby_clause_data.append(self.visit(child))
        return orderby_clause_data

    def order_by_expression(self, tree):
        expr = None
        sort_order = ParserOrderBySortType.ASC
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data.endswith('expression'):
                    expr = self.visit(child)
                elif child.data == 'sort_order':
                    sort_order = self.visit(child)
        return (expr, sort_order)

    def sort_order(self, tree):
        token = tree.children[0]
        sort_order = None
        if str.upper(token) == 'ASC':
            sort_order = ParserOrderBySortType.ASC
        elif str.upper(token) == 'DESC':
            sort_order = ParserOrderBySortType.DESC
        return sort_order

    def limit_clause(self, tree):
        output = ConstantValueExpression(self.visit(tree.children[1]))
        return output

class Insert:

    def insert_statement(self, tree):
        table_ref = None
        column_list = []
        value_list = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_name':
                    table_name = self.visit(child)
                    table_ref = TableRef(table_name)
                elif child.data == 'uid_list':
                    column_list = self.visit(child)
                elif child.data == 'insert_statement_value':
                    value_list = self.visit(child)
        insert_stmt = InsertTableStatement(table_ref, column_list, value_list)
        return insert_stmt

    def uid_list(self, tree):
        uid_expr_list = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'uid':
                    uid = self.visit(child)
                    uid_expr = TupleValueExpression(uid)
                    uid_expr_list.append(uid_expr)
        return uid_expr_list

    def insert_statement_value(self, tree):
        insert_stmt_value = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'expressions_with_defaults':
                    expr = self.visit(child)
                    insert_stmt_value.append(expr)
        return insert_stmt_value

class CommonClauses:

    def table_name(self, tree):
        child = self.visit(tree.children[0])
        if isinstance(child, tuple):
            database_name, table_name = (child[0], child[1])
        else:
            database_name, table_name = (None, child)
        if table_name is not None:
            return TableInfo(table_name=table_name, database_name=database_name)
        else:
            error = 'Invalid Table Name'
            logger.error(error)

    def full_id(self, tree):
        if len(tree.children) == 1:
            return self.visit(tree.children[0])
        elif len(tree.children) == 2:
            return (self.visit(tree.children[0]), self.visit(tree.children[1]))

    def uid(self, tree):
        if hasattr(tree.children[0], 'type') and tree.children[0].type == 'REVERSE_QUOTE_ID':
            tree.children[0].type = 'simple_id'
            non_tick_string = str(tree.children[0]).replace('`', '')
            return non_tick_string
        return self.visit(tree.children[0])

    def full_column_name(self, tree):
        uid = self.visit(tree.children[0])
        if len(tree.children) > 1:
            dotted_id = self.visit(tree.children[1])
            return TupleValueExpression(table_alias=uid, name=dotted_id)
        else:
            return TupleValueExpression(name=uid)

    def dotted_id(self, tree):
        dotted_id = str(tree.children[0])
        dotted_id = dotted_id.lstrip('.')
        return dotted_id

    def simple_id(self, tree):
        simple_id = str(tree.children[0])
        return simple_id

    def decimal_literal(self, tree):
        decimal = None
        token = tree.children[0]
        if str.upper(token) == 'ANYDIM':
            decimal = Dimension.ANYDIM
        else:
            decimal = int(str(token))
        return decimal

    def real_literal(self, tree):
        real_literal = float(tree.children[0])
        return real_literal

class DecordReader(AbstractReader):

    def __init__(self, *args, predicate: AbstractExpression=None, sampling_rate: int=None, sampling_type: str=None, read_audio: bool=False, read_video: bool=True, **kwargs):
        """Read frames from the disk

        Args:
            predicate (AbstractExpression, optional): If only subset of frames
            need to be read. The predicate should be only on single column and
            can be converted to ranges. Defaults to None.
            sampling_rate (int, optional): Set if the caller wants one frame
            every `sampling_rate` number of frames. For example, if `sampling_rate = 10`, it returns every 10th frame. If both `predicate` and `sampling_rate` are specified, `sampling_rate` is given precedence.
            sampling_type (str, optional): Set as IFRAMES if caller want to sample on top on iframes only. e.g if the IFRAME frame numbers are [10,20,30,40,50] then 'SAMPLE IFRAMES 2' will return [10,30,50]
            read_audio (bool, optional): Whether to read audio stream from the video. Defaults to False
            read_video (bool, optional): Whether to read video stream from the video. Defaults to True
        """
        self._predicate = predicate
        self._sampling_rate = sampling_rate or 1
        self._sampling_type = sampling_type
        self._read_audio = read_audio
        self._read_video = read_video
        self._reader = None
        self._get_frame = None
        super().__init__(*args, **kwargs)
        self.initialize_reader()

    def _read(self) -> Iterator[Dict]:
        num_frames = int(len(self._reader))
        if self._predicate:
            range_list = extract_range_list_from_predicate(self._predicate, 0, num_frames - 1)
        else:
            range_list = [(0, num_frames - 1)]
        logger.debug('Reading frames')
        if self._sampling_type == IFRAMES:
            iframes = self._reader.get_key_indices()
            idx = 0
            for begin, end in range_list:
                while idx < len(iframes) and iframes[idx] < begin:
                    idx += self._sampling_rate
                while idx < len(iframes) and iframes[idx] <= end:
                    frame_id = iframes[idx]
                    idx += self._sampling_rate
                    yield self._get_frame(frame_id)
        elif self._sampling_rate == 1 or self._read_audio:
            for begin, end in range_list:
                frame_id = begin
                while frame_id <= end:
                    yield self._get_frame(frame_id)
                    frame_id += 1
        else:
            for begin, end in range_list:
                if begin % self._sampling_rate:
                    begin += self._sampling_rate - begin % self._sampling_rate
                for frame_id in range(begin, end + 1, self._sampling_rate):
                    yield self._get_frame(frame_id)

    def initialize_reader(self):
        try_to_import_decord()
        import decord
        if self._read_audio:
            assert self._sampling_type != IFRAMES, 'Cannot use IFRAMES with audio streams'
            sample_rate = 16000
            if self._sampling_type == AUDIORATE and self._sampling_rate != 1:
                sample_rate = self._sampling_rate
            try:
                self._reader = decord.AVReader(self.file_url, mono=True, sample_rate=sample_rate)
                self._get_frame = self.__get_audio_frame
            except decord._ffi.base.DECORDError as error_msg:
                assert "Can't find audio stream" not in str(error_msg), error_msg
        else:
            assert self._sampling_type != AUDIORATE, 'Cannot use AUDIORATE with video streams'
            self._reader = decord.VideoReader(self.file_url)
            self._get_frame = self.__get_video_frame

    def __get_video_frame(self, frame_id):
        frame_video = self._reader[frame_id]
        frame_video = frame_video.asnumpy()
        timestamp = self._reader.get_frame_timestamp(frame_id)[0]
        return {VideoColumnName.id.name: frame_id, ROW_NUM_COLUMN: frame_id, VideoColumnName.data.name: frame_video, VideoColumnName.seconds.name: round(timestamp, 2)}

    def __get_audio_frame(self, frame_id):
        frame_audio, _ = self._reader[frame_id]
        frame_audio = frame_audio.asnumpy()[0]
        return {VideoColumnName.id.name: frame_id, ROW_NUM_COLUMN: frame_id, VideoColumnName.data.name: np.empty(0), VideoColumnName.seconds.name: 0.0, VideoColumnName.audio.name: frame_audio}

