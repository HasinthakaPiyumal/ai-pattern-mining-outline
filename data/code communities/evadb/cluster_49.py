# Cluster 49

class MemoTest(unittest.TestCase):

    def test_memo_add_with_no_id(self):
        group_expr = MagicMock()
        group_expr.group_id = UNDEFINED_GROUP_ID
        memo = Memo()
        memo.add_group_expr(group_expr)
        self.assertEqual(0, group_expr.group_id)

    def test_memo_add_with_forcing_id(self):
        group_expr = MagicMock()
        group_expr.group_id = 0
        memo = Memo()
        self.assertEqual(memo.add_group_expr(group_expr), group_expr)
        self.assertEqual(len(memo.groups), 1)

    def test_memo_add_under_existing_group(self):
        group_expr1 = MagicMock()
        group_expr1.group_id = UNDEFINED_GROUP_ID
        group_expr2 = MagicMock()
        group_expr2.group_id = 0
        memo = Memo()
        expr = memo.add_group_expr(group_expr1)
        ret_expr = memo.add_group_expr(group_expr2)
        self.assertEqual(expr.group_id, 0)
        self.assertEqual(ret_expr.group_id, 1)
        self.assertEqual(len(memo.groups), 2)
        self.assertEqual(len(memo.group_exprs), 2)
        memo = Memo()
        memo.add_group_expr(group_expr2)
        expr = memo.add_group_expr(group_expr1)
        self.assertEqual(expr.group_id, 1)
        self.assertEqual(len(memo.groups), 2)
        self.assertEqual(len(memo.group_exprs), 2)

class OptimizerContext:
    """
    Maintain context information for the optimizer

    Arguments:
        _task_queue(OptimizerTaskStack):
            stack to keep track outstanding tasks
    """

    def __init__(self, db: EvaDBDatabase, cost_model: CostModel, rules_manager: RulesManager=None):
        self._db = db
        self._task_stack = OptimizerTaskStack()
        self._memo = Memo()
        self._cost_model = cost_model
        is_ray_enabled = self.db.catalog().get_configuration_catalog_value('ray')
        self._rules_manager = rules_manager or RulesManager({'ray': is_ray_enabled})

    @property
    def db(self):
        return self._db

    @property
    def rules_manager(self):
        return self._rules_manager

    @property
    def cost_model(self):
        return self._cost_model

    @property
    def task_stack(self):
        return self._task_stack

    @property
    def memo(self):
        return self._memo

    def _xform_opr_to_group_expr(self, opr: Operator) -> GroupExpression:
        """
        Note: Internal function Generate a group expressions from a
        logical operator tree. Caller is responsible for assigning
        the group to the returned GroupExpression.
        """
        child_ids = []
        for child_opr in opr.children:
            if isinstance(child_opr, Dummy):
                child_ids.append(child_opr.group_id)
            else:
                child_expr = self._xform_opr_to_group_expr(opr=child_opr)
                memo_expr = self.memo.add_group_expr(child_expr)
                child_ids.append(memo_expr.group_id)
        opr_copy = copy.copy(opr)
        opr_copy.clear_children()
        expr = GroupExpression(opr=opr_copy, children=child_ids)
        return expr

    def replace_expression(self, opr: Operator, group_id: int):
        """
        Removes all the expressions from the specified group and
        create a new expression. This is called by rewrite rules. The
        new expr gets assigned a new group id
        """
        self.memo.erase_group(group_id)
        new_expr = self._xform_opr_to_group_expr(opr)
        new_expr = self.memo.add_group_expr(new_expr, group_id)
        return new_expr

    def add_opr_to_group(self, opr: Operator, group_id: int=UNDEFINED_GROUP_ID):
        """
        Convert operator to group_expression and add to the group
        """
        grp_expr = self._xform_opr_to_group_expr(opr)
        grp_expr = self.memo.add_group_expr(grp_expr, group_id)
        return grp_expr

