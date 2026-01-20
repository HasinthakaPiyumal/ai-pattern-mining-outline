# Cluster 51

class TestBinder(unittest.TestCase):

    def helper_pre_order_match(self, cur_opr, res_opr):
        self.assertEqual(cur_opr.opr_type, res_opr.opr_type)
        self.assertEqual(len(cur_opr.children), len(res_opr.children))
        for i, child_opr in enumerate(cur_opr.children):
            self.helper_pre_order_match(child_opr, res_opr.children[i])

    def test_simple_binder_match(self):
        """
        Opr Tree:
                         LogicalFilter
                         /                             LogicalGet      LogicalGet

        Pattern:
                         LogicalFilter
                         /                             LogicalGet      LogicalGet
        """
        child1_opr = LogicalGet(MagicMock(), MagicMock(), MagicMock())
        child2_opr = LogicalGet(MagicMock(), MagicMock(), MagicMock())
        root_opr = LogicalFilter(MagicMock(), [child1_opr, child2_opr])
        child1_ptn = Pattern(OperatorType.LOGICALGET)
        child2_ptn = Pattern(OperatorType.LOGICALGET)
        root_ptn = Pattern(OperatorType.LOGICALFILTER)
        root_ptn.append_child(child1_ptn)
        root_ptn.append_child(child2_ptn)
        opt_ctxt = OptimizerContext(MagicMock(), CostModel())
        root_grp_expr = opt_ctxt.add_opr_to_group(root_opr)
        binder = Binder(root_grp_expr, root_ptn, opt_ctxt.memo)
        for match in iter(binder):
            self.helper_pre_order_match(root_opr, match)

    def test_nested_binder_match(self):
        """
        Opr Tree:
                         LogicalFilter
                         /                             LogicalGet      LogicalFilter
                                  /                                       LogicalGet       LogicalGet

        Pattern:
                         LogicalFilter
                         /                             LogicalGet      Dummy
        """
        sub_child_opr = LogicalGet(MagicMock(), MagicMock(), MagicMock())
        sub_child_opr_2 = LogicalGet(MagicMock(), MagicMock(), MagicMock())
        sub_root_opr = LogicalFilter(MagicMock(), [sub_child_opr, sub_child_opr_2])
        child_opr = LogicalGet(MagicMock(), MagicMock(), MagicMock())
        root_opr = LogicalFilter(MagicMock(), [child_opr, sub_root_opr])
        child_ptn = Pattern(OperatorType.LOGICALGET)
        root_ptn = Pattern(OperatorType.LOGICALFILTER)
        root_ptn.append_child(child_ptn)
        root_ptn.append_child(Pattern(OperatorType.DUMMY))
        opt_ctxt = OptimizerContext(MagicMock(), CostModel())
        root_grp_expr = opt_ctxt.add_opr_to_group(root_opr)
        binder = Binder(root_grp_expr, root_ptn, opt_ctxt.memo)
        expected_match = copy.copy(root_opr)
        expected_match.children = [child_opr, Dummy(2, None)]
        for match in iter(binder):
            self.helper_pre_order_match(expected_match, match)
        opt_ctxt = OptimizerContext(MagicMock(), CostModel())
        sub_root_grp_expr = opt_ctxt.add_opr_to_group(sub_root_opr)
        expected_match = copy.copy(sub_root_opr)
        expected_match.children = [sub_child_opr, Dummy(1, None)]
        binder = Binder(sub_root_grp_expr, root_ptn, opt_ctxt.memo)
        for match in iter(binder):
            self.helper_pre_order_match(expected_match, match)

class TestGroup(unittest.TestCase):

    def test_simple_add_group_expr(self):
        grp = Group(0)
        grp_expr1 = GroupExpression(MagicMock())
        grp_expr1.opr.is_logical = lambda: True
        grp_expr2 = GroupExpression(MagicMock())
        grp_expr2.opr.is_logical = lambda: False
        grp_expr3 = GroupExpression(MagicMock(), 0)
        grp_expr3.opr.is_logical = lambda: True
        grp.add_expr(grp_expr1)
        self.assertEquals(len(grp.logical_exprs), 1)
        grp.add_expr(grp_expr2)
        self.assertEquals(len(grp.logical_exprs), 1)
        self.assertEquals(len(grp.physical_exprs), 1)
        grp.add_expr(grp_expr3)
        self.assertEquals(len(grp.logical_exprs), 2)
        self.assertEquals(len(grp.physical_exprs), 1)

    def test_add_group_expr_with_unmatched_group_id(self):
        grp = Group(0)
        grp_expr1 = GroupExpression(MagicMock(), 1)
        grp_expr1.opr.is_logical = lambda: True
        grp.add_expr(grp_expr1)
        self.assertEquals(len(grp.logical_exprs), 0)
        self.assertEquals(len(grp.physical_exprs), 0)

    def test_add_group_expr_cost(self):
        grp = Group(0)
        prpty = Property(PropertyType(1))
        grp_expr1 = GroupExpression(MagicMock(), 1)
        grp_expr1.opr.is_logical = lambda: True
        grp_expr2 = GroupExpression(MagicMock())
        grp_expr2.opr.is_logical = lambda: False
        grp.add_expr(grp_expr1)
        grp.add_expr_cost(grp_expr1, prpty, 1)
        grp.add_expr(grp_expr2)
        grp.add_expr_cost(grp_expr2, prpty, 0)
        self.assertEqual(grp.get_best_expr(prpty), grp_expr2)
        self.assertEqual(grp.get_best_expr_cost(prpty), 0)

    def test_empty_group_expr(self):
        grp = Group(0)
        prpty = Property(PropertyType(1))
        self.assertEqual(grp.get_best_expr(prpty), None)
        self.assertEqual(grp.get_best_expr_cost(prpty), None)

class CostModel(unittest.TestCase):

    def execute_task_stack(self, task_stack):
        while not task_stack.empty():
            task = task_stack.pop()
            task.execute()

    def test_should_select_cheap_plan(self):

        def side_effect_func(value):
            if value is grp_expr1:
                return 1
            elif value is grp_expr2:
                return 2
        cm = CostModel()
        cm.calculate_cost = MagicMock(side_effect=side_effect_func)
        opt_cxt = OptimizerContext(MagicMock(), cm)
        grp_expr1 = GroupExpression(MagicMock())
        grp_expr1.opr.is_logical = lambda: False
        grp_expr2 = GroupExpression(MagicMock())
        grp_expr2.opr.is_logical = lambda: False
        opt_cxt.memo.add_group_expr(grp_expr1)
        opt_cxt.memo.add_group_expr(grp_expr2, grp_expr1.group_id)
        grp = opt_cxt.memo.get_group_by_id(grp_expr1.group_id)
        opt_cxt.task_stack.push(OptimizeGroup(grp, opt_cxt))
        self.execute_task_stack(opt_cxt.task_stack)
        plan = PlanGenerator(MagicMock()).build_optimal_physical_plan(grp_expr1.group_id, opt_cxt)
        self.assertEqual(plan, grp_expr1.opr)
        self.assertEqual(grp.get_best_expr_cost(PropertyType.DEFAULT), 1)

    def test_should_select_cheap_plan_with_tree(self):

        def side_effect_func(value):
            cost = dict({grp_expr00: 1, grp_expr01: 2, grp_expr10: 4, grp_expr11: 3, grp_expr20: 5})
            return cost[value]
        cm = CostModel()
        cm.calculate_cost = MagicMock(side_effect=side_effect_func)
        opt_cxt = OptimizerContext(MagicMock(), cm)
        grp_expr00 = GroupExpression(Operator(MagicMock()))
        grp_expr00.opr.is_logical = lambda: False
        grp_expr01 = GroupExpression(Operator(MagicMock()))
        grp_expr01.opr.is_logical = lambda: False
        opt_cxt.memo.add_group_expr(grp_expr00)
        opt_cxt.memo.add_group_expr(grp_expr01, grp_expr00.group_id)
        grp_expr10 = GroupExpression(Operator(MagicMock()))
        grp_expr10.opr.is_logical = lambda: False
        opt_cxt.memo.add_group_expr(grp_expr10)
        grp_expr11 = GroupExpression(Operator(MagicMock()))
        grp_expr11.opr.is_logical = lambda: False
        opt_cxt.memo.add_group_expr(grp_expr11, grp_expr10.group_id)
        grp_expr20 = GroupExpression(Operator(MagicMock()))
        grp_expr20.opr.is_logical = lambda: False
        opt_cxt.memo.add_group_expr(grp_expr20)
        grp = opt_cxt.memo.get_group_by_id(grp_expr20.group_id)
        grp_expr10.children = [grp_expr01.group_id]
        grp_expr11.children = [grp_expr01.group_id]
        grp_expr20.children = [grp_expr10.group_id]
        opt_cxt.task_stack.push(OptimizeGroup(grp, opt_cxt))
        self.execute_task_stack(opt_cxt.task_stack)
        plan = PlanGenerator(MagicMock()).build_optimal_physical_plan(grp_expr20.group_id, opt_cxt)
        subplan = copy(grp_expr11.opr)
        subplan.children = [copy(grp_expr01.opr)]
        expected_plan = copy(grp_expr20.opr)
        expected_plan.children = [subplan]
        self.assertEqual(plan, expected_plan)
        self.assertEqual(grp.get_best_expr_cost(PropertyType.DEFAULT), 9)

class TestOptimizerTask(unittest.TestCase):

    def execute_task_stack(self, task_stack):
        while not task_stack.empty():
            task = task_stack.pop()
            task.execute()

    def test_abstract_optimizer_task(self):
        task = OptimizerTask(MagicMock(), MagicMock())
        with self.assertRaises(NotImplementedError):
            task.execute()

    def top_down_rewrite(self, opr):
        opt_cxt = OptimizerContext(MagicMock(), CostModel(), RulesManager())
        grp_expr = opt_cxt.add_opr_to_group(opr)
        root_grp_id = grp_expr.group_id
        opt_cxt.task_stack.push(TopDownRewrite(grp_expr, RulesManager().stage_one_rewrite_rules, opt_cxt))
        self.execute_task_stack(opt_cxt.task_stack)
        return (opt_cxt, root_grp_id)

    def bottom_up_rewrite(self, root_grp_id, opt_cxt):
        grp_expr = opt_cxt.memo.groups[root_grp_id].logical_exprs[0]
        opt_cxt.task_stack.push(BottomUpRewrite(grp_expr, RulesManager().stage_two_rewrite_rules, opt_cxt))
        self.execute_task_stack(opt_cxt.task_stack)
        return (opt_cxt, root_grp_id)

    def implement_group(self, root_grp_id, opt_cxt):
        grp = opt_cxt.memo.groups[root_grp_id]
        opt_cxt.task_stack.push(OptimizeGroup(grp, opt_cxt))
        self.execute_task_stack(opt_cxt.task_stack)
        return (opt_cxt, root_grp_id)

    def test_simple_implementation(self):
        predicate = MagicMock()
        child_opr = LogicalGet(MagicMock(), MagicMock(), MagicMock())
        root_opr = LogicalFilter(predicate, [child_opr])
        opt_cxt, root_grp_id = self.top_down_rewrite(root_opr)
        opt_cxt, root_grp_id = self.bottom_up_rewrite(root_grp_id, opt_cxt)
        opt_cxt, root_grp_id = self.implement_group(root_grp_id, opt_cxt)
        root_grp = opt_cxt.memo.groups[root_grp_id]
        best_root_grp_expr = root_grp.get_best_expr(PropertyType.DEFAULT)
        self.assertEqual(type(best_root_grp_expr.opr), PredicatePlan)

    def test_nested_implementation(self):
        child_predicate = MagicMock()
        root_predicate = MagicMock()
        with patch('evadb.optimizer.rules.rules.extract_pushdown_predicate') as mock:
            with patch('evadb.optimizer.rules.rules.is_video_table') as mock_vid:
                mock_vid.return_value = True
                mock.side_effect = [(child_predicate, None), (root_predicate, None)]
                child_get_opr = LogicalGet(MagicMock(), MagicMock(), MagicMock())
                child_filter_opr = LogicalFilter(child_predicate, children=[child_get_opr])
                child_project_opr = LogicalProject([MagicMock()], children=[child_filter_opr])
                root_derived_get_opr = LogicalQueryDerivedGet(MagicMock(), children=[child_project_opr])
                root_filter_opr = LogicalFilter(root_predicate, children=[root_derived_get_opr])
                root_project_opr = LogicalProject([MagicMock()], children=[root_filter_opr])
                opt_cxt, root_grp_id = self.top_down_rewrite(root_project_opr)
                opt_cxt, root_grp_id = self.bottom_up_rewrite(root_grp_id, opt_cxt)
                opt_cxt, root_grp_id = self.implement_group(root_grp_id, opt_cxt)
                expected_expr_order = [ProjectPlan, PredicatePlan, SeqScanPlan, ProjectPlan, SeqScanPlan]
                curr_grp_id = root_grp_id
                idx = 0
                while True:
                    root_grp = opt_cxt.memo.groups[curr_grp_id]
                    best_root_grp_expr = root_grp.get_best_expr(PropertyType.DEFAULT)
                    self.assertEqual(type(best_root_grp_expr.opr), expected_expr_order[idx])
                    idx += 1
                    if idx == len(expected_expr_order):
                        break
                    curr_grp_id = best_root_grp_expr.children[0]

class TestOptimizerContext(unittest.TestCase):

    def test_add_root(self):
        fake_opr = MagicMock()
        fake_opr.children = []
        opt_ctxt = OptimizerContext(MagicMock(), CostModel())
        opt_ctxt.add_opr_to_group(fake_opr)
        self.assertEqual(len(opt_ctxt.memo.group_exprs), 1)

class TopDownRewrite(OptimizerTask):

    def __init__(self, root_expr: GroupExpression, rule_set: List[Rule], optimizer_context: OptimizerContext):
        self.root_expr = root_expr
        self.rule_set = rule_set
        super().__init__(optimizer_context, OptimizerTaskType.TOP_DOWN_REWRITE)

    def execute(self):
        valid_rules = []
        for rule in self.rule_set:
            if not self.root_expr.is_rule_explored(rule.rule_type) and rule.top_match(self.root_expr.opr):
                valid_rules.append(rule)
        valid_rules = sorted(valid_rules, key=lambda x: x.promise())
        for rule in valid_rules:
            binder = Binder(self.root_expr, rule.pattern, self.optimizer_context.memo)
            for match in iter(binder):
                if not rule.check(match, self.optimizer_context):
                    continue
                after = rule.apply(match, self.optimizer_context)
                plans = list(after)
                assert len(plans) <= 1, 'Rewrite rule cannot generate more than open alternate plan.'
                for plan in plans:
                    new_expr = self.optimizer_context.replace_expression(plan, self.root_expr.group_id)
                    self.optimizer_context.task_stack.push(TopDownRewrite(new_expr, self.rule_set, self.optimizer_context))
                    return
                self.root_expr.mark_rule_explored(rule.rule_type)
        for child in self.root_expr.children:
            child_expr = self.optimizer_context.memo.groups[child].logical_exprs[0]
            self.optimizer_context.task_stack.push(TopDownRewrite(child_expr, self.rule_set, self.optimizer_context))

class BottomUpRewrite(OptimizerTask):

    def __init__(self, root_expr: GroupExpression, rule_set: List[Rule], optimizer_context: OptimizerContext, children_explored=False):
        super().__init__(optimizer_context, OptimizerTaskType.BOTTOM_UP_REWRITE)
        self._children_explored = children_explored
        self.root_expr = root_expr
        self.rule_set = rule_set

    def execute(self):
        if not self._children_explored:
            self.optimizer_context.task_stack.push(BottomUpRewrite(self.root_expr, self.rule_set, self.optimizer_context, True))
            for child in self.root_expr.children:
                child_expr = self.optimizer_context.memo.groups[child].logical_exprs[0]
                self.optimizer_context.task_stack.push(BottomUpRewrite(child_expr, self.rule_set, self.optimizer_context))
            return
        valid_rules = []
        for rule in self.rule_set:
            if not self.root_expr.is_rule_explored(rule.rule_type) and rule.top_match(self.root_expr.opr):
                valid_rules.append(rule)
        sorted(valid_rules, key=lambda x: x.promise())
        for rule in valid_rules:
            binder = Binder(self.root_expr, rule.pattern, self.optimizer_context.memo)
            for match in iter(binder):
                if not rule.check(match, self.optimizer_context):
                    continue
                logger.info('In BottomUp, Rule {} matched for {}'.format(rule, self.root_expr))
                after = rule.apply(match, self.optimizer_context)
                plans = list(after)
                assert len(plans) <= 1, 'Rewrite rule cannot generate more than open alternate plan.'
                for plan in plans:
                    new_expr = self.optimizer_context.replace_expression(plan, self.root_expr.group_id)
                    logger.info('After rewriting {}'.format(self.root_expr))
                    self.optimizer_context.task_stack.push(BottomUpRewrite(new_expr, self.rule_set, self.optimizer_context))
                    return
            self.root_expr.mark_rule_explored(rule.rule_type)

class ApplyRule(OptimizerTask):
    """apply a transformation or implementation rule"""

    def __init__(self, rule: Rule, root_expr: GroupExpression, optimizer_context: OptimizerContext, explore: bool):
        self.rule = rule
        self.root_expr = root_expr
        self.explore = explore
        super().__init__(optimizer_context, OptimizerTaskType.APPLY_RULE)

    def execute(self):
        if self.root_expr.is_rule_explored(self.rule.rule_type):
            return
        binder = Binder(self.root_expr, self.rule.pattern, self.optimizer_context.memo)
        for match in iter(binder):
            if not self.rule.check(match, self.optimizer_context):
                continue
            after = self.rule.apply(match, self.optimizer_context)
            for plan in after:
                new_expr = self.optimizer_context.add_opr_to_group(plan, self.root_expr.group_id)
                if new_expr.is_logical():
                    self.optimizer_context.task_stack.push(OptimizeExpression(new_expr, self.optimizer_context, self.explore))
                else:
                    self.optimizer_context.task_stack.push(OptimizeInputs(new_expr, self.optimizer_context))
        self.root_expr.mark_rule_explored(self.rule.rule_type)

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

class PlanGenerator:
    """
    Used for building Physical Plan from Logical Plan.
    """

    def __init__(self, db: EvaDBDatabase, rules_manager: RulesManager=None, cost_model: CostModel=None) -> None:
        self.db = db
        is_ray_enabled = self.db.catalog().get_configuration_catalog_value('ray')
        self.rules_manager = rules_manager or RulesManager({'ray': is_ray_enabled})
        self.cost_model = cost_model or CostModel()

    def execute_task_stack(self, task_stack: OptimizerTaskStack):
        while not task_stack.empty():
            task = task_stack.pop()
            task.execute()

    def build_optimal_physical_plan(self, root_grp_id: int, optimizer_context: OptimizerContext):
        physical_plan = None
        root_grp = optimizer_context.memo.groups[root_grp_id]
        best_grp_expr = root_grp.get_best_expr(PropertyType.DEFAULT)
        physical_plan = best_grp_expr.opr
        for child_grp_id in best_grp_expr.children:
            child_plan = self.build_optimal_physical_plan(child_grp_id, optimizer_context)
            physical_plan.append_child(child_plan)
        return physical_plan

    def optimize(self, logical_plan: Operator):
        optimizer_context = OptimizerContext(self.db, self.cost_model, self.rules_manager)
        memo = optimizer_context.memo
        grp_expr = optimizer_context.add_opr_to_group(opr=logical_plan)
        root_grp_id = grp_expr.group_id
        root_expr = memo.groups[root_grp_id].logical_exprs[0]
        optimizer_context.task_stack.push(TopDownRewrite(root_expr, self.rules_manager.stage_one_rewrite_rules, optimizer_context))
        self.execute_task_stack(optimizer_context.task_stack)
        root_expr = memo.groups[root_grp_id].logical_exprs[0]
        optimizer_context.task_stack.push(BottomUpRewrite(root_expr, self.rules_manager.stage_two_rewrite_rules, optimizer_context))
        self.execute_task_stack(optimizer_context.task_stack)
        root_group = memo.get_group_by_id(root_grp_id)
        optimizer_context.task_stack.push(OptimizeGroup(root_group, optimizer_context))
        self.execute_task_stack(optimizer_context.task_stack)
        optimal_plan = self.build_optimal_physical_plan(root_grp_id, optimizer_context)
        return optimal_plan

    def build(self, logical_plan: Operator):
        try:
            plan = self.optimize(logical_plan)
        except TimeoutError:
            raise ValueError('Optimizer timed out!')
        return plan

class Memo:
    """
    For now, we assume every group has only one logic expression.
    """

    def __init__(self):
        self._group_exprs: Dict[int, GroupExpression] = dict()
        self._groups = dict()

    @property
    def groups(self):
        return self._groups

    @property
    def group_exprs(self):
        return self._group_exprs

    def find_duplicate_expr(self, expr: GroupExpression) -> GroupExpression:
        if hash(expr) in self.group_exprs:
            return self.group_exprs[hash(expr)]
        else:
            return None

    def get_group_by_id(self, group_id: int) -> GroupExpression:
        if group_id in self._groups.keys():
            return self._groups[group_id]
        else:
            logger.error('Missing group id')
    '\n    For the consistency of the memo, all modification should use the\n    following functions.\n    '

    def _get_table_aliases(self, expr: GroupExpression) -> List[str]:
        """
        Collects table aliases of all the children
        """
        aliases = []
        for child_grp_id in expr.children:
            child_grp = self._groups[child_grp_id]
            aliases.extend(child_grp.aliases)
        if expr.opr.opr_type == OperatorType.LOGICALGET or expr.opr.opr_type == OperatorType.LOGICALQUERYDERIVEDGET:
            aliases.append(expr.opr.alias)
        elif expr.opr.opr_type == OperatorType.LOGICALFUNCTIONSCAN:
            aliases.append(expr.opr.alias)
        return aliases

    def _create_new_group(self, expr: GroupExpression):
        """
        Create new group for the expr
        """
        new_group_id = len(self._groups)
        aliases = self._get_table_aliases(expr)
        self._groups[new_group_id] = Group(new_group_id, aliases)
        self._insert_expr(expr, new_group_id)

    def _insert_expr(self, expr: GroupExpression, group_id: int):
        """
        Insert a group expression into a particular group
        """
        assert group_id < len(self.groups), 'Group Id out of the bound'
        group = self.groups[group_id]
        group.add_expr(expr)
        self._group_exprs[hash(expr)] = expr

    def erase_group(self, group_id: int):
        """
        Remove all the expr from the group_id
        """
        group = self.groups[group_id]
        for expr in group.logical_exprs:
            del self._group_exprs[hash(expr)]
        for expr in group.physical_exprs:
            del self._group_exprs[hash(expr)]
        group.clear_grp_exprs()

    def add_group_expr(self, expr: GroupExpression, group_id: int=UNDEFINED_GROUP_ID) -> GroupExpression:
        """
        Add an expression into the memo.
        If expr exists, we return it.
        If group_id is not specified, creates a new group
        Otherwise, inserts the expr into specified group.
        """
        duplicate_expr = self.find_duplicate_expr(expr)
        if duplicate_expr is not None:
            return duplicate_expr
        expr.group_id = group_id
        if expr.group_id == UNDEFINED_GROUP_ID:
            self._create_new_group(expr)
        else:
            self._insert_expr(expr, group_id)
        assert expr.group_id is not UNDEFINED_GROUP_ID, 'Expr should have a valid group id'
        return expr

class EmbedFilterIntoGet(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern.append_child(Pattern(OperatorType.LOGICALGET))
        super().__init__(RuleType.EMBED_FILTER_INTO_GET, pattern)

    def promise(self):
        return Promise.EMBED_FILTER_INTO_GET

    def check(self, before: LogicalFilter, context: OptimizerContext):
        predicate = before.predicate
        lget: LogicalGet = before.children[0]
        if predicate and is_video_table(lget.table_obj):
            video_alias = lget.video.alias
            col_alias = f'{video_alias}.id'
            pushdown_pred, _ = extract_pushdown_predicate(predicate, col_alias)
            if pushdown_pred:
                return True
        return False

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        predicate = before.predicate
        lget = before.children[0]
        video_alias = lget.video.alias
        col_alias = f'{video_alias}.id'
        pushdown_pred, unsupported_pred = extract_pushdown_predicate(predicate, col_alias)
        if pushdown_pred:
            new_get_opr = LogicalGet(lget.video, lget.table_obj, alias=lget.alias, predicate=pushdown_pred, target_list=lget.target_list, sampling_rate=lget.sampling_rate, sampling_type=lget.sampling_type, children=lget.children)
            if unsupported_pred:
                unsupported_opr = LogicalFilter(unsupported_pred)
                unsupported_opr.append_child(new_get_opr)
                new_get_opr = unsupported_opr
            yield new_get_opr
        else:
            yield before

class EmbedSampleIntoGet(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALSAMPLE)
        pattern.append_child(Pattern(OperatorType.LOGICALGET))
        super().__init__(RuleType.EMBED_SAMPLE_INTO_GET, pattern)

    def promise(self):
        return Promise.EMBED_SAMPLE_INTO_GET

    def check(self, before: LogicalSample, context: OptimizerContext):
        lget: LogicalGet = before.children[0]
        if lget.table_obj.table_type == TableType.VIDEO_DATA:
            return True
        return False

    def apply(self, before: LogicalSample, context: OptimizerContext):
        sample_freq = before.sample_freq.value
        sample_type = before.sample_type.value.value if before.sample_type else None
        lget: LogicalGet = before.children[0]
        new_get_opr = LogicalGet(lget.video, lget.table_obj, alias=lget.alias, predicate=lget.predicate, target_list=lget.target_list, sampling_rate=sample_freq, sampling_type=sample_type, children=lget.children)
        yield new_get_opr

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

class CacheFunctionExpressionInApply(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICAL_APPLY_AND_MERGE)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.CACHE_FUNCTION_EXPRESISON_IN_APPLY, pattern)

    def promise(self):
        return Promise.CACHE_FUNCTION_EXPRESISON_IN_APPLY

    def check(self, before: LogicalApplyAndMerge, context: OptimizerContext):
        expr = before.func_expr
        if expr.has_cache() or expr.name not in CACHEABLE_FUNCTIONS:
            return False
        if len(expr.children) > 1 or not isinstance(expr.children[0], TupleValueExpression):
            return False
        return True

    def apply(self, before: LogicalApplyAndMerge, context: OptimizerContext):
        new_func_expr = enable_cache(context, before.func_expr)
        after = LogicalApplyAndMerge(func_expr=new_func_expr, alias=before.alias, do_unnest=before.do_unnest)
        after.append_child(before.children[0])
        yield after

class PushDownFilterThroughJoin(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern_join = Pattern(OperatorType.LOGICALJOIN)
        pattern_join.append_child(Pattern(OperatorType.DUMMY))
        pattern_join.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(pattern_join)
        super().__init__(RuleType.PUSHDOWN_FILTER_THROUGH_JOIN, pattern)

    def promise(self):
        return Promise.PUSHDOWN_FILTER_THROUGH_JOIN

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        predicate = before.predicate
        join: LogicalJoin = before.children[0]
        left: Dummy = join.children[0]
        right: Dummy = join.children[1]
        new_join_node = LogicalJoin(join.join_type, join.join_predicate, join.left_keys, join.right_keys)
        left_group_aliases = context.memo.get_group_by_id(left.group_id).aliases
        right_group_aliases = context.memo.get_group_by_id(right.group_id).aliases
        left_pushdown_pred, rem_pred = extract_pushdown_predicate_for_alias(predicate, left_group_aliases)
        right_pushdown_pred, rem_pred = extract_pushdown_predicate_for_alias(rem_pred, right_group_aliases)
        if left_pushdown_pred:
            left_filter = LogicalFilter(predicate=left_pushdown_pred)
            left_filter.append_child(left)
            new_join_node.append_child(left_filter)
        else:
            new_join_node.append_child(left)
        if right_pushdown_pred:
            right_filter = LogicalFilter(predicate=right_pushdown_pred)
            right_filter.append_child(right)
            new_join_node.append_child(right_filter)
        else:
            new_join_node.append_child(right)
        if rem_pred:
            new_join_node._join_predicate = conjunction_list_to_expression_tree([rem_pred, new_join_node.join_predicate])
        yield new_join_node

class XformLateralJoinToLinearFlow(Rule):
    """If the inner node of a lateral join is a function-valued expression, we
    eliminate the join node and make the inner node the parent of the outer node. This
    produces a linear data flow path. Because this scenario is common in our system,
    we chose to explicitly convert it to a linear flow, which simplifies the
    implementation of other optimizations such as function reuse and parallelized plans by
    removing the join."""

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALJOIN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(Pattern(OperatorType.LOGICALFUNCTIONSCAN))
        super().__init__(RuleType.XFORM_LATERAL_JOIN_TO_LINEAR_FLOW, pattern)

    def promise(self):
        return Promise.XFORM_LATERAL_JOIN_TO_LINEAR_FLOW

    def check(self, before: LogicalJoin, context: OptimizerContext):
        if before.join_type == JoinType.LATERAL_JOIN:
            if before.join_predicate is None and (not before.join_project):
                return True
        return False

    def apply(self, before: LogicalJoin, context: OptimizerContext):
        A: Dummy = before.children[0]
        logical_func_scan: LogicalFunctionScan = before.children[1]
        logical_apply_merge = LogicalApplyAndMerge(logical_func_scan.func_expr, logical_func_scan.alias, logical_func_scan.do_unnest)
        logical_apply_merge.append_child(A)
        yield logical_apply_merge

class PushDownFilterThroughApplyAndMerge(Rule):
    """If it is feasible to partially or fully push the predicate contained within the
    logical filter through the ApplyAndMerge operator, we should do so. This is often
    beneficial, for instance, in order to prevent decoding additional frames beyond
    those that satisfy the predicate.
    Eg:

    Filter(id < 10 and func.label = 'car')           Filter(func.label = 'car')
            |                                                   |
        ApplyAndMerge(func)                  ->          ApplyAndMerge(func)
            |                                                   |
            A                                            Filter(id < 10)
                                                                |
                                                                A

    """

    def __init__(self):
        appply_merge_pattern = Pattern(OperatorType.LOGICAL_APPLY_AND_MERGE)
        appply_merge_pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern.append_child(appply_merge_pattern)
        super().__init__(RuleType.PUSHDOWN_FILTER_THROUGH_APPLY_AND_MERGE, pattern)

    def promise(self):
        return Promise.PUSHDOWN_FILTER_THROUGH_APPLY_AND_MERGE

    def check(self, before: LogicalFilter, context: OptimizerContext):
        return True

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        A: Dummy = before.children[0].children[0]
        apply_and_merge: LogicalApplyAndMerge = before.children[0]
        aliases = context.memo.get_group_by_id(A.group_id).aliases
        predicate = before.predicate
        pushdown_pred, rem_pred = extract_pushdown_predicate_for_alias(predicate, aliases)
        if pushdown_pred is None:
            return
        if pushdown_pred:
            pushdown_filter = LogicalFilter(predicate=pushdown_pred)
            pushdown_filter.append_child(A)
            apply_and_merge.children = [pushdown_filter]
        root_node = apply_and_merge
        if rem_pred:
            root_node = LogicalFilter(predicate=rem_pred)
            root_node.append_child(apply_and_merge)
        yield root_node

class XformExtractObjectToLinearFlow(Rule):
    """If the inner node of a lateral join is a Extract_Object function-valued
    expression, we eliminate the join node and make the inner node the parent of the
    outer node. This produces a linear data flow path.
    TODO: We need to add a sorting operation after detector to ensure we always provide tracker data in order.
    """

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALJOIN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(Pattern(OperatorType.LOGICAL_EXTRACT_OBJECT))
        super().__init__(RuleType.XFORM_EXTRACT_OBJECT_TO_LINEAR_FLOW, pattern)

    def promise(self):
        return Promise.XFORM_EXTRACT_OBJECT_TO_LINEAR_FLOW

    def check(self, before: LogicalJoin, context: OptimizerContext):
        if before.join_type == JoinType.LATERAL_JOIN:
            return True
        return False

    def apply(self, before: LogicalJoin, context: OptimizerContext):
        A: Dummy = before.children[0]
        logical_extract_obj: LogicalExtractObject = before.children[1]
        detector = LogicalApplyAndMerge(logical_extract_obj.detector, alias=logical_extract_obj.detector.alias)
        tracker = LogicalApplyAndMerge(logical_extract_obj.tracker, alias=logical_extract_obj.alias, do_unnest=logical_extract_obj.do_unnest)
        detector.append_child(A)
        tracker.append_child(detector)
        yield tracker

class CombineSimilarityOrderByAndLimitToVectorIndexScan(Rule):
    """
    This rule currently rewrites Order By + Limit to a vector index scan.
    Because vector index only works for similarity search, the rule will
    only be applied when the Order By is on Similarity expression. For
    simplicity, we also only enable this rule when the Similarity expression
    applies to the full table. Predicated query will yield incorrect results
    if we use an index scan.

    Limit(10)
        |
    OrderBy(func)        ->        IndexScan(10)
        |                               |
        A                               A
    """

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALLIMIT)
        orderby_pattern = Pattern(OperatorType.LOGICALORDERBY)
        orderby_pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(orderby_pattern)
        super().__init__(RuleType.COMBINE_SIMILARITY_ORDERBY_AND_LIMIT_TO_VECTOR_INDEX_SCAN, pattern)
        self._index_catalog_entry = None
        self._query_func_expr = None

    def promise(self):
        return Promise.COMBINE_SIMILARITY_ORDERBY_AND_LIMIT_TO_VECTOR_INDEX_SCAN

    def check(self, before: LogicalLimit, context: OptimizerContext):
        return True

    def apply(self, before: LogicalLimit, context: OptimizerContext):
        catalog_manager = context.db.catalog
        limit_node = before
        orderby_node = before.children[0]
        sub_tree_root = orderby_node.children[0]

        def _exists_predicate(opr):
            if isinstance(opr, LogicalGet):
                return opr.predicate is not None
            return True
        if _exists_predicate(sub_tree_root.opr):
            return
        func_orderby_expr = None
        for column, sort_type in orderby_node.orderby_list:
            if isinstance(column, FunctionExpression) and sort_type == ParserOrderBySortType.ASC:
                func_orderby_expr = column
        if not func_orderby_expr or func_orderby_expr.name != 'Similarity':
            return
        tb_catalog_entry = list(sub_tree_root.opr.find_all(LogicalGet))[0].table_obj
        db_catalog_entry = catalog_manager().get_database_catalog_entry(tb_catalog_entry.database_name)
        is_postgres_data_source = db_catalog_entry is not None and db_catalog_entry.engine == 'postgres'
        query_func_expr, base_func_expr = func_orderby_expr.children
        tv_expr = base_func_expr
        while not isinstance(tv_expr, TupleValueExpression):
            tv_expr = tv_expr.children[0]
        column_catalog_entry = tv_expr.col_object
        if not is_postgres_data_source:
            function_signature = None if isinstance(base_func_expr, TupleValueExpression) else base_func_expr.signature()
            index_catalog_entry = catalog_manager().get_index_catalog_entry_by_column_and_function_signature(column_catalog_entry, function_signature)
            if not index_catalog_entry:
                return
        else:
            index_catalog_entry = IndexCatalogEntry(name='', save_file_path='', type=VectorStoreType.PGVECTOR, feat_column=column_catalog_entry)
        vector_index_scan_node = LogicalVectorIndexScan(index_catalog_entry, limit_node.limit_count, query_func_expr)
        for child in orderby_node.children:
            vector_index_scan_node.append_child(child)
        yield vector_index_scan_node

class LogicalInnerJoinCommutativity(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALJOIN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_INNER_JOIN_COMMUTATIVITY, pattern)

    def promise(self):
        return Promise.LOGICAL_INNER_JOIN_COMMUTATIVITY

    def check(self, before: LogicalJoin, context: OptimizerContext):
        return before.join_type == JoinType.INNER_JOIN

    def apply(self, before: LogicalJoin, context: OptimizerContext):
        new_join = LogicalJoin(before.join_type, before.join_predicate)
        new_join.append_child(before.rhs())
        new_join.append_child(before.lhs())
        yield new_join

class ReorderPredicates(Rule):
    """
    The current implementation orders conjuncts based on their individual cost.
    The optimization for OR clauses has `not` been implemented yet. Additionally, we do
    not optimize predicates that are not user-defined functions since we assume that
    they will likely be pushed to the underlying relational database, which will handle
    the optimization process.
    """

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.REORDER_PREDICATES, pattern)

    def promise(self):
        return Promise.REORDER_PREDICATES

    def check(self, before: LogicalFilter, context: OptimizerContext):
        return len(list(before.predicate.find_all(FunctionExpression))) > 0

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        conjuncts = to_conjunction_list(before.predicate)
        contains_func_exprs = []
        simple_exprs = []
        for conjunct in conjuncts:
            if list(conjunct.find_all(FunctionExpression)):
                contains_func_exprs.append(conjunct)
            else:
                simple_exprs.append(conjunct)
        function_expr_cost_tuples = [(expr, get_expression_execution_cost(context, expr)) for expr in contains_func_exprs]
        function_expr_cost_tuples = sorted(function_expr_cost_tuples, key=lambda x: x[1])
        ordered_conjuncts = simple_exprs + [expr for expr, _ in function_expr_cost_tuples]
        if ordered_conjuncts != conjuncts:
            reordered_predicate = conjunction_list_to_expression_tree(ordered_conjuncts)
            reordered_filter_node = LogicalFilter(predicate=reordered_predicate)
            reordered_filter_node.append_child(before.children[0])
            yield reordered_filter_node

class LogicalCreateToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALCREATE)
        super().__init__(RuleType.LOGICAL_CREATE_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_CREATE_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalCreate, context: OptimizerContext):
        after = CreatePlan(before.video, before.column_list, before.if_not_exists)
        yield after

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

class LogicalRenameToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALRENAME)
        super().__init__(RuleType.LOGICAL_RENAME_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_RENAME_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalRename, context: OptimizerContext):
        after = RenamePlan(before.old_table_ref, before.new_name)
        yield after

class LogicalCreateFunctionToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALCREATEFUNCTION)
        super().__init__(RuleType.LOGICAL_CREATE_FUNCTION_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_CREATE_FUNCTION_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalCreateFunction, context: OptimizerContext):
        after = CreateFunctionPlan(before.name, before.or_replace, before.if_not_exists, before.inputs, before.outputs, before.impl_path, before.function_type, before.metadata)
        yield after

class LogicalCreateFunctionFromSelectToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALCREATEFUNCTION)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_CREATE_FUNCTION_FROM_SELECT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_CREATE_FUNCTION_FROM_SELECT_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalCreateFunction, context: OptimizerContext):
        after = CreateFunctionPlan(before.name, before.or_replace, before.if_not_exists, before.inputs, before.outputs, before.impl_path, before.function_type, before.metadata)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalCreateIndexToVectorIndex(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALCREATEINDEX)
        super().__init__(RuleType.LOGICAL_CREATE_INDEX_TO_VECTOR_INDEX, pattern)

    def promise(self):
        return Promise.LOGICAL_CREATE_INDEX_TO_VECTOR_INDEX

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalCreateIndex, context: OptimizerContext):
        after = CreateIndexPlan(before.name, before.if_not_exists, before.table_ref, before.col_list, before.vector_store_type, before.project_expr_list, before.index_def)
        child = SeqScanPlan(None, before.project_expr_list, before.table_ref.alias)
        batch_mem_size = context.db.catalog().get_configuration_catalog_value('batch_mem_size')
        child.append_child(StoragePlan(before.table_ref.table.table_obj, before.table_ref, batch_mem_size=batch_mem_size))
        after.append_child(child)
        yield after

class LogicalDropObjectToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICAL_DROP_OBJECT)
        super().__init__(RuleType.LOGICAL_DROP_OBJECT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_DROP_OBJECT_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalDropObject, context: OptimizerContext):
        after = DropObjectPlan(before.object_type, before.name, before.if_exists)
        yield after

class LogicalInsertToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALINSERT)
        super().__init__(RuleType.LOGICAL_INSERT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_INSERT_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalInsert, context: OptimizerContext):
        after = InsertPlan(before.table, before.column_list, before.value_list)
        yield after

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

class LogicalLoadToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALLOADDATA)
        super().__init__(RuleType.LOGICAL_LOAD_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_LOAD_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalLoadData, context: OptimizerContext):
        after = LoadDataPlan(before.table_info, before.path, before.column_list, before.file_options)
        yield after

class LogicalGetToSeqScan(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALGET)
        super().__init__(RuleType.LOGICAL_GET_TO_SEQSCAN, pattern)

    def promise(self):
        return Promise.LOGICAL_GET_TO_SEQSCAN

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalGet, context: OptimizerContext):
        after = SeqScanPlan(None, before.target_list, before.alias)
        batch_mem_size = context.db.catalog().get_configuration_catalog_value('batch_mem_size')
        after.append_child(StoragePlan(before.table_obj, before.video, predicate=before.predicate, sampling_rate=before.sampling_rate, sampling_type=before.sampling_type, chunk_params=before.chunk_params, batch_mem_size=batch_mem_size))
        yield after

class LogicalDerivedGetToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALQUERYDERIVEDGET)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_DERIVED_GET_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_DERIVED_GET_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalQueryDerivedGet, context: OptimizerContext):
        after = SeqScanPlan(before.predicate, before.target_list, before.alias)
        after.append_child(before.children[0])
        yield after

class LogicalUnionToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALUNION)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_UNION_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_UNION_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalUnion, context: OptimizerContext):
        after = UnionPlan(before.all)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalGroupByToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALGROUPBY)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_GROUPBY_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_GROUPBY_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalGroupBy, context: OptimizerContext):
        after = GroupByPlan(before.groupby_clause)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalOrderByToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALORDERBY)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_ORDERBY_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_ORDERBY_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalOrderBy, context: OptimizerContext):
        after = OrderByPlan(before.orderby_list)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalLimitToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALLIMIT)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_LIMIT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_LIMIT_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalLimit, context: OptimizerContext):
        after = LimitPlan(before.limit_count)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalFunctionScanToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALFUNCTIONSCAN)
        super().__init__(RuleType.LOGICAL_FUNCTION_SCAN_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_FUNCTION_SCAN_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalFunctionScan, context: OptimizerContext):
        after = FunctionScanPlan(before.func_expr, before.do_unnest)
        yield after

class LogicalLateralJoinToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALJOIN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_LATERAL_JOIN_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_LATERAL_JOIN_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return before.join_type == JoinType.LATERAL_JOIN

    def apply(self, join_node: LogicalJoin, context: OptimizerContext):
        lateral_join_plan = LateralJoinPlan(join_node.join_predicate)
        lateral_join_plan.join_project = join_node.join_project
        lateral_join_plan.append_child(join_node.lhs())
        lateral_join_plan.append_child(join_node.rhs())
        yield lateral_join_plan

class LogicalJoinToPhysicalHashJoin(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALJOIN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_JOIN_TO_PHYSICAL_HASH_JOIN, pattern)

    def promise(self):
        return Promise.LOGICAL_JOIN_TO_PHYSICAL_HASH_JOIN

    def check(self, before: Operator, context: OptimizerContext):
        """
        We don't want to apply this rule to the join when FuzzDistance
        is being used, which implies that the join is a FuzzyJoin
        """
        if before.join_predicate is None:
            return False
        j_child: FunctionExpression = before.join_predicate.children[0]
        if isinstance(j_child, FunctionExpression):
            if j_child.name.startswith('FuzzDistance'):
                return before.join_type == JoinType.INNER_JOIN and (not j_child or not j_child.name.startswith('FuzzDistance'))
        else:
            return before.join_type == JoinType.INNER_JOIN

    def apply(self, join_node: LogicalJoin, context: OptimizerContext):
        a: Dummy = join_node.lhs()
        b: Dummy = join_node.rhs()
        a_table_aliases = context.memo.get_group_by_id(a.group_id).aliases
        b_table_aliases = context.memo.get_group_by_id(b.group_id).aliases
        join_predicates = join_node.join_predicate
        a_join_keys, b_join_keys = extract_equi_join_keys(join_predicates, a_table_aliases, b_table_aliases)
        build_plan = HashJoinBuildPlan(join_node.join_type, a_join_keys)
        build_plan.append_child(a)
        probe_side = HashJoinProbePlan(join_node.join_type, b_join_keys, join_predicates, join_node.join_project)
        probe_side.append_child(build_plan)
        probe_side.append_child(b)
        yield probe_side

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

class LogicalFilterToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_FILTER_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_FILTER_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        after = PredicatePlan(before.predicate)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalProjectToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALPROJECT)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_PROJECT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_PROJECT_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalProject, context: OptimizerContext):
        after = ProjectPlan(before.target_list)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalProjectNoTableToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALPROJECT)
        super().__init__(RuleType.LOGICAL_PROJECT_NO_TABLE_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_PROJECT_NO_TABLE_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalProject, context: OptimizerContext):
        after = ProjectPlan(before.target_list)
        yield after

class LogicalShowToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICAL_SHOW)
        super().__init__(RuleType.LOGICAL_SHOW_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_SHOW_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalShow, context: OptimizerContext):
        after = ShowInfoPlan(before.show_type, before.show_val)
        yield after

class LogicalExplainToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALEXPLAIN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_EXPLAIN_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_EXPLAIN_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalExplain, context: OptimizerContext):
        after = ExplainPlan(before.explainable_opr)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalApplyAndMergeToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICAL_APPLY_AND_MERGE)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_APPLY_AND_MERGE_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_APPLY_AND_MERGE_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalApplyAndMerge, context: OptimizerContext):
        after = ApplyAndMergePlan(before.func_expr, before.alias, before.do_unnest)
        for child in before.children:
            after.append_child(child)
        yield after

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

class LogicalExchangeToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALEXCHANGE)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_EXCHANGE_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_EXCHANGE_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalExchange, context: OptimizerContext):
        after = ExchangePlan(before.view)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalApplyAndMergeToRayPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICAL_APPLY_AND_MERGE)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_APPLY_AND_MERGE_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_APPLY_AND_MERGE_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalApplyAndMerge, context: OptimizerContext):
        apply_plan = ApplyAndMergePlan(before.func_expr, before.alias, before.do_unnest)
        parallelism = 2
        ray_process_env_dict = get_ray_env_dict()
        ray_parallel_env_conf_dict = [ray_process_env_dict for _ in range(parallelism)]
        exchange_plan = ExchangePlan(inner_plan=apply_plan, parallelism=parallelism, ray_pull_env_conf_dict=ray_process_env_dict, ray_parallel_env_conf_dict=ray_parallel_env_conf_dict)
        for child in before.children:
            exchange_plan.append_child(child)
        yield exchange_plan

class LogicalProjectToRayPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALPROJECT)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_PROJECT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_PROJECT_TO_PHYSICAL

    def check(self, before: LogicalProject, context: OptimizerContext):
        return True

    def apply(self, before: LogicalProject, context: OptimizerContext):
        project_plan = ProjectPlan(before.target_list)
        if before.target_list is None or not any([isinstance(expr, FunctionExpression) for expr in before.target_list]):
            for child in before.children:
                project_plan.append_child(child)
            yield project_plan
        else:
            parallelism = 2
            ray_process_env_dict = get_ray_env_dict()
            ray_parallel_env_conf_dict = [ray_process_env_dict for _ in range(parallelism)]
            exchange_plan = ExchangePlan(inner_plan=project_plan, parallelism=parallelism, ray_pull_env_conf_dict=ray_process_env_dict, ray_parallel_env_conf_dict=ray_parallel_env_conf_dict)
            for child in before.children:
                exchange_plan.append_child(child)
            yield exchange_plan

