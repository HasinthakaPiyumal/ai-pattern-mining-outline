# Cluster 68

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

