# Cluster 73

class CreateDatabase:

    def create_database(self, tree):
        database_name = None
        if_not_exists = False
        engine = None
        param_dict = {}
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'if_not_exists':
                    if_not_exists = True
                elif child.data == 'uid':
                    database_name = self.visit(child)
                elif child.data == 'create_database_engine_clause':
                    engine, param_dict = self.visit(child)
        create_stmt = CreateDatabaseStatement(database_name, if_not_exists, engine, param_dict)
        return create_stmt

    def create_database_engine_clause(self, tree):
        engine = None
        param_dict = {}
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'string_literal':
                    engine = self.visit(child).value
                elif child.data == 'colon_param_dict':
                    param_dict = self.visit(child)
        return (engine, param_dict)

