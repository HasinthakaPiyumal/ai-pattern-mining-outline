# Cluster 76

class Explain:

    def explain_statement(self, tree):
        explainable_stmt = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data.endswith('explainable_statement'):
                    explainable_stmt = self.visit(child)
        return ExplainStatement(explainable_stmt)

