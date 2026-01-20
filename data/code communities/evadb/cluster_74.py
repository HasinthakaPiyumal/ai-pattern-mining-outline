# Cluster 74

class CreateJob:

    def create_job(self, tree):
        job_name = None
        queries = []
        start_time = None
        end_time = None
        repeat_interval = None
        repeat_period = None
        if_not_exists = False
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'if_not_exists':
                    if_not_exists = True
                if child.data == 'uid':
                    job_name = self.visit(child)
                if child.data == 'job_sql_statements':
                    queries = self.visit(child)
                elif child.data == 'start_time':
                    start_time = self.visit(child)
                elif child.data == 'end_time':
                    end_time = self.visit(child)
                elif child.data == 'repeat_clause':
                    repeat_interval, repeat_period = self.visit(child)
        create_job = CreateJobStatement(job_name, queries, if_not_exists, start_time, end_time, repeat_interval, repeat_period)
        return create_job

    def start_time(self, tree):
        return self.visit(tree.children[1]).value

    def end_time(self, tree):
        return self.visit(tree.children[1]).value

    def repeat_clause(self, tree):
        return (self.visit(tree.children[1]), self.visit(tree.children[2]))

