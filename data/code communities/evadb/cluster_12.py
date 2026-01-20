# Cluster 12

class ParserStatementTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_parser_statement_types(self):
        parser = Parser()
        queries = ['CREATE INDEX testindex ON MyVideo (featCol) USING FAISS;', 'CREATE TABLE IF NOT EXISTS Persons (\n                  Frame_ID INTEGER UNIQUE,\n                  Frame_Data TEXT(10),\n                  Frame_Value FLOAT(1000, 201),\n                  Frame_Array NDARRAY UINT8(5, 100, 2432, 4324, 100)\n            )', 'RENAME TABLE student TO student_info', 'DROP TABLE IF EXISTS student_info', 'DROP TABLE student_info', 'DROP FUNCTION FastRCNN;', 'SELECT MIN(id), MAX(id), SUM(id) FROM ABC', "SELECT CLASS FROM TAIPAI                 WHERE (CLASS = 'VAN' AND REDNESS < 300)  OR REDNESS > 500;", 'SELECT CLASS, REDNESS FROM TAIPAI             UNION ALL SELECT CLASS, REDNESS FROM SHANGHAI;', 'SELECT CLASS, REDNESS FROM TAIPAI             UNION SELECT CLASS, REDNESS FROM SHANGHAI;', "SELECT FIRST(id) FROM TAIPAI GROUP BY '8 frames';", "SELECT CLASS, REDNESS FROM TAIPAI                     WHERE (CLASS = 'VAN' AND REDNESS < 400 ) OR REDNESS > 700                     ORDER BY CLASS, REDNESS DESC;", "INSERT INTO MyVideo (Frame_ID, Frame_Path)                                    VALUES    (1, '/mnt/frames/1.png');", 'INSERT INTO testDeleteOne (id, feat, salary, input)\n                VALUES (15, 2.5, [[100, 100, 100]], [[100, 100, 100]]);', 'DELETE FROM Foo WHERE id < 6', "LOAD VIDEO 'data/video.mp4' INTO MyVideo", "LOAD IMAGE 'data/pic.jpg' INTO MyImage", "LOAD CSV 'data/meta.csv' INTO\n                             MyMeta (id, frame_id, video_id, label);", "SELECT Licence_plate(bbox) FROM\n                            (SELECT Yolo(frame).bbox FROM autonomous_vehicle_1\n                              WHERE Yolo(frame).label = 'vehicle') AS T\n                          WHERE Is_suspicious(bbox) = 1 AND\n                                Licence_plate(bbox) = '12345';", 'CREATE TABLE uadtrac_fastRCNN AS\n               SELECT id, Yolo(frame).labels FROM MyVideo\n                        WHERE id<5; ', 'SELECT table1.a FROM table1 JOIN table2\n            ON table1.a = table2.a WHERE table1.a <= 5', 'SELECT table1.a FROM table1 JOIN table2\n            ON table1.a = table2.a JOIN table3\n            ON table3.a = table1.a WHERE table1.a <= 5', 'SELECT frame FROM MyVideo JOIN LATERAL\n                            ObjectDet(frame) AS OD;', "CREATE FUNCTION FaceDetector\n                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))\n                  OUTPUT (bboxes NDARRAY FLOAT32(ANYDIM, 4),\n                          scores NDARRAY FLOAT32(ANYDIM))\n                  TYPE  FaceDetection\n                  IMPL  'evadb/functions/face_detector.py';\n            ", 'SHOW TABLES;', 'SHOW FUNCTIONS;', 'SHOW DATABASES;', 'EXPLAIN SELECT a FROM foo;', 'SELECT HomeRentalForecast(12);', 'SELECT data FROM MyVideo WHERE id < 5\n                    ORDER BY Similarity(FeatureExtractor(Open("abc.jpg")),\n                                        FeatureExtractor(data))\n                    LIMIT 1;']
        randomized_cases = ['Create index TestIndex on MyVideo (featCol) using FAISS;', 'create table if not exists Persons (\n                    Frame_ID integer unique,\n                    Frame_Data text(10),\n                    Frame_Value float(1000, 201),\n                    Frame_Array ndArray uint8(5, 100, 2432, 4324, 100)\n            )', 'Rename Table STUDENT to student_info', 'drop table if exists Student_info', 'drop table Student_Info', 'Drop function FASTRCNN;', 'Select min(id), max(Id), Sum(Id) from ABC', "select CLASS from Taipai where (Class = 'VAN' and REDNESS < 300) or Redness > 500;", 'select class, REDNESS from TAIPAI Union all select Class, redness from Shanghai;', 'Select class, redness from Taipai Union Select CLASS, redness from Shanghai;', "Select first(Id) from Taipai group by '8F';", "Select Class, redness from TAIPAI\n                where (CLASS = 'VAN' and redness < 400 ) or REDNESS > 700\n                order by Class, redness DESC;", "Insert into MyVideo (Frame_ID, Frame_Path) values (1, '/mnt/frames/1.png');", 'insert into testDeleteOne (Id, feat, salary, input)\n                values (15, 2.5, [[100, 100, 100]], [[100, 100, 100]]);', 'delete from Foo where ID < 6', "Load video 'data/video.mp4' into MyVideo", "Load image 'data/pic.jpg' into MyImage", "Load csv 'data/meta.csv' into\n                MyMeta (id, Frame_ID, video_ID, label);", "select Licence_plate(bbox) from\n                (select Yolo(Frame).bbox from autonomous_vehicle_1\n                where Yolo(frame).label = 'vehicle') as T\n                where is_suspicious(bbox) = 1 and\n                Licence_plate(bbox) = '12345';", 'Create TABLE UADTrac_FastRCNN as\n                Select id, YoloV5(Frame).labels from MyVideo\n                    where id<5; ', 'Select Table1.A from Table1 join Table2\n                on Table1.A = Table2.a where Table1.A <= 5', 'Select Table1.A from Table1 Join Table2\n                    On Table1.a = Table2.A Join Table3\n                On Table3.A = Table1.A where Table1.a <= 5', 'Select Frame from MyVideo Join Lateral\n                ObjectDet(Frame) as OD;', "Create FUNCTION FaceDetector\n                Input (Frame ndArray uint8(3, anydim, anydim))\n                Output (bboxes ndArray float32(anydim, 4),\n                scores ndArray float32(ANYdim))\n                Type FaceDetection\n                Impl 'evadb/functions/face_detector.py';\n            ", 'CREATE DATABASE example_db\n                WITH ENGINE = "postgres",\n                PARAMETERS = {\n                    "user": "demo_user",\n                    "password": "demo_password",\n                    "host": "3.220.66.106",\n                    "port": "5432",\n                    "database": "demo"\n                };\n            ']
        queries = queries + randomized_cases
        ref_stmt = parser.parse(queries[0])[0]
        self.assertNotEqual(ref_stmt, None)
        self.assertNotEqual(ref_stmt.__str__(), None)
        statement_to_query_dict = {}
        statement_to_query_dict[ref_stmt] = queries[0]
        for other_query in queries[1:]:
            stmt = parser.parse(other_query)[0]
            self.assertNotEqual(stmt, ref_stmt)
            self.assertEqual(stmt, stmt)
            self.assertNotEqual(stmt, None)
            self.assertNotEqual(stmt.__str__(), None)
            statement_to_query_dict[stmt] = other_query

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

def parse_predicate_expression(expr: str):
    mock_query = f'SELECT * FROM DUMMY WHERE {expr};'
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, SelectStatement), 'Expected a select statement'
    return stmt.where_clause

def parse_table_clause(expr: str, chunk_size: int=None, chunk_overlap: int=None):
    mock_query_parts = [f'SELECT * FROM {expr}']
    if chunk_size is not None:
        mock_query_parts.append(f'CHUNK_SIZE {chunk_size}')
    if chunk_overlap is not None:
        mock_query_parts.append(f'CHUNK_OVERLAP {chunk_overlap}')
    mock_query_parts.append(';')
    mock_query = ' '.join(mock_query_parts)
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, SelectStatement), 'Expected a select statement'
    assert stmt.from_table.is_table_atom
    return stmt.from_table

def parse_create_function(function_name: str, if_not_exists: bool, function_file_path: str, type: str, **kwargs):
    mock_query = f'CREATE FUNCTION IF NOT EXISTS {function_name}' if if_not_exists else f'CREATE FUNCTION {function_name}'
    if type is not None:
        mock_query += f' TYPE {type}'
        task, model = (kwargs['task'], kwargs['model'])
        if task is not None and model is not None:
            mock_query += f" TASK '{task}' MODEL '{model}'"
    else:
        mock_query += f" IMPL '{function_file_path}'"
    mock_query += ';'
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, CreateFunctionStatement), 'Expected a create function statement'
    return stmt

def parse_create_table(table_name: str, if_not_exists: bool, columns: str, **kwargs):
    mock_query = f'CREATE TABLE IF NOT EXISTS {table_name} ({columns});' if if_not_exists else f'CREATE TABLE {table_name} ({columns});'
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, CreateTableStatement), 'Expected a create table statement'
    return stmt

def parse_show(show_type: str, **kwargs):
    mock_query = f'SHOW {show_type};'
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, ShowStatement), 'Expected a show statement'
    return stmt

def parse_explain(query: str, **kwargs):
    mock_query = f'EXPLAIN {query};'
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, ExplainStatement), 'Expected a explain statement'
    return stmt

def parse_insert(table_name: str, columns: str, values: str, **kwargs):
    mock_query = f'INSERT INTO {table_name} {columns} VALUES {values};'
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, InsertTableStatement), 'Expected a insert statement'
    return stmt

def parse_load(table_name: str, file_regex: str, format: str, **kwargs):
    mock_query = f"LOAD {format.upper()} '{file_regex}' INTO {table_name};"
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, LoadDataStatement), 'Expected a load statement'
    return stmt

def parse_drop(object_type: ObjectType, name: str, if_exists: bool):
    mock_query = f'DROP {object_type}'
    mock_query = f' {mock_query} IF EXISTS {name} ' if if_exists else f'{mock_query} {name}'
    mock_query += ';'
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, DropObjectStatement), 'Expected a drop object statement'
    return stmt

def parse_drop_table(table_name: str, if_exists: bool):
    return parse_drop(ObjectType.TABLE, table_name, if_exists)

def parse_drop_function(function_name: str, if_exists: bool):
    return parse_drop(ObjectType.FUNCTION, function_name, if_exists)

def parse_drop_index(index_name: str, if_exists: bool):
    return parse_drop(ObjectType.INDEX, index_name, if_exists)

def parse_drop_database(database_name: str, if_exists: bool):
    return parse_drop(ObjectType.DATABASE, database_name, if_exists)

def parse_query(query):
    stmt = Parser().parse(query)
    assert len(stmt) == 1
    return stmt[0]

def parse_lateral_join(expr: str, alias: str):
    mock_query = f'SELECT * FROM DUMMY JOIN LATERAL {expr} AS {alias};'
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, SelectStatement), 'Expected a select statement'
    assert stmt.from_table.is_join()
    return stmt.from_table.join_node.right

def parse_create_vector_index(index_name: str, table_name: str, expr: str, using: str):
    mock_query = f'CREATE INDEX {index_name} ON {table_name} ({expr}) USING {using};'
    stmt = Parser().parse(mock_query)[0]
    return stmt

def parse_sql_orderby_expr(expr: str):
    mock_query = f'SELECT * FROM DUMMY ORDER BY {expr};'
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, SelectStatement), 'Expected a select statement'
    return stmt.orderby_list

def parse_rename(old_name: str, new_name: str):
    mock_query = f'RENAME TABLE {old_name} TO {new_name};'
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, RenameTableStatement), 'Expected a rename statement'
    return stmt

class DropObject:

    def drop_table(self, tree):
        table_name = None
        if_exists = False
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'if_exists':
                    if_exists = True
                elif child.data == 'uid':
                    table_name = self.visit(child)
        return DropObjectStatement(ObjectType.TABLE, table_name, if_exists)

    def drop_index(self, tree):
        index_name = None
        if_exists = False
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'if_exists':
                    if_exists = True
                elif child.data == 'uid':
                    index_name = self.visit(child)
        return DropObjectStatement(ObjectType.INDEX, index_name, if_exists)

    def drop_function(self, tree):
        function_name = None
        if_exists = False
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'uid':
                    function_name = self.visit(child)
                elif child.data == 'if_exists':
                    if_exists = True
        return DropObjectStatement(ObjectType.FUNCTION, function_name, if_exists)

    def drop_database(self, tree):
        database_name = None
        if_exists = False
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'if_exists':
                    if_exists = True
                elif child.data == 'uid':
                    database_name = self.visit(child)
        return DropObjectStatement(ObjectType.DATABASE, database_name, if_exists)

    def drop_job(self, tree):
        job_name = None
        if_exists = False
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'if_exists':
                    if_exists = True
                elif child.data == 'uid':
                    job_name = self.visit(child)
        return DropObjectStatement(ObjectType.JOB, job_name, if_exists)

class Set:

    def set_statement(self, tree):
        config_name = None
        config_value = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'config_name':
                    config_name = self.visit(child)
                elif child.data == 'config_value':
                    config_value = self.visit(child)
        set_stmt = SetStatement(config_name, config_value)
        return set_stmt

class Use:

    def use_statement(self, tree):
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'database_name':
                    database_name = self.visit(child)
                if child.data == 'query_string':
                    query_string = self.visit(child)
        return UseStatement(database_name, query_string)

class Show:

    def show_statement(self, tree):
        token = tree.children[1]
        if isinstance(token, str) and str.upper(token) == 'FUNCTIONS':
            return ShowStatement(show_type=ShowType.FUNCTIONS)
        elif isinstance(token, str) and str.upper(token) == 'TABLES':
            return ShowStatement(show_type=ShowType.TABLES)
        elif isinstance(token, str) and str.upper(token) == 'DATABASES':
            return ShowStatement(show_type=ShowType.DATABASES)
        elif token is not None:
            return ShowStatement(show_type=ShowType.CONFIGS, show_val=self.visit(token))

class EvaDBCursor(object):

    def __init__(self, connection):
        self._connection = connection
        self._evadb = connection._evadb
        self._pending_query = False
        self._result = None

    async def execute_async(self, query: str):
        """
        Send query to the EvaDB server.
        """
        if self._pending_query:
            raise SystemError('EvaDB does not support concurrent queries.                     Call fetch_all() to complete the pending query')
        query = self._multiline_query_transformation(query)
        self._connection._writer.write((query + '\n').encode())
        await self._connection._writer.drain()
        self._pending_query = True
        return self

    async def fetch_one_async(self) -> Response:
        """
        fetch_one returns one batch instead of one row for now.
        """
        response = Response()
        prefix = await self._connection._reader.readline()
        if prefix != b'':
            message_length = int(prefix)
            message = await self._connection._reader.readexactly(message_length)
            response = Response.deserialize(message)
        self._pending_query = False
        return response

    async def fetch_all_async(self) -> Response:
        """
        fetch_all is the same as fetch_one for now.
        """
        return await self.fetch_one_async()

    def _multiline_query_transformation(self, query: str) -> str:
        query = query.replace('\n', ' ')
        query = query.lstrip()
        query = query.rstrip(' ;')
        query += ';'
        logger.debug('Query: ' + query)
        return query

    def stop_query(self):
        self._pending_query = False

    def __getattr__(self, name):
        """
        Auto generate sync function calls from async
        Sync function calls should not be used in an async environment.
        """
        function_name_list = ['table', 'load', 'execute', 'query', 'create_function', 'create_table', 'create_vector_index', 'drop_table', 'drop_function', 'drop_index', 'df', 'show', 'insertexplain', 'rename', 'fetch_one']
        if name not in function_name_list:
            nearest_function = find_nearest_word(name, function_name_list)
            raise AttributeError(f"EvaDBCursor does not contain a function named: '{name}'. Did you mean to run: '{nearest_function}()'?")
        try:
            func = object.__getattribute__(self, '%s_async' % name)
        except Exception as e:
            raise e

        def func_sync(*args, **kwargs):
            loop = asyncio.get_event_loop()
            res = loop.run_until_complete(func(*args, **kwargs))
            return res
        return func_sync

    def table(self, table_name: str, chunk_size: int=None, chunk_overlap: int=None) -> EvaDBQuery:
        """
        Retrieves data from a table in the database.

        Args:
            table_name (str): The name of the table to retrieve data from.
            chunk_size (int, optional): The size of the chunk to break the document into. Only valid for DOCUMENT tables.
                If not provided, the default value is 4000.
            chunk_overlap (int, optional): The overlap between consecutive chunks. Only valid for DOCUMENT tables.
                If not provided, the default value is 200.

        Returns:
            EvaDBQuery: An EvaDBQuery object representing the table query.

        Examples:
            >>> relation = cursor.table("sample_table")
            >>> relation = cursor.select('*')
            >>> relation.df()
               col1  col2
            0     1     2
            1     3     4
            2     5     6

            Read a document table using chunk_size 100 and chunk_overlap 10.

            >>> relation = cursor.table("doc_table", chunk_size=100, chunk_overlap=10)
        """
        table = parse_table_clause(table_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        select_stmt = SelectStatement(target_list=[TupleValueExpression(name='*')], from_table=table)
        try_binding(self._evadb.catalog, select_stmt)
        return EvaDBQuery(self._evadb, select_stmt, alias=Alias(table_name.lower()))

    def df(self) -> pandas.DataFrame:
        """
        Returns the result as a pandas DataFrame.

        Returns:
            pandas.DataFrame: The result as a DataFrame.

        Raises:
            Exception: If no valid result is available with the current connection.

        Examples:
            >>> result = cursor.query("CREATE TABLE IF NOT EXISTS youtube_video_text AS SELECT SpeechRecognizer(audio) FROM youtube_video;").df()
            >>> result
            Empty DataFrame
            >>> relation = cursor.table("youtube_video_text").select('*').df()
                speechrecognizer.response
            0	"Sample Text from speech recognizer"
        """
        if not self._result:
            raise Exception('No valid result with the current cursor')
        return self._result.frames

    def create_vector_index(self, index_name: str, table_name: str, expr: str, using: str) -> 'EvaDBCursor':
        """
        Creates a vector index using the provided expr on the table.
        This feature directly works on IMAGE tables.
        For VIDEO tables, the feature should be extracted first and stored in an intermediate table, before creating the index.

        Args:
            index_name (str): Name of the index.
            table_name (str): Name of the table.
            expr (str): Expression used to build the vector index.

            using (str): Method used for indexing, can be `FAISS` or `QDRANT` or `PINECONE` or `CHROMADB` or `WEAVIATE` or `MILVUS`.

        Returns:
            EvaDBCursor: The EvaDBCursor object.

        Examples:
            Create a Vector Index using QDRANT

            >>> cursor.create_vector_index(
                    "faiss_index",
                    table_name="meme_images",
                    expr="SiftFeatureExtractor(data)",
                    using="QDRANT"
                ).df()
                        0
                0	Index faiss_index successfully added to the database
            >>> relation = cursor.table("PDFs")
            >>> relation.order("Similarity(ImageFeatureExtractor(Open('/images/my_meme')), ImageFeatureExtractor(data) ) DESC")
            >>> relation.df()


        """
        stmt = parse_create_vector_index(index_name, table_name, expr, using)
        self._result = execute_statement(self._evadb, stmt)
        return self

    def load(self, file_regex: str, table_name: str, format: str, **kwargs) -> EvaDBQuery:
        """
        Loads data from files into a table.

        Args:
            file_regex (str): Regular expression specifying the files to load.
            table_name (str): Name of the table.
            format (str): File format of the data.
            **kwargs: Additional keyword arguments for configuring the load operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the load query.

        Examples:
            Load the online_video.mp4 file into table named 'youtube_video'.

            >>> cursor.load(file_regex="online_video.mp4", table_name="youtube_video", format="video").df()
                    0
            0	Number of loaded VIDEO: 1

        """
        stmt = parse_load(table_name, file_regex, format, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def drop_table(self, table_name: str, if_exists: bool=True) -> 'EvaDBQuery':
        """
        Drop a table in the database.

        Args:
            table_name (str): Name of the table to be dropped.
            if_exists (bool): If True, do not raise an error if the Table does not already exist. If False, raise an error.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the DROP TABLE.

        Examples:
            Drop table 'sample_table'

            >>> cursor.drop_table("sample_table", if_exists = True).df()
                0
            0	Table Successfully dropped: sample_table
        """
        stmt = parse_drop_table(table_name, if_exists)
        return EvaDBQuery(self._evadb, stmt)

    def drop_function(self, function_name: str, if_exists: bool=True) -> 'EvaDBQuery':
        """
        Drop a function in the database.

        Args:
            function_name (str): Name of the function to be dropped.
            if_exists (bool): If True, do not raise an error if the function does not already exist. If False, raise an error.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the DROP FUNCTION.

        Examples:
            Drop FUNCTION 'ObjectDetector'

            >>> cursor.drop_function("ObjectDetector", if_exists = True)
                0
            0	Function Successfully dropped: ObjectDetector
        """
        stmt = parse_drop_function(function_name, if_exists)
        return EvaDBQuery(self._evadb, stmt)

    def drop_index(self, index_name: str, if_exists: bool=True) -> 'EvaDBQuery':
        """
        Drop an index in the database.

        Args:
            index_name (str): Name of the index to be dropped.
            if_exists (bool): If True, do not raise an error if the index does not already exist. If False, raise an error.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the DROP INDEX.

        Examples:
            Drop the index with name 'faiss_index'

            >>> cursor.drop_index("faiss_index", if_exists = True)
        """
        stmt = parse_drop_index(index_name, if_exists)
        return EvaDBQuery(self._evadb, stmt)

    def create_function(self, function_name: str, if_not_exists: bool=True, impl_path: str=None, type: str=None, **kwargs) -> 'EvaDBQuery':
        """
        Create a function in the database.

        Args:
            function_name (str): Name of the function to be created.
            if_not_exists (bool): If True, do not raise an error if the function already exist. If False, raise an error.
            impl_path (str): Path string to function's implementation.
            type (str): Type of the function (e.g. HuggingFace).
            **kwargs: Additional keyword arguments for configuring the create function operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the function created.

        Examples:
            >>> cursor.create_function("MnistImageClassifier", if_exists = True, 'mnist_image_classifier.py')
                0
            0	Function Successfully created: MnistImageClassifier
        """
        stmt = parse_create_function(function_name, if_not_exists, impl_path, type, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def create_table(self, table_name: str, if_not_exists: bool=True, columns: str=None, **kwargs) -> 'EvaDBQuery':
        '''
        Create a function in the database.

        Args:
            function_name (str): Name of the function to be created.
            if_not_exists (bool): If True, do not raise an error if the function already exist. If False, raise an error.
            impl_path (str): Path string to function's implementation.
            type (str): Type of the function (e.g. HuggingFace).
            **kwargs: Additional keyword arguments for configuring the create function operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the function created.

        Examples:
            >>> cursor.create_table("MyCSV", if_exists = True, columns="""
                    id INTEGER UNIQUE,
                    frame_id INTEGER,
                    video_id INTEGER,
                    dataset_name TEXT(30),
                    label TEXT(30),
                    bbox NDARRAY FLOAT32(4),
                    object_id INTEGER"""
                    )
                0
            0	Table Successfully created: MyCSV
        '''
        stmt = parse_create_table(table_name, if_not_exists, columns, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def query(self, sql_query: str) -> EvaDBQuery:
        """
        Executes a SQL query.

        Args:
            sql_query (str): The SQL query to be executed

        Returns:
            EvaDBQuery: The EvaDBQuery object.

        Examples:
            >>> cursor.query("DROP FUNCTION IF EXISTS SentenceFeatureExtractor;")
            >>> cursor.query('SELECT * FROM sample_table;').df()
               col1  col2
            0     1     2
            1     3     4
            2     5     6
        """
        stmt = parse_query(sql_query)
        return EvaDBQuery(self._evadb, stmt)

    def show(self, object_type: str, **kwargs) -> EvaDBQuery:
        """
        Shows all entries of the current object_type.

        Args:
            show_type (str): The type of SHOW query to be executed
            **kwargs: Additional keyword arguments for configuring the SHOW operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object.

        Examples:
            >>> cursor.show("tables").df()
                name
            0	SampleTable1
            1	SampleTable2
            2	SampleTable3
        """
        stmt = parse_show(object_type, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def explain(self, sql_query: str) -> EvaDBQuery:
        """
        Executes an EXPLAIN query.

        Args:
            sql_query (str): The SQL query to be explained

        Returns:
            EvaDBQuery: The EvaDBQuery object.

        Examples:
            >>> proposed_plan = cursor.explain("SELECT * FROM sample_table;").df()
            >>> for step in proposed_plan[0]:
            >>>   pprint(step)
             |__ ProjectPlan
                |__ SeqScanPlan
                    |__ StoragePlan
        """
        stmt = parse_explain(sql_query)
        return EvaDBQuery(self._evadb, stmt)

    def insert(self, table_name, columns, values, **kwargs) -> EvaDBQuery:
        """
        Executes an INSERT query.

        Args:
            table_name (str): The name of the table to insert into
            columns (list): The list of columns to insert into
            values (list): The list of values to insert
            **kwargs: Additional keyword arguments for configuring the INSERT operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object.

        Examples:
            >>> cursor.insert("sample_table", ["id", "name"], [1, "Alice"])
        """
        stmt = parse_insert(table_name, columns, values, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def rename(self, table_name, new_table_name, **kwargs) -> EvaDBQuery:
        """
        Executes a RENAME query.

        Args:
            table_name (str): The name of the table to rename
            new_table_name (str): The new name of the table
            **kwargs: Additional keyword arguments for configuring the RENAME operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object.

        Examples:
            >>> cursor.show("tables").df()
                name
            0	SampleVideoTable
            1	SampleTable
            2	MyCSV
            3	videotable
            >>> cursor.rename("videotable", "sample_table").df()
            _
            >>> cursor.show("tables").df()
                        name
            0	SampleVideoTable
            1	SampleTable
            2	MyCSV
            3	sample_table

        """
        stmt = parse_rename(table_name, new_table_name, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def close(self):
        """
        Closes the connection.

        Args: None

        Returns:  None

        Examples:
            >>> cursor.close()
        """
        self._evadb.catalog().close()
        ray_enabled = self._evadb.config.get_value('experimental', 'ray')
        if is_ray_enabled_and_installed(ray_enabled):
            import ray
            ray.shutdown()

def sql_predicate_to_expresssion_tree(expr: str) -> AbstractExpression:
    return parse_predicate_expression(expr)

def string_to_lateral_join(expr: str, alias: str):
    return parse_lateral_join(expr, alias)

class EvaDBQuery:

    def __init__(self, evadb: EvaDBDatabase, query_node: Union[AbstractStatement, TableRef], alias: Alias=None):
        self._evadb = evadb
        self._query_node = query_node
        self._alias = alias

    def alias(self, alias: str) -> 'EvaDBQuery':
        """Returns a new Relation with an alias set.

        Args:
            alias (str): an alias name to be set for the Relation.

        Returns:
            EvaDBQuery: Aliased Relation.

        Examples:
            >>> relation = cursor.table("sample_table")
            >>> relation.alias('table')
        """
        self._alias = Alias(alias)

    def cross_apply(self, expr: str, alias: str) -> 'EvaDBQuery':
        """Execute a expr on all the rows of the relation

        Args:
            expr (str): sql expression
            alias (str): alias of the output of the expr

        Returns:
            `EvaDBQuery`: relation

        Examples:

            Runs Yolo on all the frames of the input table

            >>> relation = cursor.table("videos")
            >>> relation.cross_apply("Yolo(data)", "objs(labels, bboxes, scores)")

            Runs Yolo on all the frames of the input table and unnest each object as separate row.

            >>> relation.cross_apply("unnest(Yolo(data))", "obj(label, bbox, score)")
        """
        assert self._query_node.from_table is not None
        table_ref = string_to_lateral_join(expr, alias=alias)
        join_table = TableRef(JoinNode(TableRef(self._query_node, alias=self._alias), table_ref, join_type=JoinType.LATERAL_JOIN))
        self._query_node = SelectStatement(target_list=create_star_expression(), from_table=join_table)
        self._alias = Alias('Relation')
        try_binding(self._evadb.catalog, self._query_node)
        return self

    def df(self, drop_alias: bool=True) -> pandas.DataFrame:
        """
        Execute and fetch all rows as a pandas DataFrame

        Args:
            drop_alias (bool): whether to drop the table name in the output dataframe. Default: True.

        Returns:
            pandas.DataFrame:

        Example:

        Runs a SQL query and get a panda Dataframe.
            >>> cursor.query("SELECT * FROM MyTable;").df()
                col1  col2
            0      1     2
            1      3     4
            2      5     6
        """
        batch = self.execute(drop_alias=drop_alias)
        assert batch.frames is not None, 'relation execute failed'
        return batch.frames

    def execute(self, drop_alias: bool=True) -> Batch:
        """Transform the relation into a result set

        Args:
            drop_alias (bool): whether to drop the table name in the output batch. Default: True.

        Returns:
            Batch: result as evadb Batch

        Example:

            Runs a SQL query and get a Batch
            >>> batch = cursor.query("SELECT * FROM MyTable;").execute()
        """
        result = execute_statement(self._evadb, self._query_node.copy())
        if drop_alias:
            result.drop_column_alias()
        assert result is not None
        return result

    def filter(self, expr: str) -> 'EvaDBQuery':
        """
        Filters rows using the given condition. Multiple filters can be chained using `AND`

        Parameters:
            expr (str): The filter expression.

        Returns:
            EvaDBQuery : Filtered EvaDBQuery.
        Examples:
            >>> relation = cursor.table("sample_table")
            >>> relation.filter("col1 > 10")

            Filter by sql string

            >>> relation.filter("col1 > 10 AND col1 < 20")

        """
        parsed_expr = sql_predicate_to_expresssion_tree(expr)
        self._query_node = handle_select_clause(self._query_node, self._alias, 'where_clause', parsed_expr)
        try_binding(self._evadb.catalog, self._query_node)
        return self

    def limit(self, num: int) -> 'EvaDBQuery':
        """Limits the result count to the number specified.

        Args:
            num (int): Number of records to return. Will return num records or all records if the Relation contains fewer records.

        Returns:
            EvaDBQuery: Relation with subset of records

        Examples:
            >>> relation = cursor.table("sample_table")
            >>> relation.limit(10)

        """
        limit_expr = create_limit_expression(num)
        self._query_node = handle_select_clause(self._query_node, self._alias, 'limit_count', limit_expr)
        try_binding(self._evadb.catalog, self._query_node)
        return self

    def order(self, order_expr: str) -> 'EvaDBQuery':
        """Reorder the relation based on the order_expr

        Args:
            order_expr (str): sql expression to order the relation

        Returns:
            EvaDBQuery: A EvaDBQuery ordered based on the order_expr.

        Examples:
            >>> relation = cursor.table("PDFs")
            >>> relation.order("Similarity(SentenceTransformerFeatureExtractor('When was the NATO created?'), SentenceTransformerFeatureExtractor(data) ) DESC")

        """
        parsed_expr = parse_sql_orderby_expr(order_expr)
        self._query_node = handle_select_clause(self._query_node, self._alias, 'orderby_list', parsed_expr)
        try_binding(self._evadb.catalog, self._query_node)
        return self

    def select(self, expr: str) -> 'EvaDBQuery':
        """
        Projects a set of expressions and returns a new EvaDBQuery.

        Parameters:
            exprs (Union[str, List[str]]): The expression(s) to be selected. If '*' is provided, it expands to all columns in the current EvaDBQuery.

        Returns:
            EvaDBQuery: A EvaDBQuery with subset (or all) of columns.

        Examples:
            >>> relation = cursor.table("sample_table")

            Select all columns in the EvaDBQuery.

            >>> relation.select("*")

            Select all subset of columns in the EvaDBQuery.

            >>> relation.select("col1")
            >>> relation.select("col1, col2")
        """
        parsed_exprs = sql_string_to_expresssion_list(expr)
        self._query_node = handle_select_clause(self._query_node, self._alias, 'target_list', parsed_exprs)
        try_binding(self._evadb.catalog, self._query_node)
        return self

    def show(self) -> pandas.DataFrame:
        """Execute and fetch all rows as a pandas DataFrame

        Returns:
            pandas.DataFrame:
        """
        batch = self.execute()
        assert batch is not None, 'relation execute failed'
        return batch.frames

    def sql_query(self) -> str:
        """Get the SQL query that is equivalent to the relation

        Returns:
            str: the sql query

        Examples:
            >>> relation = cursor.table("sample_table").project('i')
            >>> relation.sql_query()
        """
        return str(self._query_node)

