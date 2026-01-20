# Cluster 10

def recurse(klass):
    for subclass in klass.__subclasses__():
        subclass_list.append(subclass)
        recurse(subclass)

def get_all_subclasses(cls):
    subclass_list = []

    def recurse(klass):
        for subclass in klass.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)
    recurse(cls)
    return set(subclass_list)

@pytest.mark.notparallel
class PlanNodeTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_create_plan(self):
        dummy_info = TableInfo('dummy')
        columns = [ColumnCatalogEntry('id', ColumnType.INTEGER), ColumnCatalogEntry('name', ColumnType.TEXT, array_dimensions=[50])]
        dummy_plan_node = CreatePlan(dummy_info, columns, False)
        self.assertEqual(dummy_plan_node.opr_type, PlanOprType.CREATE)
        self.assertEqual(dummy_plan_node.if_not_exists, False)
        self.assertEqual(dummy_plan_node.table_info.table_name, 'dummy')
        self.assertEqual(dummy_plan_node.column_list[0].name, 'id')
        self.assertEqual(dummy_plan_node.column_list[1].name, 'name')

    def test_rename_plan(self):
        dummy_info = TableInfo('old')
        dummy_old = TableRef(dummy_info)
        dummy_new = TableInfo('new')
        dummy_plan_node = RenamePlan(dummy_old, dummy_new)
        self.assertEqual(dummy_plan_node.opr_type, PlanOprType.RENAME)
        self.assertEqual(dummy_plan_node.old_table.table.table_name, 'old')
        self.assertEqual(dummy_plan_node.new_name.table_name, 'new')

    def test_insert_plan(self):
        video_id = 0
        column_ids = [0, 1]
        expression = type('AbstractExpression', (), {'evaluate': lambda: 1})
        values = [expression, expression]
        dummy_plan_node = InsertPlan(video_id, column_ids, values)
        self.assertEqual(dummy_plan_node.opr_type, PlanOprType.INSERT)

    def test_create_function_plan(self):
        function_name = 'function'
        or_replace = False
        if_not_exists = True
        functionIO = 'functionIO'
        inputs = [functionIO, functionIO]
        outputs = [functionIO]
        impl_path = 'test'
        ty = 'classification'
        node = CreateFunctionPlan(function_name, or_replace, if_not_exists, inputs, outputs, impl_path, ty)
        self.assertEqual(node.opr_type, PlanOprType.CREATE_FUNCTION)
        self.assertEqual(node.or_replace, or_replace)
        self.assertEqual(node.if_not_exists, if_not_exists)
        self.assertEqual(node.inputs, [functionIO, functionIO])
        self.assertEqual(node.outputs, [functionIO])
        self.assertEqual(node.impl_path, impl_path)
        self.assertEqual(node.function_type, ty)

    def test_drop_object_plan(self):
        object_type = ObjectType.TABLE
        function_name = 'function'
        if_exists = True
        node = DropObjectPlan(object_type, function_name, if_exists)
        self.assertEqual(node.opr_type, PlanOprType.DROP_OBJECT)
        self.assertEqual(node.if_exists, True)
        self.assertEqual(node.object_type, ObjectType.TABLE)

    def test_load_data_plan(self):
        table_info = 'info'
        file_path = 'test.mp4'
        file_format = FileFormatType.VIDEO
        file_options = {}
        file_options['file_format'] = file_format
        column_list = None
        batch_mem_size = 3000
        plan_str = 'LoadDataPlan(table_id={}, file_path={},             column_list={},             file_options={},             batch_mem_size={})'.format(table_info, file_path, column_list, file_options, batch_mem_size)
        plan = LoadDataPlan(table_info, file_path, column_list, file_options, batch_mem_size)
        self.assertEqual(plan.opr_type, PlanOprType.LOAD_DATA)
        self.assertEqual(plan.table_info, table_info)
        self.assertEqual(plan.file_path, file_path)
        self.assertEqual(plan.batch_mem_size, batch_mem_size)
        self.assertEqual(str(plan), plan_str)

    def test_union_plan(self):
        all = True
        plan = UnionPlan(all)
        self.assertEqual(plan.opr_type, PlanOprType.UNION)
        self.assertEqual(plan.all, all)

    def test_abstract_plan_str(self):
        derived_plan_classes = list(get_all_subclasses(AbstractPlan))
        for derived_plan_class in derived_plan_classes:
            sig = signature(derived_plan_class.__init__)
            params = sig.parameters
            plan_dict = {}
            if isabstract(derived_plan_class) is False:
                obj = get_mock_object(derived_plan_class, len(params))
                plan_dict[obj] = obj

def get_mock_object(class_type, number_of_args):
    if number_of_args == 1:
        return class_type()
    elif number_of_args == 2:
        return class_type(MagicMock())
    elif number_of_args == 3:
        return class_type(MagicMock(), MagicMock())
    elif number_of_args == 4:
        return class_type(MagicMock(), MagicMock(), MagicMock())
    elif number_of_args == 5:
        return class_type(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    elif number_of_args == 6:
        return class_type(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    elif number_of_args == 7:
        return class_type(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    elif number_of_args == 8:
        return class_type(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    elif number_of_args == 9:
        return class_type(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    elif number_of_args == 10:
        return class_type(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    elif number_of_args == 11:
        return class_type(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    elif number_of_args == 12:
        return class_type(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    elif number_of_args == 13:
        return class_type(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    else:
        raise Exception('Too many args')

class AbstractExecutorTest(unittest.TestCase):

    def test_constructor_args(self):
        derived_executor_classes = list(get_all_subclasses(AbstractExecutor))
        for derived_executor_class in derived_executor_classes:
            sig = signature(derived_executor_class.__init__)
            params = sig.parameters
            self.assertTrue(len(params) < 10)

class AbstractFunctionTest(unittest.TestCase):

    def test_function_abstract_functions(self):
        derived_function_classes = list(get_all_subclasses(AbstractFunction))
        for derived_function_class in derived_function_classes:
            if issubclass(derived_function_class, (Yolo, AbstractHFFunction)):
                continue
            if isabstract(derived_function_class) is False:
                class_type = derived_function_class
                sig = inspect.signature(class_type.__init__)
                params = sig.parameters
                len_params = len(params)
                if 'kwargs' in params:
                    len_params = len_params - 1
                if 'args' in params:
                    len_params = len_params - 1
                dummy_object = get_mock_object(class_type, len_params)
                self.assertTrue(str(dummy_object.name) is not None)

    def test_all_classes(self):

        def get_all_classes(module, level):
            class_list = []
            if level == 4:
                return []
            for _, obj in inspect.getmembers(module):
                if inspect.ismodule(obj):
                    sublist = get_all_classes(obj, level + 1)
                    if sublist != []:
                        class_list.append(sublist)
                elif inspect.isclass(obj):
                    if inspect.isabstract(obj) is True:
                        sublist = get_all_classes(obj, level + 1)
                        if sublist != []:
                            class_list.append(sublist)
                    elif inspect.isbuiltin(obj) is False:
                        try:
                            source_file = inspect.getsourcefile(obj)
                            if source_file is None:
                                continue
                            if issubclass(obj, Enum):
                                continue
                            if 'python' not in str(source_file):
                                class_list.append([obj])
                        except OSError:
                            pass
                        except TypeError:
                            pass
            flat_class_list = [item for sublist in class_list for item in sublist]
            return set(flat_class_list)
        class_list = get_all_classes(evadb, 1)
        base_id = 0
        ref_object = None
        for class_type in class_list:
            base_id = base_id + 1
            sig = inspect.signature(class_type.__init__)
            params = sig.parameters
            len_params = len(params)
            if 'kwargs' in params:
                len_params = len_params - 1
            if 'args' in params:
                len_params = len_params - 1
            try:
                dummy_object = get_mock_object(class_type, len_params)
            except Exception:
                continue
            if base_id == 0:
                ref_object = dummy_object
            else:
                self.assertEqual(dummy_object, dummy_object)
                self.assertNotEqual(ref_object, dummy_object)
            try:
                inspect.signature(class_type.name)
            except Exception:
                continue
            name_str = dummy_object.name()
            self.assertTrue(name_str is not None)

