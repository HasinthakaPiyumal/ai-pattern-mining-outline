# Cluster 43

class FactorioNamespace:

    def __init__(self, instance, agent_index):
        self.logging_results = {}
        self.line_value = 0
        self.persistent_vars = {}
        self.instance = instance
        self.tcp_port = instance.tcp_port
        self.max_sequential_exception_count = 1
        self._sequential_exception_count = 0
        self.log_counter = 0
        self.player_location = ent.Position(x=0, y=0)
        self.agent_index = agent_index
        self.agent_id = str(agent_index)
        self.loop_context = LoopContext()
        for name in dir(builtins):
            if not name.startswith('_'):
                try:
                    setattr(self, name, getattr(builtins, name))
                    self.persistent_vars[name] = getattr(builtins, name)
                except Exception as e:
                    print(f'Failed to add builtin {name}: {e}')

        def assert_(expr, msg=None):
            if not expr:
                raise AssertionError(msg)
        self.essential_builtins = {'print': print, 'len': len, 'range': range, 'int': int, 'str': str, 'float': float, 'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple, 'set': set, 'sum': sum, 'min': min, 'max': max, 'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter, 'any': any, 'all': all, 'sorted': sorted, 'reversed': reversed, 'round': round, 'abs': abs, 'isinstance': isinstance, 'type': type, 'assert': assert_}
        for name, func in self.essential_builtins.items():
            setattr(self, name, func)
            self.persistent_vars[name] = func
        self.capture_whole_output = False
        self.execution_trace = True
        self.Prototype = Prototype
        self.Resource = Resource
        self.Direction = ent.Direction
        self.Position = ent.Position
        self.EntityStatus = ent.EntityStatus
        self.BoundingBox = ent.BoundingBox
        self.BuildingBox = ent.BuildingBox
        self.BeltGroup = ent.BeltGroup
        self.Technology = Technology
        self.Recipe = ent.Recipe
        self.PipeGroup = ent.PipeGroup
        self.ElectricityGroup = ent.ElectricityGroup
        self.BeltGroup = ent.BeltGroup
        self.Pipe = ent.Pipe
        self.RecipeName = RecipeName
        self.prototype_by_name = prototype_by_name
        for name, value in self.prototype_by_name.items():
            if value.entity_class:
                setattr(self, value.name, value.entity_class)
        self.UP, self.ABOVE, self.TOP = [ent.Direction.UP] * 3
        self.RIGHT, self.EAST = [ent.Direction.RIGHT] * 2
        self.LEFT, self.WEST = [ent.Direction.LEFT] * 2
        self.DOWN, self.BELOW, self.BOTTOM = [ent.Direction.DOWN] * 3
        self.sqrt = math.sqrt
        self.sin = math.sin
        self.cos = math.cos
        self.tan = math.tan
        self.pi = math.pi
        self.floor = math.floor
        self.ceil = math.ceil
        self.abs = abs
        self.pow = pow
        self.Optional = Optional
        self.Union = Union
        self.List = List
        self.Dict = Dict
        self.Tuple = Tuple
        self.Set = Set
        entity_module = sys.modules[Entity.__module__]
        entity_classes = inspect.getmembers(entity_module, lambda member: inspect.isclass(member) and issubclass(member, BaseModel) and (member != BaseModel))
        for name, entity_class in entity_classes:
            setattr(self, name, entity_class)
        self._static_members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and (not attr.startswith('__'))]

    def get_functions(self) -> List[SerializableFunction]:
        """
        Gets all defined functions mapped from their attribute name in the namespace.
        @return:
        """
        return list(filter(lambda x: isinstance(x, SerializableFunction), self.persistent_vars.values()))

    def load(self, namespace_str: bytes):
        try:
            env = pickle.loads(namespace_str)
            for key, value in env.items():
                restored_value = unwrap_after_deserialization(self, value)
                self.persistent_vars[key] = restored_value
                setattr(self, key, restored_value)
        except Exception as e:
            print(f'Error restoring namespace: {e}')
            pass

    def _assign_target(self, target, value, eval_dict):
        """Helper function to handle different types of assignment targets"""
        if isinstance(target, ast.Name):
            eval_dict[target.id] = value
        elif isinstance(target, ast.Tuple):
            if not isinstance(value, (tuple, list)) or len(target.elts) != len(value):
                raise ValueError('Cannot unpack - length mismatch')
            for t, v in zip(target.elts, value):
                self._assign_target(t, v, eval_dict)
        elif isinstance(target, ast.List):
            if not isinstance(value, (tuple, list)) or len(target.elts) != len(value):
                raise ValueError('Cannot unpack - length mismatch')
            for t, v in zip(target.elts, value):
                self._assign_target(t, v, eval_dict)
        else:
            raise SyntaxError(f'Unsupported target type: {type(target)}')

    def reset(self):
        """
        Delete all variables that have accrued in the namespace by the agent, except for preexisting members
        @return:
        """
        for attr in dir(self):
            if not callable(getattr(self, attr)) and attr[0] != '_' and (attr not in self._static_members):
                self[attr] = None

    def __getitem__(self, key):
        if key not in dir(self) or key.startswith('__'):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def log(self, *arg):
        if self.execution_trace:
            self.log_counter += 1
            self.logging_results[self.log_counter] = [(self.line_value, repr(arg))]
        else:
            if self.line_value not in self.logging_results:
                self.logging_results[self.line_value] = []
            self.logging_results[self.line_value].append(repr(arg))
        return None

    def _get_suggestions_from_name_error(self, eval_dict, error_msg) -> List[Tuple[str, str]]:
        var_name = error_msg.split("'")[1]
        available_vars = {}
        for name, value in eval_dict.items():
            if not name.startswith('_'):
                available_vars[name] = value
        for name in dir(self):
            if not name.startswith('_'):
                available_vars[name] = getattr(self, name)
        matches = get_close_matches(var_name, available_vars.keys(), n=3, cutoff=0.6)
        suggestions = []
        for match in matches:
            value = available_vars[match]
            type_hint = get_value_type_str(value)
            suggestions.append((match, type_hint))
        return suggestions

    def _extract_error_lines(self, expr, traceback_str):
        lines = expr.splitlines()
        error_lines = []
        for line in traceback_str.splitlines():
            if 'File "file", line' in line:
                line_num = int(line.split(', line')[1].split(',')[0])
                if 1 <= line_num <= len(lines):
                    error_lines.append((line_num, lines[line_num - 1].strip()))
        return error_lines

    def _change_print_to_log(self, node):
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and (node.value.func.id == 'print'):
                node.value.func.id = 'log'
            elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and (node.value.func.id == 'time.sleep'):
                node.value.func.id = 'sleep'
        elif isinstance(node, ast.If) or isinstance(node, ast.For) or isinstance(node, ast.While):
            for subnode_idx, subnode in enumerate(node.body):
                node.body[subnode_idx] = self._change_print_to_log(subnode)
            for subnode_idx, subnode in enumerate(node.orelse):
                node.orelse[subnode_idx] = self._change_print_to_log(subnode)
        elif isinstance(node, ast.FunctionDef):
            for subnode_idx, subnode in enumerate(node.body):
                node.body[subnode_idx] = self._change_print_to_log(subnode)
        return node

    def execute_body(self, body, eval_dict, parent_node=None):
        """Execute a sequence of nodes while maintaining line numbers"""
        for n in body:
            n = self._change_print_to_log(n)
            result = self.execute_node(n, eval_dict, parent_node)
            if isinstance(result, bool) and (not result):
                if self.loop_context.state in ('BREAK', 'CONTINUE'):
                    return False
            elif isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                return result
        return True

    def execute_node(self, node, eval_dict, parent_node=None):
        """
        Helper function to execute a single AST node
        Returns: True for normal execution, False or string for control flow changes
        """

        def process_annotation(annotation, eval_dict):
            """Process a type annotation node and return the evaluated type"""
            if annotation is None:
                return None
            if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
                return annotation.value
            if isinstance(annotation, ast.Name):
                return eval_dict.get(annotation.id, annotation.id)
            if isinstance(annotation, ast.Subscript):
                base_type = process_annotation(annotation.value, eval_dict)
                if isinstance(annotation.slice, ast.Tuple):
                    type_args = [process_annotation(arg, eval_dict) for arg in annotation.slice.elts]
                    return f'{base_type}[{', '.join(map(str, type_args))}]'
                else:
                    type_arg = process_annotation(annotation.slice, eval_dict)
                    return f'{base_type}[{type_arg}]'
            try:
                compiled = compile(ast.Expression(annotation), 'annotation', 'eval')
                return eval(compiled, eval_dict)
            except Exception:
                return ast.unparse(annotation)
        if hasattr(node, 'lineno'):
            self.line_value = node.lineno
        if isinstance(node, ast.Break):
            return self.loop_context.handle_break()
        elif isinstance(node, ast.Continue):
            return self.loop_context.handle_continue()
        elif isinstance(node, ast.For):
            try:
                self.loop_context.enter_loop(node)
                iter_obj = eval(compile(ast.Expression(node.iter), 'file', 'eval'), eval_dict)
                for item in iter_obj:
                    self._assign_target(node.target, item, eval_dict)
                    result = self.execute_body(node.body, eval_dict, node)
                    if isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                        return result
                    if self.loop_context.state == 'BREAK':
                        break
                    if self.loop_context.state == 'CONTINUE':
                        self.loop_context.state = 'NORMAL'
                        continue
                if node.orelse and self.loop_context.state != 'BREAK':
                    result = self.execute_body(node.orelse, eval_dict, node)
                    if isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                        return result
                return True
            finally:
                self.loop_context.exit_loop()
        elif isinstance(node, ast.While):
            self.loop_context.enter_loop(node)
            try:
                while eval(compile(ast.Expression(node.test), 'file', 'eval'), eval_dict):
                    result = self.execute_body(node.body, eval_dict, node)
                    if isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                        return result
                    if self.loop_context.state == 'BREAK':
                        break
                    if self.loop_context.state == 'CONTINUE':
                        self.loop_context.state = 'NORMAL'
                        continue
                if node.orelse and self.loop_context.state != 'BREAK':
                    result = self.execute_body(node.orelse, eval_dict, node)
                    if isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                        return result
                return True
            finally:
                self.loop_context.exit_loop()
        elif isinstance(node, ast.If):
            test_result = eval(compile(ast.Expression(node.test), 'file', 'eval'), eval_dict)
            if test_result:
                result = self.execute_body(node.body, eval_dict, node)
                if isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                    return result
            elif node.orelse:
                result = self.execute_body(node.orelse, eval_dict, node)
                if isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                    return result
            return True
        elif isinstance(node, ast.FunctionDef):
            try:
                return_annotation = process_annotation(node.returns, eval_dict) if node.returns else None
                arg_annotations = {}
                for arg in node.args.args:
                    if arg.annotation:
                        arg_annotations[arg.arg] = process_annotation(arg.annotation, eval_dict)
                for arg in node.args.kwonlyargs:
                    if arg.annotation:
                        arg_annotations[arg.arg] = process_annotation(arg.annotation, eval_dict)
                for arg in getattr(node.args, 'posonlyargs', []):
                    if arg.annotation:
                        arg_annotations[arg.arg] = process_annotation(arg.annotation, eval_dict)
                if node.args.vararg and node.args.vararg.annotation:
                    arg_annotations['*' + node.args.vararg.arg] = process_annotation(node.args.vararg.annotation, eval_dict)
                if node.args.kwarg and node.args.kwarg.annotation:
                    arg_annotations['**' + node.args.kwarg.arg] = process_annotation(node.args.kwarg.annotation, eval_dict)
                setattr(node, '__annotations__', {'return': return_annotation, 'args': arg_annotations})
                function_namespace = {**self.essential_builtins, **eval_dict}
                wrapped_node = ast.Module([node], type_ignores=[])
                compiled = compile(wrapped_node, 'file', 'exec')
                exec(compiled, function_namespace, eval_dict)
                func = function_namespace[node.name]
                if hasattr(node, '__annotations__'):
                    func.__annotations__ = getattr(node, '__annotations__')
                serialized_func = SerializableFunction(func, self)
                self.persistent_vars[node.name] = serialized_func
                setattr(self, node.name, serialized_func)
                eval_dict[node.name] = serialized_func
                return True
            except Exception:
                compiled = compile(ast.Module([node], type_ignores=[]), 'file', 'exec')
                exec(compiled, eval_dict)
                if node.name in eval_dict:
                    func = eval_dict[node.name]
                    if callable(func):
                        self.persistent_vars[node.name] = wrap_for_serialization(func)
                        setattr(self, node.name, func)
                return True
        elif isinstance(node, ast.Assign):
            original_keys = set(eval_dict.keys())
            compiled = compile(ast.Module([node], type_ignores=[]), 'file', 'exec')
            exec(compiled, eval_dict)
            new_or_updated_keys = set()
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            new_or_updated_keys.update(targets)
            current_keys = set(eval_dict.keys())
            new_keys = current_keys - original_keys
            new_or_updated_keys.update(new_keys)
            for name in new_or_updated_keys:
                if name in eval_dict and (not name.startswith('_')):
                    value = eval_dict[name]
                    self.persistent_vars[name] = wrap_for_serialization(value)
                    setattr(self, name, value)
            return True
        elif isinstance(node, ast.AnnAssign):
            if node.value:
                compiled = compile(ast.Module([node], type_ignores=[]), 'file', 'exec')
                exec(compiled, eval_dict)
                if isinstance(node.target, ast.Name):
                    name = node.target.id
                    if name in eval_dict:
                        value = eval_dict[name]
                        self.persistent_vars[name] = wrap_for_serialization(value)
                        setattr(self, name, value)
            return True
        elif isinstance(node, ast.AugAssign):
            compiled = compile(ast.Module([node], type_ignores=[]), 'file', 'exec')
            exec(compiled, eval_dict)
            if isinstance(node.target, ast.Name):
                name = node.target.id
                if name in eval_dict:
                    value = eval_dict[name]
                    self.persistent_vars[name] = wrap_for_serialization(value)
                    setattr(self, name, value)
            elif isinstance(node.target, (ast.Attribute, ast.Subscript)):
                target_vars = set()

                def extract_names(node):
                    if isinstance(node, ast.Name):
                        target_vars.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        extract_names(node.value)
                    elif isinstance(node, ast.Subscript):
                        extract_names(node.value)
                        extract_names(node.slice)
                extract_names(node.target)
                for name in target_vars:
                    if name in eval_dict and (not name.startswith('_')):
                        value = eval_dict[name]
                        self.persistent_vars[name] = wrap_for_serialization(value)
                        setattr(self, name, value)
            return True
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                func_name = node.value.func.id
                if func_name in eval_dict:
                    func = eval_dict[func_name]
                    args = []
                    kwargs = {}
                    for arg in node.value.args:
                        args.append(eval(compile(ast.Expression(arg), 'file', 'eval'), eval_dict, eval_dict))
                    for keyword in node.value.keywords:
                        key = keyword.arg
                        value = eval(compile(ast.Expression(keyword.value), 'file', 'eval'), eval_dict, eval_dict)
                        kwargs[key] = value
                    response = func(*args, **kwargs)
                    if isinstance(func, SerializableFunction) and hasattr(func, '_cached_func') and func._cached_func:
                        func_globals = func._cached_func.__globals__
                        for name, value in func_globals.items():
                            if not name.startswith('_') and name in eval_dict:
                                if eval_dict[name] != value:
                                    eval_dict[name] = value
                                    self.persistent_vars[name] = wrap_for_serialization(value)
                                    setattr(self, name, value)
            else:
                compiled = compile(ast.Expression(node.value), 'file', 'eval')
                response = eval(compiled, eval_dict, eval_dict)
            if self.capture_whole_output:
                if not (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and (node.value.func.id == 'print')):
                    if response is not True and response is not None and (not isinstance(node.value, ast.Constant)):
                        self._sequential_exception_count = 0
                        self.log(response)
            return True
        elif isinstance(node, ast.Return):
            if node.value:
                return_value = eval(compile(ast.Expression(node.value), 'file', 'eval'), eval_dict)
                return ('RETURN', return_value)
            else:
                return ('RETURN', None)
        elif isinstance(node, ast.Raise):
            if node.exc:
                exception = eval(compile(ast.Expression(node.exc), 'file', 'eval'), eval_dict)
                if node.cause:
                    cause = eval(compile(ast.Expression(node.cause), 'file', 'eval'), eval_dict)
                    raise exception from cause
                else:
                    raise exception
            else:
                raise
        elif isinstance(node, ast.Assert):
            test_result = eval(compile(ast.Expression(node.test), 'file', 'eval'), eval_dict)
            if not test_result:
                if node.msg:
                    msg = eval(compile(ast.Expression(node.msg), 'file', 'eval'), eval_dict)
                    raise AssertionError(msg)
                else:
                    raise AssertionError()
            return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                try:
                    parts = alias.name.split('.')
                    top_module = __import__(alias.name)
                    if alias.asname:
                        final_module = top_module
                        for part in parts[1:]:
                            final_module = getattr(final_module, part)
                        eval_dict[alias.asname] = final_module
                        self.persistent_vars[alias.asname] = final_module
                        setattr(self, alias.asname, final_module)
                    else:
                        top_name = parts[0]
                        eval_dict[top_name] = top_module
                        self.persistent_vars[top_name] = top_module
                        setattr(self, top_name, top_module)
                except ImportError:
                    raise
            return True
        elif isinstance(node, ast.ImportFrom):
            try:
                module_name = node.module if node.module else ''
                level = node.level if node.level else 0
                if level > 0:
                    compiled = compile(ast.Module([node], type_ignores=[]), 'file', 'exec')
                    exec(compiled, eval_dict)
                elif node.names[0].name == '*':
                    compiled = compile(ast.Module([node], type_ignores=[]), 'file', 'exec')
                    exec(compiled, eval_dict)
                    protected_names = {'log', 'print'}
                    for name, value in eval_dict.items():
                        if not name.startswith('_') and name not in self.persistent_vars and (name not in protected_names):
                            self.persistent_vars[name] = value
                            setattr(self, name, value)
                    for name in protected_names:
                        if hasattr(self, name):
                            eval_dict[name] = getattr(self, name)
                else:
                    imported_names = [alias.name for alias in node.names]
                    module = __import__(module_name, fromlist=imported_names)
                    protected_names = {'log', 'print'}
                    for alias in node.names:
                        obj = getattr(module, alias.name)
                        name = alias.asname if alias.asname else alias.name
                        if name not in protected_names:
                            eval_dict[name] = obj
                            self.persistent_vars[name] = obj
                            setattr(self, name, obj)
            except ImportError:
                raise
            return True
        elif isinstance(node, ast.Global):
            return True
        elif isinstance(node, ast.Nonlocal):
            for name in node.names:
                pass
            compiled = compile(ast.Module([node], type_ignores=[]), 'file', 'exec')
            exec(compiled, eval_dict)
            return True
        elif isinstance(node, ast.Try):
            try:
                result = self.execute_body(node.body, eval_dict, node)
                if isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                    return result
            except Exception as e:
                handled = False
                for handler in node.handlers:
                    if handler.type is None or isinstance(e, eval(compile(ast.Expression(handler.type), 'file', 'eval'), eval_dict)):
                        if handler.name:
                            eval_dict[handler.name] = e
                        result = self.execute_body(handler.body, eval_dict, handler)
                        if isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                            return result
                        handled = True
                        break
                if not handled:
                    raise
            else:
                if node.orelse:
                    result = self.execute_body(node.orelse, eval_dict, node)
                    if isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                        return result
            finally:
                if node.finalbody:
                    result = self.execute_body(node.finalbody, eval_dict, node)
                    if isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                        return result
            return True
        else:
            compiled = compile(ast.Module([node], type_ignores=[]), 'file', 'exec')
            exec(compiled, eval_dict)
            return True

    def eval_with_timeout(self, expr):
        """
        Executes a Python expression with a timeout and returns the result.
        Supports try-except blocks, type annotations, and nested control flows.
        """

        def parse_result_into_str(data, max_lines=64):
            result = []
            for key, values in data.items():
                if self.execution_trace:
                    for line_no, value in values:
                        result.append(f'{line_no}: {value}')
                else:
                    for value in values:
                        result.append(f'{key}: {value}')
            if len(result) > max_lines:
                result = [f'{len(result) - max_lines} lines truncated...'] + result[max_lines:]
            return '\n'.join(result)

        def find_actual_line_number(node, code_lines):
            """Find the actual line number in the source code for a given node"""
            if not hasattr(node, 'lineno'):
                return 0
            return node.lineno
        tree = ast.parse(expr)
        self.logging_results = {}
        self.line_value = 0
        self.loop_context = LoopContext()
        eval_dict = {**{name: getattr(builtins, name) for name in dir(builtins) if not name.startswith('_')}, **{name: getattr(self, name) for name in dir(self) if not name.startswith('_')}, **self.persistent_vars}
        for key, value in eval_dict.items():
            if isinstance(value, SerializableFunction):
                eval_dict[key] = value.bind(self)
        last_successful_state = None
        for index, node in enumerate(tree.body):
            try:
                node = self._change_print_to_log(node)
                result = self.execute_node(node, eval_dict)
                if isinstance(result, tuple) and len(result) == 2 and (result[0] == 'RETURN'):
                    return_value = result[1]
                    if return_value is not None:
                        self.log(return_value)
                    break
                last_successful_state = dict(self.persistent_vars)
            except (Exception, NameError) as e:
                self._sequential_exception_count += 1
                error_traceback = traceback.format_exc()
                error_lines = self._extract_error_lines(expr, error_traceback)
                error_message = ''
                if error_lines:
                    error_message += 'Error occurred:\n'
                    for line_num, line_content in error_lines:
                        error_message += f'  Line {line_num}: {line_content}\n'
                error_type = error_traceback.strip().split('\n')[-1]
                if isinstance(e, NameError) and "name '" in str(e) and ("' is not defined" in str(e)):
                    suggestions = [f'{sug} ({_type})' for sug, _type in self._get_suggestions_from_name_error(eval_dict, str(e))]
                    error_message += f'\n{error_type}'
                    if suggestions:
                        error_message += f'\nDid you mean one of these?\n{suggestions}'
                else:
                    error_message += f'\n{error_type}'
                self.log(error_message)
                if last_successful_state is not None:
                    self.persistent_vars = last_successful_state.copy()
                break
            eval_dict.update(self.persistent_vars)
            for key, value in eval_dict.items():
                if isinstance(value, SerializableFunction):
                    eval_dict[key] = value.bind(self)
        score, goal = self.score()
        result_output = parse_result_into_str(self.logging_results)
        return (score, goal, result_output)

    def get_messages(self) -> List[Dict]:
        return []

    def load_messages(self, messages: List[Dict]):
        pass

def wrap_for_serialization(value):
    """Wrap values that need special serialization handling"""
    if isinstance(value, types.FunctionType):
        if value.__module__ == 'builtins':
            return value
        return SerializableFunction(value)
    return value

