# Cluster 20

def BuildTypeFromAST(node, namespace):
    if isinstance(node, ast.Name):
        if node.id not in namespace:
            raise TypeError(f'BuildTypeFromAST(): Unsupported type {str(node.id)}.')
        return namespace[node.id]
    elif isinstance(node, ast.Attribute):
        attrChain = []
        current = node
        while isinstance(current, ast.Attribute):
            attrChain.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            attrChain.append(current.id)
        attrChain.reverse()
        baseModule = namespace.get(attrChain[0], None)
        if baseModule is None:
            if attrChain[:3] == ['ailice', 'common', 'ADataType']:
                baseModule = importlib.import_module(attrChain[0])
            else:
                raise TypeError(f'BuildTypeFromAST(): Unsupported type {str(attrChain)}.')
        currentObj = baseModule
        for i in range(1, len(attrChain)):
            currentObj = getattr(currentObj, attrChain[i])
        return currentObj
    elif isinstance(node, ast.Subscript):
        container = BuildTypeFromAST(node.value, namespace)
        args = node.slice
        if isinstance(args, ast.Tuple):
            type_args = [BuildTypeFromAST(arg, namespace) for arg in args.elts]
            return container[tuple(type_args)]
        else:
            type_arg = BuildTypeFromAST(args, namespace)
            return container[type_arg]
    elif isinstance(node, ast.Tuple):
        return tuple((BuildTypeFromAST(elt, namespace) for elt in node.elts))
    elif isinstance(node, ast.Constant) and node.value is None:
        return None
    else:
        raise TypeError(f'BuildTypeFromAST(): Unsupported type {str(node)}.')

