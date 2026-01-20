# Cluster 19

def _reconstruct_persistent_obj(meta):
    """Hook that is called internally by the `pickle` module to unpickle
    a persistent object.
    """
    meta = dnnlib.EasyDict(meta)
    meta.state = dnnlib.EasyDict(meta.state)
    for hook in _import_hooks:
        meta = hook(meta)
        assert meta is not None
    assert meta.version == _version
    module = _src_to_module(meta.module_src)
    assert meta.type == 'class'
    orig_class = module.__dict__[meta.class_name]
    decorator_class = persistent_class(orig_class)
    obj = decorator_class.__new__(decorator_class)
    setstate = getattr(obj, '__setstate__', None)
    if callable(setstate):
        setstate(meta.state)
    else:
        obj.__dict__.update(meta.state)
    return obj

def persistent_class(orig_class):
    """Class decorator that extends a given class to save its source code
    when pickled.

    Example:

        from torch_utils import persistence

        @persistence.persistent_class
        class MyNetwork(torch.nn.Module):
            def __init__(self, num_inputs, num_outputs):
                super().__init__()
                self.fc = MyLayer(num_inputs, num_outputs)
                ...

        @persistence.persistent_class
        class MyLayer(torch.nn.Module):
            ...

    When pickled, any instance of `MyNetwork` and `MyLayer` will save its
    source code alongside other internal state (e.g., parameters, buffers,
    and submodules). This way, any previously exported pickle will remain
    usable even if the class definitions have been modified or are no
    longer available.

    The decorator saves the source code of the entire Python module
    containing the decorated class. It does *not* save the source code of
    any imported modules. Thus, the imported modules must be available
    during unpickling, also including `torch_utils.persistence` itself.

    It is ok to call functions defined in the same module from the
    decorated class. However, if the decorated class depends on other
    classes defined in the same module, they must be decorated as well.
    This is illustrated in the above example in the case of `MyLayer`.

    It is also possible to employ the decorator just-in-time before
    calling the constructor. For example:

        cls = MyLayer
        if want_to_make_it_persistent:
            cls = persistence.persistent_class(cls)
        layer = cls(num_inputs, num_outputs)

    As an additional feature, the decorator also keeps track of the
    arguments that were used to construct each instance of the decorated
    class. The arguments can be queried via `obj.init_args` and
    `obj.init_kwargs`, and they are automatically pickled alongside other
    object state. This feature can be disabled on a per-instance basis
    by setting `self._record_init_args = False` in the constructor.

    A typical use case is to first unpickle a previous instance of a
    persistent class, and then upgrade it to use the latest version of
    the source code:

        with open('old_pickle.pkl', 'rb') as f:
            old_net = pickle.load(f)
        new_net = MyNetwork(*old_obj.init_args, **old_obj.init_kwargs)
        misc.copy_params_and_buffers(old_net, new_net, require_all=True)
    """
    assert isinstance(orig_class, type)
    if is_persistent(orig_class):
        return orig_class
    assert orig_class.__module__ in sys.modules
    orig_module = sys.modules[orig_class.__module__]
    orig_module_src = _module_to_src(orig_module)

    class Decorator(orig_class):
        _orig_module_src = orig_module_src
        _orig_class_name = orig_class.__name__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            record_init_args = getattr(self, '_record_init_args', True)
            self._init_args = copy.deepcopy(args) if record_init_args else None
            self._init_kwargs = copy.deepcopy(kwargs) if record_init_args else None
            assert orig_class.__name__ in orig_module.__dict__
            _check_pickleable(self.__reduce__())

        @property
        def init_args(self):
            assert self._init_args is not None
            return copy.deepcopy(self._init_args)

        @property
        def init_kwargs(self):
            assert self._init_kwargs is not None
            return dnnlib.EasyDict(copy.deepcopy(self._init_kwargs))

        def __reduce__(self):
            fields = list(super().__reduce__())
            fields += [None] * max(3 - len(fields), 0)
            if fields[0] is not _reconstruct_persistent_obj:
                meta = dict(type='class', version=_version, module_src=self._orig_module_src, class_name=self._orig_class_name, state=fields[2])
                fields[0] = _reconstruct_persistent_obj
                fields[1] = (meta,)
                fields[2] = None
            return tuple(fields)
    Decorator.__name__ = orig_class.__name__
    Decorator.__module__ = orig_class.__module__
    _decorators.add(Decorator)
    return Decorator

