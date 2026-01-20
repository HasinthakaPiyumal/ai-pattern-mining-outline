# Cluster 50

class MetaModule(ModelMetaclass):
    """
    MetaModule is a metaclass that automatically registers all subclasses of BaseModule.

    
    Attributes:
        No public attributes
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        Creates a new class and registers it in MODULE_REGISTRY.
        
        Args:
            mcs: The metaclass itself
            name: The name of the class being created
            bases: Tuple of base classes
            namespace: Dictionary containing the class attributes and methods
            **kwargs: Additional keyword arguments
        
        Returns:
            The created class object
        """
        cls = super().__new__(mcs, name, bases, namespace)
        register_module(name, cls)
        return cls

def register_module(cls_name, cls):
    MODULE_REGISTRY.register_module(cls_name=cls_name, cls=cls)

