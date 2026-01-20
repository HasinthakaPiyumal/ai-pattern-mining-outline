# Cluster 46

class SqlVisibility(TypeDecorator):
    """Sql type for Visibility."""
    impl = String

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        """Inherited, see superclass."""
        return dialect.type_descriptor(String(8))

    def process_bind_param(self, value: Optional[Visibility], dialect: Dialect) -> Any:
        """Inherited, see superclass."""
        if not value:
            value = Visibility.unknown
        return value.value

    def process_result_value(self, value: Optional[str], dialect: Dialect) -> Visibility:
        """Inherited, see superclass."""
        if not value:
            return Visibility.unknown
        return Visibility(value)

