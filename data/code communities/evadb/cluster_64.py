# Cluster 64

class Parser(object):
    """
    Parser based on EvaDB grammar: evadb.lark
    """
    _lark_parser = None

    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Parser, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._lark_parser = LarkParser()
        self._initialized = True

    def parse(self, query_string: str) -> list:
        lark_output = self._lark_parser.parse(query_string)
        return lark_output

