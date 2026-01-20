# Cluster 106

class FunctionIOCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(FunctionIOCatalog, db_session)

    def get_input_entries_by_function_id(self, function_id: int) -> List[FunctionIOCatalogEntry]:
        try:
            result = self.session.execute(select(self.model).filter(self.model._function_id == function_id, self.model._is_input == True)).scalars().all()
            return [obj.as_dataclass() for obj in result]
        except Exception as e:
            error = f'Getting inputs for function id {function_id} raised {e}'
            logger.error(error)
            raise RuntimeError(error)

    def get_output_entries_by_function_id(self, function_id: int) -> List[FunctionIOCatalogEntry]:
        try:
            result = self.session.execute(select(self.model).filter(self.model._function_id == function_id, self.model._is_input == False)).scalars().all()
            return [obj.as_dataclass() for obj in result]
        except Exception as e:
            error = f'Getting outputs for function id {function_id} raised {e}'
            logger.error(error)
            raise RuntimeError(error)

    def create_entries(self, io_list: List[FunctionIOCatalogEntry]):
        io_objs = []
        for io in io_list:
            io_obj = FunctionIOCatalog(name=io.name, type=io.type, is_nullable=io.is_nullable, array_type=io.array_type, array_dimensions=io.array_dimensions, is_input=io.is_input, function_id=io.function_id)
            io_objs.append(io_obj)
        return io_objs

