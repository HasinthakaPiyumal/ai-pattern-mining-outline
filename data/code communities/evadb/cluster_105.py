# Cluster 105

class IndexCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(IndexCatalog, db_session)

    def insert_entry(self, name: str, save_file_path: str, type: VectorStoreType, feat_column: ColumnCatalogEntry, function_signature: str, index_def: str) -> IndexCatalogEntry:
        index_entry = IndexCatalog(name, save_file_path, type, feat_column.row_id, function_signature, index_def)
        index_entry = index_entry.save(self.session)
        return index_entry.as_dataclass()

    def get_entry_by_name(self, name: str) -> IndexCatalogEntry:
        try:
            entry = self.query.filter(self.model._name == name).one()
            return entry.as_dataclass()
        except NoResultFound:
            return None

    def get_entry_by_id(self, id: int) -> IndexCatalogEntry:
        try:
            entry = self.query.filter(self.model._row_id == id).one()
            return entry.as_dataclass()
        except NoResultFound:
            return None

    def get_entry_by_column_and_function_signature(self, column: ColumnCatalogEntry, function_signature: str):
        try:
            entry = self.query.filter(self.model._feat_column_id == column.row_id, self.model._function_signature == function_signature).one()
            return entry.as_dataclass()
        except NoResultFound:
            return None

    def delete_entry_by_name(self, name: str):
        try:
            index_obj = self.query.filter(self.model._name == name).one()
            index_metadata = index_obj.as_dataclass()
            if os.path.exists(index_metadata.save_file_path):
                if os.path.isfile(index_metadata.save_file_path):
                    os.remove(index_metadata.save_file_path)
            index_obj.delete(self.session)
        except Exception:
            logger.exception('Delete index failed for name {}'.format(name))
            return False
        return True

