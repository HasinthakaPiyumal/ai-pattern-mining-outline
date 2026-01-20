# Cluster 15

class DocumentStorageEngine(AbstractMediaStorageEngine):

    def __init__(self, db: EvaDBDatabase):
        super().__init__(db)

    def read(self, table: TableCatalogEntry, chunk_params: dict) -> Iterator[Batch]:
        for doc_files in self._rdb_handler.read(self._get_metadata_table(table), 12):
            for _, (row_id, file_name, _) in doc_files.iterrows():
                system_file_name = self._xform_file_url_to_file_name(file_name)
                doc_file = Path(table.file_url) / system_file_name
                reader = DocumentReader(str(doc_file), batch_mem_size=1, chunk_params=chunk_params)
                for batch in reader.read():
                    batch.frames[table.columns[0].name] = row_id
                    batch.frames[table.columns[1].name] = str(file_name)
                    batch.frames[ROW_NUM_COLUMN] = row_id * ROW_NUM_MAGIC + batch.frames[ROW_NUM_COLUMN]
                    yield batch

