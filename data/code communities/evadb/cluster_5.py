# Cluster 5

class ImageStorageEngine(AbstractMediaStorageEngine):

    def __init__(self, db: EvaDBDatabase):
        super().__init__(db)

    def read(self, table: TableCatalogEntry) -> Iterator[Batch]:
        for image_files in self._rdb_handler.read(self._get_metadata_table(table)):
            for _, (row_id, file_name, _) in image_files.iterrows():
                system_file_name = self._xform_file_url_to_file_name(file_name)
                image_file = Path(table.file_url) / system_file_name
                reader = CVImageReader(str(image_file), batch_mem_size=1)
                for batch in reader.read():
                    batch.frames[table.columns[0].name] = row_id
                    batch.frames[table.columns[1].name] = str(file_name)
                    batch.frames[ROW_NUM_COLUMN] = batch.frames[table.columns[0].name]
                    yield batch

