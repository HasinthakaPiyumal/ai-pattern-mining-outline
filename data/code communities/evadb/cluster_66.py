# Cluster 66

def init_evadb_instance(db_dir: str, host: str=None, port: int=None, custom_db_uri: str=None):
    if db_dir is None:
        db_dir = EvaDB_DATABASE_DIR
    config_obj = bootstrap_environment(Path(db_dir), evadb_installation_dir=Path(EvaDB_INSTALLATION_DIR))
    catalog_uri = custom_db_uri or get_default_db_uri(Path(db_dir))
    bootstrap_configs(get_catalog_instance(catalog_uri), config_obj)
    return EvaDBDatabase(db_dir, catalog_uri, get_catalog_instance)

@pytest.mark.notparallel
class CatalogManagerTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls) -> None:
        cls.mocks = [mock.patch('evadb.catalog.catalog_manager.SQLConfig'), mock.patch('evadb.catalog.catalog_manager.init_db')]
        for single_mock in cls.mocks:
            single_mock.start()
            cls.addClassCleanup(single_mock.stop)

    @mock.patch('evadb.catalog.catalog_manager.init_db')
    def test_catalog_bootstrap(self, mocked_db):
        x = CatalogManager(MagicMock())
        x._bootstrap_catalog()
        mocked_db.assert_called()

    @mock.patch('evadb.catalog.catalog_manager.CatalogManager.create_and_insert_table_catalog_entry')
    def test_create_multimedia_table_catalog_entry(self, mock):
        x = CatalogManager(MagicMock())
        name = 'myvideo'
        x.create_and_insert_multimedia_table_catalog_entry(name=name, format_type=FileFormatType.VIDEO)
        columns = get_video_table_column_definitions()
        mock.assert_called_once_with(TableInfo(name), columns, table_type=TableType.VIDEO_DATA)

    @mock.patch('evadb.catalog.catalog_manager.init_db')
    @mock.patch('evadb.catalog.catalog_manager.TableCatalogService')
    def test_insert_table_catalog_entry_should_create_table_and_columns(self, ds_mock, initdb_mock):
        catalog = CatalogManager(MagicMock())
        file_url = 'file1'
        table_name = 'name'
        columns = [ColumnCatalogEntry('c1', ColumnType.INTEGER)]
        catalog.insert_table_catalog_entry(table_name, file_url, columns)
        ds_mock.return_value.insert_entry.assert_called_with(table_name, file_url, identifier_column='id', table_type=TableType.VIDEO_DATA, column_list=[ANY] + columns)

    @mock.patch('evadb.catalog.catalog_manager.init_db')
    @mock.patch('evadb.catalog.catalog_manager.TableCatalogService')
    def test_get_table_catalog_entry_when_table_exists(self, ds_mock, initdb_mock):
        catalog = CatalogManager(MagicMock())
        table_name = 'name'
        database_name = 'database'
        row_id = 1
        table_obj = MagicMock(row_id=row_id)
        ds_mock.return_value.get_entry_by_name.return_value = table_obj
        actual = catalog.get_table_catalog_entry(table_name, database_name)
        ds_mock.return_value.get_entry_by_name.assert_called_with(database_name, table_name)
        self.assertEqual(actual.row_id, row_id)

    @mock.patch('evadb.catalog.catalog_manager.init_db')
    @mock.patch('evadb.catalog.catalog_manager.TableCatalogService')
    @mock.patch('evadb.catalog.catalog_manager.ColumnCatalogService')
    def test_get_table_catalog_entry_when_table_doesnot_exists(self, dcs_mock, ds_mock, initdb_mock):
        catalog = CatalogManager(MagicMock())
        table_name = 'name'
        database_name = 'database'
        table_obj = None
        ds_mock.return_value.get_entry_by_name.return_value = table_obj
        actual = catalog.get_table_catalog_entry(table_name, database_name)
        ds_mock.return_value.get_entry_by_name.assert_called_with(database_name, table_name)
        dcs_mock.return_value.filter_entries_by_table_id.assert_not_called()
        self.assertEqual(actual, table_obj)

    @mock.patch('evadb.catalog.catalog_manager.FunctionCatalogService')
    @mock.patch('evadb.catalog.catalog_manager.FunctionIOCatalogService')
    @mock.patch('evadb.catalog.catalog_manager.FunctionMetadataCatalogService')
    @mock.patch('evadb.catalog.catalog_manager.get_file_checksum')
    def test_insert_function(self, checksum_mock, functionmetadata_mock, functionio_mock, function_mock):
        catalog = CatalogManager(MagicMock())
        function_io_list = [MagicMock()]
        function_metadata_list = [MagicMock()]
        actual = catalog.insert_function_catalog_entry('function', 'sample.py', 'classification', function_io_list, function_metadata_list)
        function_mock.return_value.insert_entry.assert_called_with('function', 'sample.py', 'classification', checksum_mock.return_value, function_io_list, function_metadata_list)
        checksum_mock.assert_called_with('sample.py')
        self.assertEqual(actual, function_mock.return_value.insert_entry.return_value)

    @mock.patch('evadb.catalog.catalog_manager.FunctionCatalogService')
    def test_get_function_catalog_entry_by_name(self, function_mock):
        catalog = CatalogManager(MagicMock())
        actual = catalog.get_function_catalog_entry_by_name('name')
        function_mock.return_value.get_entry_by_name.assert_called_with('name')
        self.assertEqual(actual, function_mock.return_value.get_entry_by_name.return_value)

    @mock.patch('evadb.catalog.catalog_manager.FunctionCatalogService')
    def test_delete_function(self, function_mock):
        CatalogManager(MagicMock()).delete_function_catalog_entry_by_name('name')
        function_mock.return_value.delete_entry_by_name.assert_called_with('name')

    @mock.patch('evadb.catalog.catalog_manager.FunctionIOCatalogService')
    def test_get_function_outputs(self, function_mock):
        mock_func = function_mock.return_value.get_output_entries_by_function_id
        function_obj = MagicMock(spec=FunctionCatalogEntry)
        CatalogManager(MagicMock()).get_function_io_catalog_output_entries(function_obj)
        mock_func.assert_called_once_with(function_obj.row_id)

    @mock.patch('evadb.catalog.catalog_manager.FunctionIOCatalogService')
    def test_get_function_inputs(self, function_mock):
        mock_func = function_mock.return_value.get_input_entries_by_function_id
        function_obj = MagicMock(spec=FunctionCatalogEntry)
        CatalogManager(MagicMock()).get_function_io_catalog_input_entries(function_obj)
        mock_func.assert_called_once_with(function_obj.row_id)

def bootstrap_environment(evadb_dir: Path, evadb_installation_dir: Path):
    """
    Populates necessary configuration for EvaDB to be able to run.

    Arguments:
        evadb_dir: path to evadb database directory
        evadb_installation_dir: path to evadb package
    """
    config_obj = BASE_EVADB_CONFIG
    config_default_dict = create_directories_and_get_default_config_values(Path(evadb_dir), Path(evadb_installation_dir))
    assert evadb_dir.exists(), f'{evadb_dir} does not exist'
    assert evadb_installation_dir.exists(), f'{evadb_installation_dir} does not exist'
    config_obj = merge_dict_of_dicts(config_default_dict, config_obj)
    mode = config_obj['mode']
    level = logging.WARN if mode == 'release' else logging.DEBUG
    evadb_logger.setLevel(level)
    evadb_logger.debug(f'Setting logging level to: {str(level)}')
    return config_obj

def get_default_db_uri(evadb_dir: Path):
    return f'sqlite:///{evadb_dir.resolve()}/{DB_DEFAULT_NAME}'

def bootstrap_configs(catalog, configs: dict):
    """
    load all the configuration values into the catalog table configuration_catalog
    """
    for key, value in configs.items():
        catalog.upsert_configuration_catalog_entry(key, value)

def get_catalog_instance(db_uri: str):
    from evadb.catalog.catalog_manager import CatalogManager
    return CatalogManager(db_uri)

def create_directories_and_get_default_config_values(evadb_dir: Path, evadb_installation_dir: Path) -> Union[dict, str]:
    default_install_dir = evadb_installation_dir
    dataset_location = evadb_dir / EvaDB_DATASET_DIR
    index_dir = evadb_dir / INDEX_DIR
    cache_dir = evadb_dir / CACHE_DIR
    s3_dir = evadb_dir / S3_DOWNLOAD_DIR
    tmp_dir = evadb_dir / TMP_DIR
    function_dir = evadb_dir / FUNCTION_DIR
    model_dir = evadb_dir / MODEL_DIR
    if not evadb_dir.exists():
        evadb_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_location.exists():
        dataset_location.mkdir(parents=True, exist_ok=True)
    if not index_dir.exists():
        index_dir.mkdir(parents=True, exist_ok=True)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
    if not s3_dir.exists():
        s3_dir.mkdir(parents=True, exist_ok=True)
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True, exist_ok=True)
    if not function_dir.exists():
        function_dir.mkdir(parents=True, exist_ok=True)
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
    config_obj = {}
    config_obj['evadb_installation_dir'] = str(default_install_dir.resolve())
    config_obj['datasets_dir'] = str(dataset_location.resolve())
    config_obj['catalog_database_uri'] = get_default_db_uri(evadb_dir)
    config_obj['index_dir'] = str(index_dir.resolve())
    config_obj['cache_dir'] = str(cache_dir.resolve())
    config_obj['s3_download_dir'] = str(s3_dir.resolve())
    config_obj['tmp_dir'] = str(tmp_dir.resolve())
    config_obj['function_dir'] = str(function_dir.resolve())
    config_obj['model_dir'] = str(model_dir.resolve())
    return config_obj

def format_col_str(col):
    suffix = '(FK)' if col.name in fk_col_names else '(PK)' if col.name in pk_col_names else ''
    if show_datatypes:
        return '- %s : %s' % (col.name + suffix, format_col_type(col))
    else:
        return '- %s' % (col.name + suffix)

def format_col_type(col):
    try:
        return col.type.get_col_spec()
    except (AttributeError, NotImplementedError):
        return str(col.type)

def _render_table_html(table, metadata, show_indexes, show_datatypes, show_column_keys, show_schema_name, format_schema_name, format_table_name):
    use_column_key_attr = hasattr(ForeignKeyConstraint, 'column_keys')
    if show_column_keys:
        if use_column_key_attr:
            fk_col_names = set([h for f in table.foreign_key_constraints for h in f.columns.keys()])
        else:
            fk_col_names = set([h.name for f in table.foreign_keys for h in f.constraint.columns])
        pk_col_names = set([f for f in table.primary_key.columns.keys()])
    else:
        fk_col_names = set()
        pk_col_names = set()

    def format_col_type(col):
        try:
            return col.type.get_col_spec()
        except (AttributeError, NotImplementedError):
            return str(col.type)

    def format_col_str(col):
        suffix = '(FK)' if col.name in fk_col_names else '(PK)' if col.name in pk_col_names else ''
        if show_datatypes:
            return '- %s : %s' % (col.name + suffix, format_col_type(col))
        else:
            return '- %s' % (col.name + suffix)

    def format_name(obj_name, format_dict):
        if format_dict is not None:
            return '<FONT COLOR="{color}" POINT-SIZE="{size}">{bld}{it}{name}{e_it}{e_bld}</FONT>'.format(name=obj_name, color=format_dict.get('color') if 'color' in format_dict else 'initial', size=float(format_dict['fontsize']) if 'fontsize' in format_dict else 'initial', it='<I>' if format_dict.get('italics') else '', e_it='</I>' if format_dict.get('italics') else '', bld='<B>' if format_dict.get('bold') else '', e_bld='</B>' if format_dict.get('bold') else '')
        else:
            return obj_name
    schema_str = ''
    if show_schema_name == True and hasattr(table, 'schema') and (table.schema is not None):
        schema_str = format_name(table.schema, format_schema_name)
    table_str = format_name(table.name, format_table_name)
    html = '<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0"><TR><TD ALIGN="CENTER">%s%s%s</TD></TR><TR><TD BORDER="1" CELLPADDING="0"></TD></TR>' % (schema_str, '.' if show_schema_name else '', table_str)
    html += ''.join(('<TR><TD ALIGN="LEFT" PORT="%s">%s</TD></TR>' % (col.name, format_col_str(col)) for col in table.columns))
    if metadata.bind and isinstance(metadata.bind.dialect, PGDialect):
        indexes = dict(((name, defin) for name, defin in metadata.bind.execute(text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = '%s'" % table.name))))
        if indexes and show_indexes:
            html += '<TR><TD BORDER="1" CELLPADDING="0"></TD></TR>'
            for index, defin in indexes.items():
                ilabel = 'UNIQUE' in defin and 'UNIQUE ' or 'INDEX '
                ilabel += defin[defin.index('('):]
                html += '<TR><TD ALIGN="LEFT">%s</TD></TR>' % ilabel
    html += '</TABLE>>'
    return html

def format_name(obj_name, format_dict):
    if format_dict is not None:
        return '<FONT COLOR="{color}" POINT-SIZE="{size}">{bld}{it}{name}{e_it}{e_bld}</FONT>'.format(name=obj_name, color=format_dict.get('color') if 'color' in format_dict else 'initial', size=float(format_dict['fontsize']) if 'fontsize' in format_dict else 'initial', it='<I>' if format_dict.get('italics') else '', e_it='</I>' if format_dict.get('italics') else '', bld='<B>' if format_dict.get('bold') else '', e_bld='</B>' if format_dict.get('bold') else '')
    else:
        return obj_name

def create_schema_graph(tables=None, metadata=None, show_indexes=True, show_datatypes=True, font='Bitstream-Vera Sans', concentrate=True, relation_options={}, rankdir='TB', show_column_keys=False, restrict_tables=None, show_schema_name=False, format_schema_name=None, format_table_name=None):
    """
    Args:
      - metadata (sqlalchemy.MetaData, default=None): SqlAlchemy `MetaData` with reference to related tables.  If none
        is provided, uses metadata from first entry of `tables` argument.
      - concentrate (bool, default=True): Specifies if multiedges should be merged into a single edge & partially
        parallel edges to share overlapping path.  Passed to `pydot.Dot` object.
      - relation_options (dict, default: None): kwargs passed to pydot.Edge init.  Most attributes in
        pydot.EDGE_ATTRIBUTES are viable options.  A few values are set programmatically.
      - rankdir (string, default='TB'): Sets direction of graph layout.  Passed to `pydot.Dot` object.  Options are
        'TB' (top to bottom), 'BT' (bottom to top), 'LR' (left to right), 'RL' (right to left).
      - show_column_keys (bool, default=False): If true then add a PK/FK suffix to columns names that are primary and
        foreign keys.
      - restrict_tables (None or list of strings): Restrict the graph to only consider tables whose name are defined
        `restrict_tables`.
      - show_schema_name (bool, default=False): If true, then prepend '<schema name>.' to the table name resulting in
        '<schema name>.<table name>'.
      - format_schema_name (dict, default=None): If provided, allowed keys include: 'color' (hex color code incl #),
        'fontsize' as a float, and 'bold' and 'italics' as bools.
      - format_table_name (dict, default=None): If provided, allowed keys include: 'color' (hex color code incl #),
        'fontsize' as a float, and 'bold' and 'italics' as bools.
    """
    relation_kwargs = {'fontsize': '7.0', 'dir': 'both'}
    relation_kwargs.update(relation_options)
    if metadata is None and tables is not None and len(tables):
        metadata = tables[0].metadata
    elif tables is None and metadata is not None:
        if not len(metadata.tables):
            metadata.reflect()
        tables = metadata.tables.values()
    else:
        raise ValueError('You need to specify at least tables or metadata')
    if format_schema_name is not None and len(set(format_schema_name.keys()).difference({'color', 'fontsize', 'italics', 'bold'})) > 0:
        raise KeyError('Unrecognized keys were used in dict provided for `format_schema_name` parameter')
    if format_table_name is not None and len(set(format_table_name.keys()).difference({'color', 'fontsize', 'italics', 'bold'})) > 0:
        raise KeyError('Unrecognized keys were used in dict provided for `format_table_name` parameter')
    graph = pydot.Dot(prog='dot', mode='ipsep', overlap='ipsep', sep='0.01', concentrate=str(concentrate), rankdir=rankdir)
    if restrict_tables is None:
        restrict_tables = set([t.name.lower() for t in tables])
    else:
        restrict_tables = set([t.lower() for t in restrict_tables])
    tables = [t for t in tables if t.name.lower() in restrict_tables]
    for table in tables:
        graph.add_node(pydot.Node(str(table.name), shape='plaintext', label=_render_table_html(table, metadata, show_indexes, show_datatypes, show_column_keys, show_schema_name, format_schema_name, format_table_name), fontname=font, fontsize='7.0'))
    for table in tables:
        for fk in table.foreign_keys:
            if fk.column.table not in tables:
                continue
            edge = [table.name, fk.column.table.name]
            is_inheritance = fk.parent.primary_key and fk.column.primary_key
            if is_inheritance:
                edge = edge[::-1]
            graph_edge = pydot.Edge(*edge, headlabel='+ %s' % fk.column.name, taillabel='+ %s' % fk.parent.name, arrowhead=is_inheritance and 'none' or 'odot', arrowtail=(fk.parent.primary_key or fk.parent.unique) and 'empty' or 'crow', fontname=font, **relation_kwargs)
            graph.add_edge(graph_edge)
    return graph

def show_schema_graph(*args, **kwargs):
    from cStringIO import StringIO
    from PIL import Image
    iostream = StringIO(create_schema_graph(*args, **kwargs).create_png())
    Image.open(iostream).show(command=kwargs.get('command', 'gwenview'))

