# Cluster 22

class MetadataCombiner(BaseModel):
    """
    Combine different tables with relationship, used for describing the relationship between tables.

    Args:
        version (str): version
        named_metadata (Dict[str, Any]): pairs of table name and metadata
        relationships (List[Any]): list of relationships
    """
    version: str = '1.0'
    named_metadata: Dict[str, Metadata] = {}
    relationships: List[Relationship] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check(self):
        """Do necessary checks:

        - Whether number of tables corresponds to relationships.
        - Whether table names corresponds to the relationship between tables;
        """
        for m in self.named_metadata.values():
            m.check()
        table_names = set(self.named_metadata.keys())
        relationship_parents = set((r.parent_table for r in self.relationships))
        relationship_children = set((r.child_table for r in self.relationships))
        if not table_names.issuperset(relationship_parents):
            raise MetadataCombinerInvalidError(f"Relationships' parent table {relationship_parents - table_names} is missing.")
        if not table_names.issuperset(relationship_children):
            raise MetadataCombinerInvalidError(f"Relationships' child table {relationship_children - table_names} is missing.")
        if not (relationship_parents | relationship_children).issuperset(table_names):
            raise MetadataCombinerInvalidError(f'Table {table_names - (relationship_parents + relationship_children)} is missing in relationships.')
        logger.info('MultiTableCombiner check finished.')

    @classmethod
    def from_dataloader(cls, dataloaders: list[DataLoader], metadata_from_dataloader_kwargs: None | dict=None, relationshipe_inspector: None | str | type[Inspector]='SubsetRelationshipInspector', relationships_inspector_kwargs: None | dict=None, relationships: None | list[Relationship]=None):
        """
        Combine multiple dataloaders with relationship.

        Args:
            dataloaders (list[DataLoader]): list of dataloaders
            max_chunk (int): max chunk count for relationship inspector.
            metadata_from_dataloader_kwargs (dict): kwargs for :func:`Metadata.from_dataloader`
            relationshipe_inspector (str | type[Inspector]): relationship inspector
            relationships_inspector_kwargs (dict): kwargs for :func:`InspectorManager.init`
            relationships (list[Relationship]): list of relationships
        """
        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]
        metadata_from_dataloader_kwargs = metadata_from_dataloader_kwargs or {}
        named_metadata = {d.identity: Metadata.from_dataloader(d, **metadata_from_dataloader_kwargs) for d in dataloaders}
        if relationships is None and relationshipe_inspector is not None:
            if relationships_inspector_kwargs is None:
                relationships_inspector_kwargs = {}
            inspector = InspectorManager().init(relationshipe_inspector, **relationships_inspector_kwargs)
            for d in dataloaders:
                for chunk in d.iter():
                    inspector.fit(chunk, name=d.identity, metadata=named_metadata[d.identity])
            relationships = inspector.inspect()['relationships']
        return cls(named_metadata=named_metadata, relationships=relationships)

    @classmethod
    def from_dataframe(cls, dataframes: list[pd.DataFrame], names: list[str], metadata_from_dataloader_kwargs: None | dict=None, relationshipe_inspector: None | str | type[Inspector]='SubsetRelationshipInspector', relationships_inspector_kwargs: None | dict=None, relationships: None | list[Relationship]=None) -> 'MetadataCombiner':
        """
        Combine multiple dataframes with relationship.

        Args:
            dataframes (list[pd.DataFrame]): list of dataframes
            names (list[str]): list of names
            metadata_from_dataloader_kwargs (dict): kwargs for :func:`Metadata.from_dataloader`
            relationshipe_inspector (str | type[Inspector]): relationship inspector
            relationships_inspector_kwargs (dict): kwargs for :func:`InspectorManager.init`
            relationships (list[Relationship]): list of relationships
        """
        if not isinstance(dataframes, list):
            dataframes = [dataframes]
        if not isinstance(names, list):
            names = [names]
        metadata_from_dataloader_kwargs = metadata_from_dataloader_kwargs or {}
        if len(dataframes) != len(names):
            raise MetadataCombinerInitError('dataframes and names should have same length.')
        named_metadata = {n: Metadata.from_dataframe(d, **metadata_from_dataloader_kwargs) for n, d in zip(names, dataframes)}
        if relationships is None and relationshipe_inspector is not None:
            if relationships_inspector_kwargs is None:
                relationships_inspector_kwargs = {}
            inspector = InspectorManager().init(relationshipe_inspector, **relationships_inspector_kwargs)
            for n, d in zip(names, dataframes):
                inspector.fit(d, name=n, metadata=named_metadata[n])
            relationships = inspector.inspect()['relationships']
        return cls(named_metadata=named_metadata, relationships=relationships)

    def _dump_json(self):
        return self.model_dump_json()

    def save(self, save_dir: str | Path, metadata_subdir: str='metadata', relationship_subdir: str='relationship'):
        """
        Save metadata to json file.

        This will create several subdirectories for metadata and relationship.

        Args:
            save_dir (str | Path): directory to save
            metadata_subdir (str): subdirectory for metadata, default is "metadata"
            relationship_subdir (str): subdirectory for relationship, default is "relationship"
        """
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        version_file = save_dir / 'version'
        version_file.write_text(self.version)
        metadata_subdir = save_dir / metadata_subdir
        relationship_subdir = save_dir / relationship_subdir
        metadata_subdir.mkdir(parents=True, exist_ok=True)
        for name, metadata in self.named_metadata.items():
            metadata.save(metadata_subdir / f'{name}.json')
        relationship_subdir.mkdir(parents=True, exist_ok=True)
        for relationship in self.relationships:
            relationship.save(relationship_subdir / f'{relationship.parent_table}_{relationship.child_table}.json')

    @classmethod
    def load(cls, save_dir: str | Path, metadata_subdir: str='metadata', relationship_subdir: str='relationship', version: None | str=None) -> 'MetadataCombiner':
        """
        Load metadata from json file.

        Args:
            save_dir (str | Path): directory to save
            metadata_subdir (str): subdirectory for metadata, default is "metadata"
            relationship_subdir (str): subdirectory for relationship, default is "relationship"
            version (str): Manual version, if not specified, try to load from version file
        """
        save_dir = Path(save_dir).expanduser().resolve()
        if not version:
            logger.debug('No version specified, try to load from version file.')
            version_file = save_dir / 'version'
            if version_file.exists():
                version = version_file.read_text().strip()
            else:
                logger.info('No version file found, assume version is 1.0')
                version = '1.0'
        named_metadata = {p.stem: Metadata.load(p) for p in (save_dir / metadata_subdir).glob('*')}
        relationships = [Relationship.load(p) for p in (save_dir / relationship_subdir).glob('*')]
        cls.upgrade(version, named_metadata, relationships)
        return cls(version=version, named_metadata=named_metadata, relationships=relationships)

    @classmethod
    def upgrade(cls, old_version: str, named_metadata: dict[str, Metadata], relationships: list[Relationship]) -> None:
        """
        Upgrade metadata from old version to new version

        :ref:`Metadata.upgrade` and :ref:`Relationship.upgrade` will try upgrade when loading.
        So here we just do Combiner's upgrade.
        """
        pass

    @property
    def fields(self) -> Iterable[str]:
        """
        Return all fields in MetadataCombiner.
        """
        return chain((k for k in self.model_fields if k.endswith('_columns')))

    def __eq__(self, other):
        if not isinstance(other, MetadataCombiner):
            return super().__eq__(other)
        return self.version == other.version and all((self.get(key) == other.get(key) for key in set(chain(self.fields, other.fields)))) and (set(self.fields) == set(other.fields))

class SubsetRelationshipInspector(RelationshipInspector):
    """
    Inspecting relationships by comparing two columns is subset or not. So it needs to inspect all data for prev
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maybe_related_columns: dict[str, dict[str, pd.Series]] = {}

    def _is_related(self, p: pd.Series, c: pd.Series) -> bool:
        """
        If child is subset of parent, assume related
        """
        return c.isin(p).all()

    def _build_relationship(self) -> list[Relationship]:
        r = []
        for parent, p_m_related in self.maybe_related_columns.items():
            for child, c_m_related in self.maybe_related_columns.items():
                if parent == child:
                    continue
                related_pairs = []
                for p_col, p_df in p_m_related.items():
                    for c_col, c_df in c_m_related.items():
                        if self._is_related(p_df, c_df):
                            related_pairs.append((p_col, c_col) if p_col != c_col else p_col)
                if related_pairs:
                    r.append(Relationship.build(parent, child, related_pairs))
        return r

    def fit(self, raw_data: pd.DataFrame, name: str | None=None, metadata: 'Metadata' | None=None, *args, **kwargs):
        columns = set((n for n in chain(metadata.id_columns, metadata.primary_keys)))
        for c in columns:
            cur_map = self.maybe_related_columns.setdefault(name, dict())
            cur_map[c] = pd.concat((cur_map.get(c, pd.Series()), raw_data[[c]].squeeze()), ignore_index=True)

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""
        return {'relationships': self._build_relationship()}

@pytest.fixture
def metadata():
    yield Metadata()

@pytest.fixture
def demo_multi_data_relationship():
    yield Relationship.build(parent_table='store', child_table='train', foreign_keys=['Store'])

@pytest.mark.parametrize('parent_table, parent_metadata, child_table, child_metadata, foreign_keys, exception', [('parent', parent_metadata, 'child', child_metadata, [KeyTuple('parent_id', 'parent_id')], None), ('parent', error_parent_metadata, 'child', child_metadata, [KeyTuple('parent_id', 'parent_id')], RelationshipInitError), ('parent', parent_metadata, 'child', error_child_metadata, [KeyTuple('parent_id', 'parent_id')], RelationshipInitError), ('parent', parent_metadata, 'parent', parent_metadata, [KeyTuple('parent_id', 'parent_id')], RelationshipInitError), ('parent', parent_metadata, 'parent', parent_metadata, [], RelationshipInitError), ('', parent_metadata, 'child', child_metadata, [KeyTuple('parent_id', 'parent_id')], RelationshipInitError), ('parent', parent_metadata, '', child_metadata, [KeyTuple('parent_id', 'parent_id')], RelationshipInitError), ('', parent_metadata, '', child_metadata, [KeyTuple('parent_id', 'parent_id')], RelationshipInitError), ('', parent_metadata, '', child_metadata, [], RelationshipInitError)])
def test_build(parent_table, parent_metadata, child_table, child_metadata, foreign_keys, exception):
    if exception:
        with pytest.raises(exception):
            Relationship.build(parent_table=parent_table, parent_metadata=parent_metadata, child_table=child_table, child_metadata=child_metadata, foreign_keys=foreign_keys)
    else:
        relationship = Relationship.build(parent_table=parent_table, parent_metadata=parent_metadata, child_table=child_table, child_metadata=child_metadata, foreign_keys=foreign_keys)
        assert relationship.parent_table == parent_table
        assert relationship.child_table == child_table
        assert relationship.foreign_keys == foreign_keys

def test_save_and_load(tmpdir):
    save_file = tmpdir / 'relationship.json'
    relationship = Relationship.build(parent_table='parent', parent_metadata=Metadata(primary_keys=['parent_id'], column_list=['parent_id'], id_columns={'parent_id'}), child_table='child', child_metadata=Metadata(primary_keys=['child_id'], column_list=['parent_id', 'child_id'], id_columns={'parent_id', 'child_id'}), foreign_keys=[KeyTuple('parent_id', 'parent_id')])
    relationship.save(save_file)
    assert save_file.exists()
    assert relationship == Relationship.load(save_file)

def test_metadata_save_load(metadata: Metadata, tmp_path: Path):
    test_path = tmp_path / 'metadata_path_test.json'
    metadata.save(test_path)
    new_meatadata = Metadata.load(test_path)
    assert metadata == new_meatadata

def test_metadata_primary_query_filed_tags():
    metadata = Metadata()
    metadata.set('id_columns', {'user_id'})
    metadata.set('int_columns', {'user_id', 'age'})
    results = metadata.query('user_id')
    results_list = list(results)
    print(results_list)
    assert 'id_columns' in results_list
    assert 'int_columns' in results_list

def test_from_dataloader(demo_relational_table_path, tmp_path):
    table_a_path, table_b_path, pairs = demo_relational_table_path
    dl_a = DataLoader(CsvConnector(path=table_a_path))
    dl_b = DataLoader(CsvConnector(path=table_b_path))
    relationship = Relationship.build(parent_table=dl_a.identity, parent_metadata=Metadata(primary_keys=['id'], column_list=['id'], id_columns={'id'}), child_table=dl_b.identity, child_metadata=Metadata(primary_keys=['child_id'], column_list=['child_id', 'foreign_id'], id_columns={'child_id', 'foreign_id'}), foreign_keys=pairs)
    combiner = MetadataCombiner.from_dataloader(dataloaders=[dl_a, dl_b], metadata_from_dataloader_kwargs={}, relationshipe_inspector=MockInspector, relationships_inspector_kwargs=dict(dummy_data=[relationship]))
    assert dl_a.identity in combiner.named_metadata
    assert dl_b.identity in combiner.named_metadata
    assert combiner.relationships == [relationship]
    save_dir = tmp_path / 'unittest-combinner'
    combiner.save(save_dir)
    assert save_dir.exists()
    loaded_combiner = MetadataCombiner.load(save_dir)
    assert combiner == loaded_combiner

def test_from_dataframe(demo_relational_table_path, tmp_path):
    table_a_path, table_b_path, pair = demo_relational_table_path
    relationship = Relationship.build(parent_table='table_a', parent_metadata=Metadata(primary_keys=['id'], column_list=['id'], id_columns={'id'}), child_table='table_b', child_metadata=Metadata(primary_keys=['child_id'], column_list=['child_id', 'foreign_id'], id_columns={'child_id', 'foreign_id'}), foreign_keys=pair)
    tb_a = pd.read_csv(table_a_path)
    tb_b = pd.read_csv(table_b_path)
    combiner = MetadataCombiner.from_dataframe(dataframes=[tb_a, tb_b], names=['table_a', 'table_b'], metadata_from_dataloader_kwargs={}, relationshipe_inspector=MockInspector, relationships_inspector_kwargs=dict(dummy_data=[relationship]))
    assert 'table_a' in combiner.named_metadata
    assert 'table_b' in combiner.named_metadata
    assert combiner.relationships == [relationship]
    save_dir = tmp_path / 'unittest-combinner'
    combiner.save(save_dir)
    assert save_dir.exists()
    loaded_combiner = MetadataCombiner.load(save_dir)
    assert combiner == loaded_combiner

def test_custom_build_from_dataloaders(demo_relational_table_path, tmp_path):
    table_a_path, table_b_path, pairs = demo_relational_table_path
    dl_a = DataLoader(CsvConnector(path=table_a_path))
    dl_b = DataLoader(CsvConnector(path=table_b_path))
    relationship = Relationship.build(parent_table=dl_a.identity, parent_metadata=Metadata(primary_keys=['id'], column_list=['id'], id_columns={'id'}), child_table=dl_b.identity, child_metadata=Metadata(primary_keys=['child_id'], column_list=['child_id', 'foreign_id'], id_columns={'child_id', 'foreign_id'}), foreign_keys=pairs)
    combiner = MetadataCombiner.from_dataloader(dataloaders=[dl_a, dl_b], metadata_from_dataloader_kwargs={}, relationshipe_inspector=MockInspector, relationships_inspector_kwargs=dict(dummy_data=Relationship.build(parent_table='balaP', parent_metadata=Metadata(primary_keys=['balabala'], column_list=['balabala'], id_columns={'balabala'}), child_table='balaC', child_metadata=Metadata(primary_keys=['child_id'], column_list=['balabala', 'child_id'], id_columns={'balabala', 'child_id'}), foreign_keys=['balabala'])), relationships=[relationship])
    assert dl_a.identity in combiner.named_metadata
    assert dl_b.identity in combiner.named_metadata
    assert combiner.relationships == [relationship]
    save_dir = tmp_path / 'unittest-combinner'
    combiner.save(save_dir)
    assert save_dir.exists()
    loaded_combiner = MetadataCombiner.load(save_dir)
    assert combiner == loaded_combiner

def test_custom_build_from_dataframe(demo_relational_table_path, tmp_path):
    table_a_path, table_b_path, pair = demo_relational_table_path
    relationship = Relationship.build(parent_table='table_a', parent_metadata=Metadata(primary_keys=['id'], column_list=['id'], id_columns={'id'}), child_table='table_b', child_metadata=Metadata(primary_keys=['child_id'], column_list=['child_id', 'foreign_id'], id_columns={'child_id', 'foreign_id'}), foreign_keys=pair)
    tb_a = pd.read_csv(table_a_path)
    tb_b = pd.read_csv(table_b_path)
    combiner = MetadataCombiner.from_dataframe(dataframes=[tb_a, tb_b], names=['table_a', 'table_b'], metadata_from_dataloader_kwargs={}, relationshipe_inspector=MockInspector, relationships_inspector_kwargs=dict(dummy_data=Relationship.build(parent_table='balaP', parent_metadata=Metadata(primary_keys=['balabala'], column_list=['balabala'], id_columns={'balabala'}), child_table='balaC', child_metadata=Metadata(primary_keys=['child_id'], column_list=['balabala', 'child_id'], id_columns={'balabala', 'child_id'}), foreign_keys=['balabala'])), relationships=[relationship])
    assert 'table_a' in combiner.named_metadata
    assert 'table_b' in combiner.named_metadata
    assert combiner.relationships == [relationship]
    save_dir = tmp_path / 'unittest-combinner'
    combiner.save(save_dir)
    assert save_dir.exists()
    loaded_combiner = MetadataCombiner.load(save_dir)
    assert combiner == loaded_combiner

@pytest.fixture
def dummy_relationship(demo_relational_table_path):
    _, _, pairs = demo_relational_table_path
    yield Relationship.build(parent_table='parent', child_table='child', foreign_keys=pairs)

