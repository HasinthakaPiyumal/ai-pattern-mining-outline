# Cluster 20

class TestModels(unittest.TestCase):

    def test_tqueryentities(self):
        query_entities = TQueryEntities(named=['Entity1', 'Entity2'], generic=['Generic1', 'Generic2'])
        self.assertEqual(query_entities.named, ['ENTITY1', 'ENTITY2'])
        self.assertEqual(query_entities.generic, ['Generic1', 'Generic2'])
        with self.assertRaises(ValidationError):
            TQueryEntities(entities=['Entity1', 'Entity2'], n='two')

    def test_teditrelationship(self):
        edit_relationship = TEditRelation(ids=[1, 2], description='Combined relationship description')
        self.assertEqual(edit_relationship.ids, [1, 2])
        self.assertEqual(edit_relationship.description, 'Combined relationship description')

    def test_teditrelationshiplist(self):
        edit_relationship = TEditRelation(ids=[1, 2], description='Combined relationship description')
        edit_relationship_list = TEditRelationList(grouped_facts=[edit_relationship])
        self.assertEqual(edit_relationship_list.groups, [edit_relationship])

    def test_dump_to_csv(self):
        data = [TEntity(name='Sample name', type='SAMPLE TYPE', description='Sample description')]
        fields = ['name', 'type']
        values = {'score': [0.9]}
        csv_output = dump_to_csv(data, fields, with_header=True, **values)
        expected_output = ['name\ttype\tscore', 'Sample name\tSAMPLE TYPE\t0.9']
        self.assertEqual(csv_output, expected_output)

