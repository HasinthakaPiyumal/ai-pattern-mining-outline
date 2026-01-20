# Cluster 47

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, help='Path to the directory with documents', default='source_documents')
    args = parser.parse_args()
    directory_path = args.directory
    print(f'🔮 Welcome to EvaDB! Ingesting data in `{directory_path}`')
    load_data(source_folder_path=directory_path)
    print('🔥 Data ingestion complete! You can now run `privateGPT.py` to query your loaded data.')

def load_data(source_folder_path: str):
    path = os.path.dirname(evadb.__file__)
    cursor = evadb.connect(path).cursor()
    cursor.query('DROP FUNCTION IF EXISTS embedding;').execute()
    text_feat_function_query = f"CREATE FUNCTION IF NOT EXISTS embedding\n            IMPL  '{path}/functions/sentence_feature_extractor.py';\n            "
    print(text_feat_function_query)
    cursor.query(text_feat_function_query).execute()
    print('🧹 Dropping existing tables in EvaDB')
    cursor.query('DROP TABLE IF EXISTS data_table;').execute()
    cursor.query('DROP TABLE IF EXISTS embedding_table;').execute()
    print('📄 Loading PDFs into EvaDB')
    text_load_query = f"LOAD PDF '{source_folder_path}/*.pdf' INTO data_table;"
    print(text_load_query)
    cursor.query(text_load_query).execute()
    print('🤖 Extracting Feature Embeddings. This may take some time ...')
    cursor.query('CREATE TABLE IF NOT EXISTS embedding_table AS SELECT embedding(data), data FROM data_table;').execute()
    print('🔍 Building FAISS Index ...')
    cursor.query('\n        CREATE INDEX embedding_index\n        ON embedding_table (features)\n        USING FAISS;\n    ').execute()

