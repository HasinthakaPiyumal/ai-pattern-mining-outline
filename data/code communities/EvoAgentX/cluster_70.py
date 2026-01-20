# Cluster 70

def run_faiss_examples():
    """Run examples using FaissToolkit for vector database operations."""
    print('\n===== FAISS TOOLKIT EXAMPLES =====\n')
    if not os.getenv('OPENAI_API_KEY'):
        print('❌ OPENAI_API_KEY not found in environment variables')
        print('To test FAISS examples, set your OpenAI API key:')
        print("export OPENAI_API_KEY='your-openai-api-key-here'")
        print('Get your key from: https://platform.openai.com/api-keys')
        return
    try:
        toolkit = FaissToolkit(name='DemoFaissToolkit', default_corpus_id='demo_corpus')
        print('✓ FaissToolkit initialized with default storage')
        print(f'✓ Using OpenAI API key: {os.getenv('OPENAI_API_KEY')[:8]}...')
        insert_tool = toolkit.get_tool('faiss_insert')
        query_tool = toolkit.get_tool('faiss_query')
        list_tool = toolkit.get_tool('faiss_list')
        stats_tool = toolkit.get_tool('faiss_stats')
        delete_tool = toolkit.get_tool('faiss_delete')
        print(f'✓ Available tools: {[tool.name for tool in toolkit.get_tools()]}')
        print('\n1. Inserting AI knowledge documents...')
        ai_documents = ['Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.', 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.', 'Deep learning is a specialized form of machine learning that uses neural networks with multiple layers to analyze and learn from data.', 'Natural Language Processing (NLP) helps computers understand, interpret, and generate human language in a useful way.', 'Computer vision enables machines to interpret and understand visual information from the world, including images and videos.', 'Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to achieve maximum cumulative reward.', 'Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information.', 'Transfer learning allows a model trained on one task to be adapted for a related task, improving efficiency and performance.', 'Generative AI models can create new content, such as text, images, music, and code, based on patterns learned from training data.', "Explainable AI focuses on making AI systems' decisions and processes transparent and understandable to humans."]
        insert_result = insert_tool(documents=ai_documents, metadata={'source': 'ai_knowledge_base', 'topic': 'artificial_intelligence', 'language': 'en', 'difficulty': 'intermediate'})
        if insert_result.get('success'):
            docs_inserted = insert_result.get('data', {}).get('documents_inserted', 0)
            chunks_created = insert_result.get('data', {}).get('chunks_created', 0)
            print(f'✓ Successfully inserted {docs_inserted} documents')
            print(f'  Chunks created: {chunks_created}')
        else:
            print(f'❌ Insert failed: {insert_result.get('error', 'Unknown error')}')
            return
        print('\n2. Performing semantic search queries...')
        search_queries = ['How do machines learn?', 'What is neural network?', 'Explain deep learning', 'How does AI generate content?', 'What is computer vision?']
        for i, query in enumerate(search_queries, 1):
            print(f"\n  Query {i}: '{query}'")
            search_result = query_tool(query=query, top_k=3, similarity_threshold=0.1)
            if search_result.get('success'):
                results = search_result.get('data', {}).get('results', [])
                print(f'    ✓ Found {len(results)} relevant results:')
                for j, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    content = result.get('content', '')[:80]
                    print(f'      {j}. Score: {score:.3f} - {content}...')
            else:
                print(f'    ❌ Search failed: {search_result.get('error', 'Unknown error')}')
        print('\n3. Searching with metadata filters...')
        filtered_search_result = query_tool(query='machine learning algorithms', top_k=5, similarity_threshold=0.1, metadata_filters={'topic': 'artificial_intelligence', 'difficulty': 'intermediate'})
        if filtered_search_result.get('success'):
            results = filtered_search_result.get('data', {}).get('results', [])
            print(f'✓ Found {len(results)} results with metadata filters:')
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                content = result.get('content', '')[:100]
                metadata = result.get('metadata', {})
                print(f'  {i}. Score: {score:.3f} - {content}...')
                print(f'     Metadata: {metadata}')
        else:
            print(f'❌ Filtered search failed: {filtered_search_result.get('error', 'Unknown error')}')
        print('\n4. Getting database statistics...')
        stats_result = stats_tool()
        if stats_result.get('success'):
            stats = stats_result.get('data', {})
            print(f'✓ Database statistics:')
            print(f'  - Total corpora: {stats.get('total_corpora', 'Unknown')}')
            print(f'  - Corpora: {', '.join(stats.get('corpora', []))}')
            print(f'  - Embedding model: {stats.get('embedding_model', 'Unknown')}')
            print(f'  - Vector store type: {stats.get('vector_store_type', 'Unknown')}')
        else:
            print(f'❌ Stats failed: {stats_result.get('error', 'Unknown error')}')
        print('\n5. Listing all corpora...')
        list_result = list_tool()
        if list_result.get('success'):
            corpora = list_result.get('data', {}).get('corpora', [])
            print(f'✓ Found {len(corpora)} corpora:')
            for corpus in corpora:
                corpus_id = corpus.get('corpus_id', 'Unknown')
                doc_count = corpus.get('document_count', 'Unknown')
                chunk_count = corpus.get('chunk_count', 'Unknown')
                print(f'  - {corpus_id}: {doc_count} documents, {chunk_count} chunks')
        else:
            print(f'❌ List failed: {list_result.get('error', 'Unknown error')}')
        print('\n6. Testing delete functionality...')
        delete_result = delete_tool(metadata_filters={'source': 'ai_knowledge_base'})
        if delete_result.get('success'):
            deleted_count = delete_result.get('data', {}).get('deleted_count', 0)
            print(f'✓ Deleted {deleted_count} documents with metadata filter')
            verify_result = query_tool(query='artificial intelligence', top_k=5, similarity_threshold=0.1)
            if verify_result.get('success'):
                remaining = verify_result.get('data', {}).get('total_results', 0)
                print(f'✓ Remaining documents after deletion: {remaining}')
        else:
            print(f'❌ Delete failed: {delete_result.get('error', 'Unknown error')}')
        print('\n✓ FAISS examples completed successfully!')
    except Exception as e:
        print(f'❌ Error running FAISS examples: {str(e)}')
        if 'DocumentMetadata' in str(e):
            print('Note: This appears to be a dependency issue with the RAG engine components')
            print('The FAISS toolkit may need additional setup or dependencies')

def main():
    """Main function to run all database tool examples."""
    print('===== DATABASE TOOLS EXAMPLES =====\n')
    run_mongodb_examples()
    run_postgresql_examples()
    run_faiss_examples()
    print('\n===== ALL DATABASE EXAMPLES COMPLETED =====')

def run_mongodb_examples():
    """Run examples using MongoDBToolkit for document database operations."""
    print('\n===== MONGODB TOOLKIT EXAMPLES =====\n')
    try:
        toolkit = MongoDBToolkit(name='DemoMongoDBToolkit', database_name='demo_db', auto_save=True)
        print('✓ MongoDBToolkit initialized with default storage')
        execute_tool = toolkit.get_tool('mongodb_execute_query')
        find_tool = toolkit.get_tool('mongodb_find')
        update_tool = toolkit.get_tool('mongodb_update')
        delete_tool = toolkit.get_tool('mongodb_delete')
        info_tool = toolkit.get_tool('mongodb_info')
        print(f'✓ Available tools: {[tool.name for tool in toolkit.get_tools()]}')
        print('\n1. Inserting product data...')
        products = [{'id': 'P001', 'name': 'Laptop', 'category': 'Electronics', 'price': 999.99, 'stock': 50, 'brand': 'TechCorp'}, {'id': 'P002', 'name': 'Wireless Mouse', 'category': 'Electronics', 'price': 29.99, 'stock': 100, 'brand': 'TechCorp'}, {'id': 'P003', 'name': 'Desk Chair', 'category': 'Furniture', 'price': 199.99, 'stock': 25, 'brand': 'ComfortCo'}, {'id': 'P004', 'name': 'Coffee Table', 'category': 'Furniture', 'price': 149.99, 'stock': 15, 'brand': 'ComfortCo'}, {'id': 'P005', 'name': 'Smartphone', 'category': 'Electronics', 'price': 799.99, 'stock': 75, 'brand': 'MobileTech'}]
        insert_result = execute_tool(query=json.dumps(products), query_type='insert', collection_name='products')
        if insert_result.get('success'):
            print(f'✓ Successfully inserted {len(products)} products')
            print(f'  Documents inserted: {insert_result.get('data', {}).get('inserted_count', 'Unknown')}')
        else:
            print(f'❌ Insert failed: {insert_result.get('error', 'Unknown error')}')
            return
        print('\n2. Finding electronics products...')
        find_result = find_tool(collection_name='products', filter='{"category": "Electronics"}', sort='{"price": -1}', limit=5)
        if find_result.get('success'):
            electronics = find_result.get('data', [])
            print(f'✓ Found {len(electronics)} electronics products:')
            for product in electronics:
                name = product.get('name', 'Unknown')
                price = product.get('price', 0)
                brand = product.get('brand', 'Unknown')
                print(f'  - {name}: ${price} ({brand})')
        else:
            print(f'❌ Find failed: {find_result.get('error', 'Unknown error')}')
        print('\n3. Updating product prices (10% discount on electronics)...')
        update_result = update_tool(collection_name='products', filter='{"category": "Electronics"}', update='{"$mul": {"price": 0.9}}', multi=True)
        if update_result.get('success'):
            updated_count = update_result.get('data', {}).get('modified_count', 0)
            print(f'✓ Updated {updated_count} electronics products with 10% discount')
        else:
            print(f'❌ Update failed: {update_result.get('error', 'Unknown error')}')
        print('\n4. Running aggregation query (average price by category)...')
        aggregation_pipeline = [{'$group': {'_id': '$category', 'avg_price': {'$avg': '$price'}, 'total_stock': {'$sum': '$stock'}}}, {'$sort': {'avg_price': -1}}]
        agg_result = execute_tool(query=json.dumps(aggregation_pipeline), query_type='aggregate', collection_name='products')
        if agg_result.get('success'):
            categories = agg_result.get('data', [])
            print(f'✓ Category analysis:')
            for category in categories:
                cat_name = category.get('_id', 'Unknown')
                avg_price = category.get('avg_price', 0)
                total_stock = category.get('total_stock', 0)
                print(f'  - {cat_name}: Avg price ${avg_price:.2f}, Total stock: {total_stock}')
        else:
            print(f'❌ Aggregation failed: {agg_result.get('error', 'Unknown error')}')
        print('\n5. Testing delete functionality...')
        delete_result = delete_tool(collection_name='products', filter='{"category": "Furniture"}', multi=True)
        if delete_result.get('success'):
            deleted_count = delete_result.get('data', {}).get('deleted_count', 0)
            print(f'✓ Deleted {deleted_count} furniture products')
        else:
            print(f'❌ Delete failed: {delete_result.get('error', 'Unknown error')}')
        print('\n6. Getting database information...')
        info_result = info_tool()
        if info_result.get('success'):
            info = info_result.get('data', {})
            print(f'✓ Database info:')
            print(f'  - Database: {info.get('database_name', 'Unknown')}')
            collections = info.get('collections', [])
            if isinstance(collections, (list, tuple)) and collections:
                print(f'  - Collections: {', '.join(collections)}')
            elif collections:
                print(f'  - Collections: {collections}')
            else:
                print('  - Collections: None')
            print(f'  - Total documents: {info.get('total_documents', 'Unknown')}')
        else:
            print(f'❌ Info failed: {info_result.get('error', 'Unknown error')}')
        print('\n✓ MongoDB examples completed successfully!')
    except Exception as e:
        print(f'❌ Error running MongoDB examples: {str(e)}')

def run_postgresql_examples():
    """Powerful example using PostgreSQLToolkit for database operations."""
    print('\n===== POSTGRESQL TOOL EXAMPLE =====\n')
    try:
        toolkit = PostgreSQLToolkit(name='DemoPostgreSQLToolkit', database_name='demo_db', auto_save=True)
        print('✓ PostgreSQLToolkit initialized with default storage')
        execute_tool = toolkit.get_tool('postgresql_execute')
        find_tool = toolkit.get_tool('postgresql_find')
        create_tool = toolkit.get_tool('postgresql_create')
        delete_tool = toolkit.get_tool('postgresql_delete')
        create_sql = '\n        CREATE TABLE IF NOT EXISTS users (\n            id SERIAL PRIMARY KEY,\n            name VARCHAR(100) NOT NULL,\n            email VARCHAR(100) UNIQUE NOT NULL,\n            age INTEGER,\n            department VARCHAR(50)\n        );\n        '
        result = create_tool(create_sql)
        if result['success']:
            print('✓ Created users table')
            insert_sql = "\n            INSERT INTO users (name, email, age, department) VALUES\n            ('Alice Johnson', 'alice@example.com', 28, 'Engineering'),\n            ('Bob Smith', 'bob@example.com', 32, 'Marketing'),\n            ('Carol Davis', 'carol@example.com', 25, 'Engineering')\n            "
            result = execute_tool(insert_sql)
            if result['success']:
                print('✓ Inserted users')
                find_result = find_tool('users', where="department = 'Engineering'", columns='name, age', sort='age ASC')
                if find_result['success']:
                    engineers = find_result['data']['data']
                    print(f'✓ Found {len(engineers)} engineers:')
                    for user in engineers:
                        name = user.get('name', 'Unknown')
                        age = user.get('age', 'N/A')
                        print(f'  - {name} (age: {age})')
                print('\n🗑️ Testing delete functionality...')
                delete_result = delete_tool('users', "department = 'Marketing'")
                if delete_result['success']:
                    deleted_count = delete_result['data'].get('rowcount', 0)
                    print(f'✓ Deleted {deleted_count} marketing users')
                    verify_result = find_tool('users')
                    if verify_result['success']:
                        remaining = verify_result['data']
                        print(f'✓ Remaining users after deletion: {len(remaining)}')
        print('\n✓ PostgreSQLToolkit test completed with default storage')
    except Exception as e:
        print(f'Error: {str(e)}')

