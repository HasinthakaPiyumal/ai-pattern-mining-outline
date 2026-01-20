# Cluster 19

def main():
    if not os.getenv('OPENAI_API_KEY'):
        print('Please set your OPENAI_API_KEY environment variable')
        print('You can create a .env file with: OPENAI_API_KEY=your_api_key_here')
        return
    system = MultiAgentSystem()
    system.draw_and_save_graph()
    query = 'Given an m x n matrix, return all elements of the matrix in spiral order, where m = 1000000000 and n = 1000000000.'
    logger.info(f'Processing query:\n{query}')
    system.process_query(query)

