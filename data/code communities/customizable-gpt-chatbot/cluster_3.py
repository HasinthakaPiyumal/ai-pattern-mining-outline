# Cluster 3

def get_pinecone_index(index_name, name_space):
    pinecone_manager = PineconeManager(PINECONE_API_KEY, PINECONE_ENVIRONMENT)
    pinecone_index_manager = PineconeIndexManager(pinecone_manager, index_name)
    try:
        pinecone_index = Pinecone.from_existing_index(index_name=pinecone_index_manager.index_name, embedding=embeddings, namespace=settings.PINECONE_NAMESPACE_NAME)
        return pinecone_index
    except Exception as e:
        logger.error(f'Failed to load Pinecone index: {e}')
        return None

@shared_task
def send_gpt_request(message_list, name_space, system_prompt):
    try:
        base_index = get_pinecone_index(PINECONE_INDEX_NAME, name_space)
        if base_index:
            last_message = message_list[-1]
            query_text = last_message['content']
            try:
                docs = base_index.similarity_search(query=last_message['content'], k=2)
                updated_content = '"""'
                for doc in docs:
                    updated_content += doc.page_content + '\n\n'
                updated_content += '"""\nQuestion:' + query_text
            except Exception as e:
                logger.error(f'Failed to get similar documents: {e}')
                updated_content = query_text
            updated_message = {'role': 'user', 'content': updated_content}
            message_list[-1] = updated_message
        openai.api_key = settings.OPENAI_API_KEY
        gpt3_response = openai.ChatCompletion.create(model='gpt-3.5-turbo-16k', messages=[{'role': 'system', 'content': f'{system_prompt}'}] + message_list)
        assistant_response = gpt3_response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f'Failed to send request to GPT-3.5: {e}')
        return "Sorry, I'm having trouble understanding you."
    return assistant_response

