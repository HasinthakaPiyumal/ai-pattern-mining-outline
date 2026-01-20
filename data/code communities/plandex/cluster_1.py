# Cluster 1

def stream_generator():
    try:
        for chunk in response_stream:
            yield f'data: {json.dumps(chunk.to_dict())}\n\n'
        yield 'data: [DONE]\n\n'
    except Exception as e:
        yield f'data: {json.dumps({'error': str(e)})}\n\n'
        return
    finally:
        try:
            response_stream.close()
        except AttributeError:
            pass

