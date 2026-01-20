# Cluster 63

def encode_string(content: str, model_name: str='gpt-4o'):
    global ENCODER
    if ENCODER is None:
        tiktoken.get_encoding('cl100k_base')
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens

def decode_tokens(tokens: list[int], model_name: str='gpt-4o'):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content

def limit_async_func_call(max_size: int, waitting_time: float=0.1):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result
        return wait_func
    return final_decro

