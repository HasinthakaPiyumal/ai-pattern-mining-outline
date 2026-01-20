# Cluster 15

def test_chunk():
    c = Chunk()
    c_str = '{}'.format(c)
    assert 'content_or_path=' in c_str

def test_query():
    q = Query(text='hello', image='test.jpg')
    q_str = '{}'.format(q)
    assert 'hello' in q_str
    assert 'image=' in q_str
    p = Query('hello')
    p_str = '{}'.format(p)
    assert 'text=' in p_str

def test_reject(retriever: Retriever, sample: str=None):
    """Simple test reject pipeline."""
    if sample is None:
        real_questions = ['SAM 10个T 的训练集，怎么比比较公平呢~？速度上还有缺陷吧？', '想问下，如果只是推理的话，amp的fp16是不会省显存么，我看parameter仍然是float32，开和不开推理的显存占用都是一样的。能不能直接用把数据和model都 .half() 代替呢，相比之下amp好在哪里', 'mmdeploy支持ncnn vulkan部署么，我只找到了ncnn cpu 版本', '大佬们，如果我想在高空检测安全帽，我应该用 mmdetection 还是 mmrotate', '请问 ncnn 全称是什么', '有啥中文的 text to speech 模型吗?', '今天中午吃什么？', 'huixiangdou 是什么？', 'mmpose 如何安装？', '使用科研仪器需要注意什么？']
    else:
        with open(sample) as f:
            real_questions = json.load(f)
    for example in real_questions:
        relative, score = retriever.is_relative(example)
        if relative:
            logger.warning(f'process query: {example}')
        else:
            logger.error(f'reject query: {example}')
        if sample is not None:
            if relative:
                with open('workdir/positive.txt', 'a+') as f:
                    f.write(example)
                    f.write('\n')
            else:
                with open('workdir/negative.txt', 'a+') as f:
                    f.write(example)
                    f.write('\n')
    empty_cache()

def empty_cache():
    try:
        from torch.cuda import empty_cache as cuda_empty_cache
        cuda_empty_cache()
    except Exception as e:
        logger.error(e)

def test_query(retriever: Retriever, sample: str=None):
    """Simple test response pipeline."""
    from texttable import Texttable
    if sample is not None:
        with open(sample) as f:
            real_questions = json.load(f)
        logger.add('logs/feature_store_query.log', rotation='4MB')
    else:
        real_questions = ['百草园里有啥？', 'how to use std::vector ?']
    table = Texttable()
    table.set_cols_valign(['t', 't', 't', 't'])
    table.header(['Query', 'State', 'Chunks', 'References'])
    for example in real_questions:
        example = example[0:400]
        chunks, context, refs, context_texts = retriever.query(example)
        if chunks:
            table.add_row([example, 'Accepted', chunks[0:100] + '..', ','.join(refs)])
        else:
            table.add_row([example, 'Rejected', 'None', 'None'])
        empty_cache()
    logger.info('\n' + table.draw())
    empty_cache()

def test_reject(retriever: Retriever, sample: str=None):
    """Simple test reject pipeline."""
    if sample is None:
        real_questions = ['SAM 10个T 的训练集，怎么比比较公平呢~？速度上还有缺陷吧？', '想问下，如果只是推理的话，amp的fp16是不会省显存么，我看parameter仍然是float32，开和不开推理的显存占用都是一样的。能不能直接用把数据和model都 .half() 代替呢，相比之下amp好在哪里', 'mmdeploy支持ncnn vulkan部署么，我只找到了ncnn cpu 版本', '大佬们，如果我想在高空检测安全帽，我应该用 mmdetection 还是 mmrotate', '请问 ncnn 全称是什么', '有啥中文的 text to speech 模型吗?', '今天中午吃什么？', 'huixiangdou 是什么？', 'mmpose 如何安装？', '使用科研仪器需要注意什么？']
    else:
        with open(sample) as f:
            real_questions = json.load(f)
    for example in real_questions:
        relative, _ = retriever.is_relative(example)
        if relative:
            logger.warning(f'process query: {example}')
        else:
            logger.error(f'reject query: {example}')
        if sample is not None:
            if relative:
                with open('workdir/positive.txt', 'a+') as f:
                    f.write(example)
                    f.write('\n')
            else:
                with open('workdir/negative.txt', 'a+') as f:
                    f.write(example)
                    f.write('\n')
    empty_cache()

