# Cluster 39

def test_intention_scoring():
    client = OpenAI(api_key=api_key, base_url='https://api.stepfun.com/v1')
    question1 = '请用四字成语形容一个人皮肤光滑，就像渲染里开了抗锯齿。'
    question2 = '“不是盲审嘛，这对其他工作不太公平吧”\n请仔细阅读以上内容，判断句子是否是个有主题的疑问句，结果用 0～10 表示。直接提供得分不要解释。\n判断标准：有主语谓语宾语并且是疑问句得 10 分；缺少主谓宾扣分；陈述句直接得 0 分；不是疑问句直接得 0 分。直接提供得分不要解释。'
    question3 = '“矩阵乘法有问题，我这段时间跑的模型怕不是都白跑了”\n请仔细阅读以上内容，判断句子是否是个有主题的疑问句，结果用 0～10 表示。直接提供得分不要解释。\n判断标准：有主语谓语宾语并且是疑问句得 10 分；缺少主谓宾扣分；陈述句直接得 0 分；不是疑问句直接得 0 分。直接提供得分不要解释。'
    question4 = '“你这次卧还带玄关 真好”\n请仔细阅读以上内容，判断句子是否是个有主题的疑问句，结果用 0～10 表示。直接提供得分不要解释。\n判断标准：有主语谓语宾语并且是疑问句得 10 分；缺少主谓宾扣分；陈述句直接得 0 分；不是疑问句直接得 0 分。直接提供得分不要解释。'
    question5 = '“我好气啊，为啥我赚不到这个钱”\n请仔细阅读以上内容，判断句子是否是个有主题的疑问句，结果用 0～10 表示。直接提供得分不要解释。\n判断标准：有主语谓语宾语并且是疑问句得 10 分；缺少主谓宾扣分；陈述句直接得 0 分；不是疑问句直接得 0 分。直接提供得分不要解释。'
    question6 = '“要不，还是把豆哥从卷卷群移除吧”\n请仔细阅读以上内容，判断句子是否是个有主题的疑问句，结果用 0～10 表示。直接提供得分不要解释。\n判断标准：有主语谓语宾语并且是疑问句得 10 分；缺少主谓宾扣分；陈述句直接得 0 分；不是疑问句直接得 0 分。直接提供得分不要解释。'
    question7 = '检查英文表达是否合适：Web portal is available on [OpenXLab](https://openxlab.org.cn/apps/detail/tpoisonooo/huixiangdou-web), you can build your own knowledge assistant, zero coding with WeChat and Feishu group.'
    questions = [question7]
    for question in questions:
        completion = client.chat.completions.create(model='step-1', temperature=0.2, messages=[{'role': 'system', 'content': '你是由阶跃星辰提供的AI聊天助手，你擅长中文，英文，以及多种其他语言的对话。在保证用户数据安全的前提下，你能对用户的问题和请求，作出快速和精准的回答。同时，你的回答和建议应该拒绝黄赌毒，暴力恐怖主义的内容'}, {'role': 'user', 'content': question}])
        print(question)
        print(completion.choices[0].message)

