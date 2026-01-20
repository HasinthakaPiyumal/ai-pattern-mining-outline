# Cluster 3

def randomize_labels(true_label):
    options = ['A', 'B', 'C']
    answers = ['不需要提取，信息完整', '需要提取', '不知道']
    random.shuffle(answers)
    correct_answer_index = answers.index('需要提取') if true_label == True else answers.index('不需要提取，信息完整')
    option_str = ''
    for index in range(len(options)):
        option_str += '{}:{}  '.format(options[index], answers[index])
    option_str = option_str.strip()
    gt_str = '{}:{}'.format(options[correct_answer_index], answers[correct_answer_index])
    return (option_str, gt_str)

def convert_alpaca(input_filepath: str, output_filepath: str):
    gts = []
    with open(input_filepath) as gt:
        for line in gt:
            target = json.loads(line)
            if 'cr_need_gt' not in target:
                continue
            gt_bool = target['cr_need_gt']
            text = target['text']
            window = target['cr_window']
            name_map = dict()
            name_int = ord('A')
            format_history = []
            for item in window:
                sender = item['sender']
                if sender not in name_map:
                    name_map[sender] = chr(name_int)
                    name_int += 1
                format_history.append({'username': name_map[sender], 'content': item['text']})
            target_sender = target['sender']
            if target_sender not in name_map:
                name_map[target_sender] = chr(name_int)
                name_int += 1
            target_str = json.dumps({'username': name_map[target_sender], 'content': target['text']}, indent=2, ensure_ascii=False)
            BASE_PROMPT_TEMPLATE = '群聊场景中“这”、“它”、“哪”等代词需要查看上下文和其他用户的回复才能确定具体指什么，请完成群聊场景代词替换任务。\n以下是历史对话，可能有多个人的发言：\n{}\n输入内容：\n"{}"\n'
            prompt_base = BASE_PROMPT_TEMPLATE.format(json.dumps(format_history, ensure_ascii=False), target_str)
            option_str, output = randomize_labels(gt_bool)
            instruction = '{} 输入内容中的 content 信息是否完整，是否需要从历史对话中提取代词或宾语来替代 content 中的一部分词汇？ {} \n一步步分析，首先历史消息包含哪些话题；其次哪个话题与问题最相关；如果都不相关就不提取。 '.format(prompt_base, option_str)
            gts.append({'instruction': instruction, 'input': '', 'output': output})
    alpaca_str = json.dumps(gts, ensure_ascii=False, indent=2)
    with open(output_filepath, 'w') as fout:
        fout.write(alpaca_str)

