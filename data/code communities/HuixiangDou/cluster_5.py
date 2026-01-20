# Cluster 5

def read_badcase(llm_type: str, input_filepath: str):
    gts = []
    dts = []
    unknow_count = 0
    with open('groups/input.jsonl') as gt:
        for line in gt:
            json_obj = json.loads(line)
            if 'cr_need_gt' not in json_obj:
                continue
            cr_need_gt = json_obj['cr_need_gt']
            gts.append(cr_need_gt)
    ret = dict()
    idx = 0
    with open(input_filepath) as dt:
        for line in dt:
            json_obj = json.loads(line)
            if 'cr_need_gt' not in json_obj:
                continue
            dt = json_obj['{}_cr_need'.format(llm_type)] == 'yes'
            if dt != gts[idx]:
                ret[json_obj['text']] = line
            idx += 1
    return ret

