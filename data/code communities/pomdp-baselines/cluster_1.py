# Cluster 1

def get_variant_tags(trial_str, max_episode_len):
    v = dict()
    if 'sac_' in trial_str:
        v['RL'] = 'sac'
    elif 'sacd_' in trial_str:
        v['RL'] = 'sacd'
    elif 'td3' in trial_str:
        v['RL'] = 'td3'
    if 'lstm' in trial_str:
        v['Encoder'] = 'lstm'
    elif 'gru' in trial_str:
        v['Encoder'] = 'gru'
    if 'shared' in trial_str:
        v['Arch'] = 'shared'
    else:
        v['Arch'] = 'separate'
    v['Len'] = int(trial_str[trial_str.index('len-') + 4:].split('/')[0])
    if v['Len'] == -1:
        v['Len'] = max_episode_len
    v['Inputs'] = trial_str.split('/')[-3]
    return v

