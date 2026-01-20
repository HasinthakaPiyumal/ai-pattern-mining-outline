# Cluster 124

def transform_gpt_to_trajectory(answer, agent, time, input_map=None, post_transform=(False, None), obj=None):
    python_file = 'work_dirs/created_python_file/traj_' + str(time) + '.py'
    if os.path.exists(python_file):
        os.remove(python_file)
    with open(python_file, 'w') as f:
        f.write(extract_python_code(answer))
    python_command = 'python ' + python_file
    result = os.popen(python_command)
    res = result.read()
    coordinates = ast.literal_eval(res)
    return coordinates

def extract_python_code(s):
    pattern = '```python(.*?)```'
    match = re.search(pattern, s, re.DOTALL)
    return match.group(1).strip()

def transform_coord_to_trajectory(answer, agent, time, input_map=None, post_transform=(False, None), obj=None):
    python_file = 'work_dirs/created_python_file/traj_' + str(time) + '.py'
    if os.path.exists(python_file):
        os.remove(python_file)
    with open(python_file, 'w') as f:
        f.write(extract_python_code(answer))
    python_command = 'python ' + python_file
    result = os.popen(python_command)
    res = result.read()
    coordinates = ast.literal_eval(res)
    return coordinates

