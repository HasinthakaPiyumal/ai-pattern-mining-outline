# Cluster 66

def plot_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for jsonl_file in os.listdir('./'):
        if not jsonl_file.endswith('.jsonl'):
            continue
        if not 'chunk_size' in jsonl_file:
            continue
        x = []
        y = []
        z = []
        print(jsonl_file)
        datas = []
        with open(jsonl_file) as f:
            for json_str in f:
                json_obj = json.loads(json_str)
                datas.append(json_obj)
        datas.sort(key=lambda x: x['throttle'])
        for data in datas:
            chunk_size = data['chunk_size']
            throttle = data['throttle']
            f1 = data['f1']
            x.append(chunk_size)
            y.append(throttle)
            z.append(f1)
        ax.plot(x, y, z)
    ax.set_title('3D Line Plot')
    ax.set_xlabel('chunk_size')
    ax.set_ylabel('throttle')
    ax.set_zlabel('f1')
    plt.show()

