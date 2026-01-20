# Cluster 6

def plot_graph_like_tree(G, root):
    pos = hierarchy_pos(G, root)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    plt.title('Retweet Tree')
    plt.show()

def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Compute the positions of all nodes in the tree starting from a given root
    node position
    """
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

class prop_graph:

    def __init__(self, source_post_content, db_path='', viz=False):
        self.source_post_content = source_post_content
        self.db_path = db_path
        self.viz = viz
        self.post_exist = False

    def build_graph(self):
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM post'
        df = pd.read_sql(query, conn)
        conn.close()
        all_reposts_and_time = []
        for i in range(len(df)):
            content = df.loc[i]['content']
            if self.post_exist is False and self.source_post_content[0:10] in content:
                self.post_exist = True
                self.root_id = df.loc[i]['user_id']
            if 'repost from' in content and self.source_post_content[0:10] in content:
                repost_history = content.split('. original_post: ')[:-1]
                repost_time = df.loc[i]['created_at']
                all_reposts_and_time.append((repost_history, repost_time))
        data = all_reposts_and_time
        start_time = 0
        self.G = nx.DiGraph()
        first_flag = 1
        for reposts, timestamp in data:
            time_diff = timestamp - start_time
            for repost in reposts:
                repost_info = repost.split(' repost from ')
                user = repost_info[0]
                original_user = repost_info[1]
                if first_flag:
                    self.root_id = original_user
                    first_flag = 0
                    if original_user not in self.G:
                        self.G.add_node(original_user, timestamp=0)
                if user not in self.G:
                    self.G.add_node(user, timestamp=time_diff)
                self.G.add_edge(original_user, user)
        self.start_timestamp = 0
        timestamps = nx.get_node_attributes(self.G, 'timestamp')
        try:
            self.end_timestamp = max(timestamps.values()) + 3
        except Exception as e:
            print(self.source_post_content)
            print(f'ERROR: {e}, may be caused by empty repost path')
            print(f'the simulation db is empty: {not self.post_exist}')
            print('Length of repost path:', len(all_reposts_and_time))
        self.total_depth = get_dpeth(self.G, source=self.root_id)
        self.total_scale = self.G.number_of_nodes()
        self.total_max_breadth = 0
        last_breadth_list = [1]
        for depth in range(self.total_depth):
            breadth = len(list(nx.bfs_tree(self.G, source=self.root_id, depth_limit=depth + 1).nodes())) - sum(last_breadth_list)
            last_breadth_list.append(breadth)
            if breadth > self.total_max_breadth:
                self.total_max_breadth = breadth
        undirect_G = self.G.to_undirected()
        self.total_structural_virality = nx.average_shortest_path_length(undirect_G)

    def viz_graph(self, time_threshold=10000):
        subG = get_subgraph_by_time(self.G, time_threshold)
        plot_graph_like_tree(subG, self.root_id)

    def plot_depth_time(self, separate_ratio: float=1):
        """
        Entire propagation process
        Detailed depiction of the data for the process before separate_ratio
        Rough depiction of the data afterwards
        Default to 1
        Use this parameter when the propagation time is very long, can be set
        to 0.01
        """
        depth_list = []
        self.d_t_list = list(range(int(self.start_timestamp), int(self.end_timestamp), 1))
        depth = 0
        for t in self.d_t_list:
            if depth < self.total_depth:
                try:
                    sub_g = get_subgraph_by_time(self.G, time_threshold=t)
                    depth = get_dpeth(sub_g, source=self.root_id)
                except Exception:
                    import pdb
                    pdb.set_trace()
            depth_list.append(depth)
        self.depth_list = depth_list
        if self.viz:
            _, ax = plt.subplots()
            ax.plot(self.d_t_list, self.depth_list)
            plt.title('Propagation depth-time')
            plt.xlabel('Time/minute')
            plt.ylabel('Depth')
            plt.show()
        else:
            return (self.d_t_list, self.depth_list)

    def plot_scale_time(self, separate_ratio: float=1.0):
        """
        Detailed depiction of the data between the start and separate_ratio*T
        of the entire propagation process
        Rough depiction of the data afterwards
        Default to 1
        Use this parameter when the propagation time is very long, can be set
        to 0.1
        """
        self.node_nums = []
        separate_point = int(int(self.start_timestamp) + separate_ratio * (int(self.end_timestamp) - int(self.start_timestamp)))
        self.s_t_list = list(range(int(self.start_timestamp), separate_point, 1))
        for t in self.s_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
                node_num = sub_g.number_of_nodes()
            except Exception:
                import pdb
                pdb.set_trace()
            self.node_nums.append(node_num)
        if self.viz:
            _, ax = plt.subplots()
            ax.plot(self.s_t_list, self.node_nums)
            plt.title('Propagation scale-time')
            plt.xlabel('Time/minute')
            plt.ylabel('Scale')
            plt.show()
        else:
            return (self.s_t_list, self.node_nums)

    def plot_max_breadth_time(self, interval=1):
        self.max_breadth_list = []
        self.b_t_list = list(range(int(self.start_timestamp), int(self.end_timestamp), interval))
        for t in self.b_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
            except Exception:
                import pdb
                pdb.set_trace()
            max_depth = self.depth_list[t - self.b_t_list[0]]
            max_breadth = 0
            last_breadth_list = [1]
            for depth in range(max_depth):
                breadth = len(list(nx.bfs_tree(sub_g, source=self.root_id, depth_limit=depth + 1).nodes())) - sum(last_breadth_list)
                last_breadth_list.append(breadth)
                if breadth > max_breadth:
                    max_breadth = breadth
            self.max_breadth_list.append(max_breadth)
        if self.viz:
            _, ax = plt.subplots()
            ax.plot(self.b_t_list, self.max_breadth_list)
            plt.title('Propagation max breadth-time')
            plt.xlabel('Time/minute')
            plt.ylabel('Max breadth')
            plt.show()
        else:
            return (self.b_t_list, self.max_breadth_list)

    def plot_structural_virality_time(self, interval=1):
        self.sv_list = []
        self.sv_t_list = list(range(int(self.start_timestamp), int(self.end_timestamp), interval))
        for t in self.sv_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
            except Exception:
                import pdb
                pdb.set_trace()
            sub_g = sub_g.to_undirected()
            sv = nx.average_shortest_path_length(sub_g)
            self.sv_list.append(sv)
        if self.viz:
            _, ax = plt.subplots()
            ax.plot(self.sv_t_list, self.sv_list)
            plt.title('Propagation structural virality-time')
            plt.xlabel('Time/minute')
            plt.ylabel('Structural virality')
            plt.show()
        else:
            return (self.sv_t_list, self.sv_list)

def get_dpeth(G: nx.Graph, source=0):
    dfs_tree = nx.dfs_tree(G, source=source)
    max_depth = max(nx.single_source_shortest_path_length(dfs_tree, source=source).values())
    return max_depth

def get_subgraph_by_time(G: nx.Graph, time_threshold=10):
    filtered_nodes = []
    for node, attr in G.nodes(data=True):
        try:
            if attr['timestamp'] <= time_threshold:
                filtered_nodes.append(node)
        except Exception:
            pass
    subG = G.subgraph(filtered_nodes)
    return subG

