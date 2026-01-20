# Cluster 82

class ProgressionVisualizer:
    """Creates publication-quality visualizations of agent progression"""
    VERTICAL_SPACING_PIXELS = 12
    HORIZONTAL_OFFSET_PIXELS = 0

    def __init__(self, db_client, icons_path: str, x_axis: Literal['steps', 'ticks']='steps', cache_file: str='viz_cache.pkl', x_base: float=10, y_base: float=10, use_value_gdp=False, recipes_file='recipes.jsonl', use_log_scale: bool=True):
        self.db_client = db_client
        self.icons_path = icons_path
        self.x_axis = x_axis
        self.cache_file = cache_file
        self.x_base = x_base
        self.y_base = y_base
        self.versions = {}
        self.achievements = defaultdict(list)
        self.colors = ['#8fd7d7', '#FFCD8E', '#00b0be', '#ff8ca1', '#f45f74', '#bdd373', '#98c127', '#ffb255']
        self.use_value_gdp = use_value_gdp
        self.value_calculator = ValueCalculator(recipes_file) if use_value_gdp else None
        self.use_log_scale = use_log_scale

    def _serialize_version_data(self):
        """Convert version data to cacheable format"""
        serialized = {}
        for version, data in self.versions.items():
            nodes_dict = {}
            for root in data['nodes']:
                stack = [root]
                while stack:
                    node = stack.pop()
                    nodes_dict[node.id] = node.to_dict()
                    stack.extend(node.children)
            serialized[version] = {'nodes_dict': nodes_dict, 'root_ids': [root.id for root in data['nodes']], 'label': data['label']}
        return serialized

    def _deserialize_version_data(self, serialized):
        """Restore version data from cached format"""
        for version, data in serialized.items():
            nodes_dict = {}
            for node_id, node_data in data['nodes_dict'].items():
                nodes_dict[node_id] = Node(id=node_data['id'], parent_id=node_data['parent_id'], metrics=node_data['metrics'], static_achievements=node_data['static_achievements'], dynamic_achievements=node_data['dynamic_achievements'], children=[])
            for node_id, node_data in data['nodes_dict'].items():
                nodes_dict[node_id].children = [nodes_dict[child_id] for child_id in node_data['children_ids']]
            self.versions[version] = {'nodes': [nodes_dict[root_id] for root_id in data['root_ids']], 'label': data['label']}

    def load_data(self, version_groups: Dict[str, List[int]], labels: Dict[str, str]):
        """
        Load and process data for multiple version groups, using cache if available

        Args:
            version_groups: Dict mapping model names to lists of version numbers
            labels: Dict mapping model names to display labels
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data.get('version_groups') == version_groups:
                        self._deserialize_version_data(cached_data['data'])
                        self.achievements = cached_data['achievements']
                        return
            except Exception as e:
                print(f'Error loading cache: {e}')
                os.remove(self.cache_file)
        for model_name, versions in version_groups.items():
            print(f'\nLoading model {model_name} (versions: {versions})')
            all_nodes = []
            for version in versions:
                nodes = self._load_version_from_db(version)
                if nodes:
                    all_nodes.extend(nodes)
            if all_nodes:
                gdps = [self._calculate_gdp(root) for root in all_nodes]
                print(f'Mean GDP across all versions: {np.mean(gdps):.1f}')
                print(f'STD GDP across all versions: {np.std(gdps):.1f}')
                self.versions[model_name] = {'nodes': all_nodes, 'label': f'{labels[model_name]}'}
                self._process_merged_achievements(model_name, versions)
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({'version_groups': version_groups, 'data': self._serialize_version_data(), 'achievements': self.achievements}, f)
        except Exception as e:
            print(f'Error saving cache: {e}')

    def _process_merged_achievements(self, model_name: str, versions: List[int]):
        """Process achievements across multiple versions of the same model"""
        print(f'\nProcessing achievements for model {model_name}')
        with open('recipes.jsonl', 'r') as f:
            recipes = {r['name']: r for r in map(json.loads, f)}
        seen = set()
        model_achievements = []
        for version in versions:
            for root in self._load_version_from_db(version):
                current_path = []
                stack = [(root, 0, [])]
                while stack:
                    node, depth, path = stack.pop()
                    current_path = path + [node]
                    path_ticks = sum((n.metrics['ticks'] for n in current_path))
                    for achievements_dict, is_dynamic in [(node.static_achievements, False), (node.dynamic_achievements, True)]:
                        for item, quantity in achievements_dict.items():
                            achievement_key = (version, item)
                            if achievement_key not in seen:
                                print(f'\nProcessing achievement: {item} (version {version})')
                                print(f'Original ticks: {path_ticks}, depth: {depth}')
                                model_achievements.append(Achievement(depth=depth, ticks=path_ticks, item_name=item, ingredients=self._count_ingredients(recipes.get(item, {})), is_dynamic=is_dynamic))
                                seen.add(achievement_key)
                    for child in reversed(node.children):
                        stack.append((child, depth + 1, current_path))
        earliest_achievements = {}
        for achievement in model_achievements:
            key = achievement.item_name
            if key not in earliest_achievements or (achievement.ticks < earliest_achievements[key].ticks or (achievement.ticks == earliest_achievements[key].ticks and achievement.depth < earliest_achievements[key].depth)):
                earliest_achievements[key] = achievement
        self.achievements[model_name] = list(earliest_achievements.values())
        print(f'Total unique achievements processed for model {model_name}: {len(earliest_achievements)}')

    def load_data2(self, versions: List[int], labels: Dict[int, str]):
        """Load and process data for multiple versions, using cache if available"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data.get('versions') == versions:
                        self._deserialize_version_data(cached_data['data'])
                        self.achievements = cached_data['achievements']
                        return
            except Exception as e:
                print(f'Error loading cache: {e}')
                os.remove(self.cache_file)
        for version in versions:
            print(f'\nLoading version {version}')
            nodes = self._load_version_from_db(version)
            if nodes:
                gdps = [self._calculate_gdp(root) for root in nodes]
                print(f'Mean {np.mean(gdps):.1f}')
                print(f'STD {np.std(gdps):.1f}')
                self.versions[version] = {'nodes': nodes, 'label': f'{labels[version]}'}
                self._process_achievements(version)
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({'versions': versions, 'data': self._serialize_version_data(), 'achievements': self.achievements}, f)
        except Exception as e:
            print(f'Error saving cache: {e}')

    def organize_achievement_positions(self, achievements_by_depth, depth_stats, ax, series_index, used_positions):
        """Organize achievement positions with improved tick mapping"""
        positions = {}
        final_positions = {}
        for x_coord, achievements_list in achievements_by_depth.items():
            if x_coord in depth_stats and depth_stats[x_coord]['mean'] > 0:
                base_position = depth_stats[x_coord]['mean']
                positions[x_coord] = {'achievements': achievements_list, 'base_y': base_position}
        for x_coord, group_data in positions.items():
            achievements = group_data['achievements']
            base_y = group_data['base_y']
            achievements.sort(key=lambda a: (a.ingredients, a.item_name))
            display_coords = ax.transData.transform([[x_coord, base_y]])[0]
            base_display_x, base_display_y = display_coords
            x_key = round(x_coord, -3 if self.x_axis == 'ticks' else 1)
            if x_key in used_positions:
                base_display_x += self.HORIZONTAL_OFFSET_PIXELS * (series_index + 1)
            else:
                used_positions[x_key] = True
            for i, achievement in enumerate(achievements):
                vertical_offset = i * self.VERTICAL_SPACING_PIXELS
                display_y = base_display_y + vertical_offset
                data_coords = ax.transData.inverted().transform([[base_display_x, display_y]])[0]
                data_x, data_y = data_coords
                achievement_key = (achievement.item_name, x_coord)
                final_positions[achievement_key] = (data_x, data_y)
        return final_positions

    def _load_version_from_db(self, version: int) -> List[Node]:
        """Load all trajectories for a version from database"""
        with self.db_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('\n                    SELECT id, parent_id, achievements_json, value, ticks \n                    FROM programs WHERE version = %s\n                ', (version,))
                rows = cur.fetchall()
        nodes = {}
        roots = []
        for id, parent_id, achievements, value, ticks in rows:
            node = Node(id=id, parent_id=parent_id, metrics={'value': value or 0, 'ticks': ticks or 0}, static_achievements=achievements.get('static', {}), dynamic_achievements=achievements.get('dynamic', {}), children=[])
            nodes[id] = node
        for node in nodes.values():
            if node.parent_id is None:
                roots.append(node)
            elif node.parent_id in nodes:
                nodes[node.parent_id].children.append(node)
        return roots

    def _calculate_gdp(self, root: Node) -> float:
        """Calculate GDP for a trajectory using either method"""
        if not self.use_value_gdp:
            'Calculate final GDP for a trajectory'
            total = 0
            stack = [root]
            while stack:
                node = stack.pop()
                total += node.metrics['value']
                stack.extend(node.children)
            return total
        total_value = 0
        stack = [root]
        all_achievements = defaultdict(int)
        while stack:
            node = stack.pop()
            for item, quantity in node.static_achievements.items():
                all_achievements[item] += quantity
            for item, quantity in node.dynamic_achievements.items():
                all_achievements[item] += quantity
            stack.extend(node.children)
        print('\nValue-based GDP calculation:')
        for item, quantity in all_achievements.items():
            item_value = self.value_calculator.get_value(item)
            contribution = item_value * quantity
            total_value += contribution
            print(f'  {item}: {quantity} x {item_value:.2f} = {contribution:.2f}')
        print(f'Total value-based GDP: {total_value}')
        return total_value

    def _process_achievements(self, version: int):
        """Process achievements with correct tick accumulation"""
        print(f'\nProcessing achievements for version {version}')
        with open('recipes.jsonl', 'r') as f:
            recipes = {r['name']: r for r in map(json.loads, f)}
        seen = set()
        version_achievements = []
        for root in self.versions[version]['nodes']:
            current_path = []
            stack = [(root, 0, [])]
            while stack:
                node, depth, path = stack.pop()
                current_path = path + [node]
                path_ticks = sum((n.metrics['ticks'] for n in current_path))
                for achievements_dict, is_dynamic in [(node.static_achievements, False), (node.dynamic_achievements, True)]:
                    for item, quantity in achievements_dict.items():
                        if item not in seen:
                            print(f'\nProcessing achievement: {item}')
                            print(f'Original ticks: {path_ticks}, depth: {depth}')
                            version_achievements.append(Achievement(depth=depth, ticks=path_ticks, item_name=item, ingredients=self._count_ingredients(recipes.get(item, {})), is_dynamic=is_dynamic))
                            seen.add(item)
                for child in reversed(node.children):
                    stack.append((child, depth + 1, current_path))
        self.achievements[version] = version_achievements
        print(f'Total achievements processed for version {version}: {len(seen)}')

    def _count_ingredients(self, recipe: Dict) -> int:
        """Count total unique ingredients in recipe"""
        seen = set()
        if not recipe:
            return 1

        def traverse(item):
            seen.add(item['name'])
            for ingredient in item.get('ingredients', []):
                traverse(ingredient)
        traverse(recipe)
        return len(seen) - 1

    def _prepare_icon(self, item_name: str):
        """Prepare achievement icon for visualization with size limits"""
        src = os.path.join(self.icons_path, f'{item_name}.png')
        dst = f'icons/{item_name}.png'
        if os.path.exists(src) and (not os.path.exists(dst)):
            os.makedirs('icons', exist_ok=True)
            with Image.open(src) as img:
                if hasattr(img, 'n_frames'):
                    img.seek(0)
                size = min(img.width, img.height)
                tile = img.crop((0, 0, size, size))
                if size > 256:
                    tile = tile.resize((256, 256), Image.Resampling.LANCZOS)
                if tile.mode != 'RGBA':
                    tile = tile.convert('RGBA')
                tile.save(dst, format='PNG')

    def weighted_percentile(self, values, weights, q):
        """Calculate weighted percentile for confidence intervals"""
        order = np.argsort(values)
        values = np.array(values)[order]
        weights = np.array(weights)[order]
        cumsum = np.cumsum(weights)
        cumsum = cumsum / cumsum[-1]
        return np.interp(q / 100, cumsum, values)

    def create_circle_background(self, color, alpha=0.7):
        """Create a circular background image with transparency"""
        size = 50
        img = np.zeros((size, size, 4))
        y, x = np.ogrid[-size / 2:size / 2, -size / 2:size / 2]
        mask = x * x + y * y <= size / 2 * (size / 2)
        img[..., :3][mask] = 1
        img[..., 3][mask] = alpha
        edge = np.zeros_like(mask)
        edge_width = 3
        for i in range(edge_width):
            edge_mask = (x * x + y * y <= (size / 2 - i) * (size / 2 - i)) & (x * x + y * y >= (size / 2 - i - 1) * (size / 2 - i - 1))
            edge |= edge_mask
        rgb_color = np.array([int(color[1:][i:i + 2], 16) / 255 for i in (0, 2, 4)])
        img[..., :3][edge] = rgb_color
        img[..., 3][edge] = 0.7
        return img

    def _get_step_at_ticks(self, version, target_ticks):
        """Find the step number at which a version reaches the target ticks"""
        if target_ticks <= 0 or np.isinf(target_ticks) or np.isnan(target_ticks):
            return None
        nodes = self.versions[version]['nodes']
        min_step = float('inf')
        for root in nodes:
            current_ticks = 0
            stack = [(root, 0, 0)]
            while stack:
                node, step, prev_ticks = stack.pop()
                current_ticks = prev_ticks + node.metrics['ticks']
                if current_ticks >= target_ticks:
                    min_step = min(min_step, step)
                    break
                for child in node.children:
                    stack.append((child, step + 1, current_ticks))
        return min_step if min_step != float('inf') else None

    def add_connecting_lines(self, ax1, ax2, step=990):
        y_min, y_max = ax2.get_ylim()
        fig = ax1.get_figure()
        ax1_bbox = ax1.get_position()
        ax2_bbox = ax2.get_position()
        x1 = ax1_bbox.x1
        x2 = ax2_bbox.x0
        y1_min = ax1.transData.transform([[0, y_min]])[0, 1]
        y1_min = fig.transFigure.inverted().transform([[0, y1_min]])[0, 1]
        y1_max = ax1.transData.transform([[0, y_max]])[0, 1]
        y1_max = fig.transFigure.inverted().transform([[0, y1_max]])[0, 1]
        y2_min = ax2.transData.transform([[0, y_min]])[0, 1]
        y2_min = fig.transFigure.inverted().transform([[0, y2_min]])[0, 1]
        y2_max = ax2.transData.transform([[0, y_max]])[0, 1]
        y2_max = fig.transFigure.inverted().transform([[0, y2_max]])[0, 1]
        polygon = plt.Polygon([[x1, y1_min], [x2, y2_min], [x2, y2_max], [x1, y1_max]], transform=fig.transFigure, facecolor='gray', alpha=0.15, edgecolor='#404040', linestyle='--', linewidth=1)
        fig.add_artist(polygon)

    def export_split_visualization(self, output_file: str, max_depth: int=4950):
        """Export a visualization with main progression chart and final GDP scatter plot"""
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        fig = plt.figure(figsize=(12, 5))
        gs = plt.GridSpec(1, 2, width_ratios=[15, 1], wspace=0.1)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax1.set_xscale('log', base=self.x_base)
        ax1.set_yscale('log', base=self.y_base)
        if self.x_axis == 'ticks':
            ax1.set_xlim(1000.0, 100000000.0)
            ax1.set_xlabel('Ticks', fontsize=12)
        else:
            ax1.set_xlim(100, 5000.0)
            ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylim(10.0, 1000000.0)
        ax1.set_ylabel('Production Score', fontsize=12)
        final_values = []
        final_cis = []
        colors = []
        used_positions = {}
        for idx, (version, data) in enumerate(self.versions.items()):
            color = self.colors[idx % len(self.colors)]
            colors.append(color)
            stats = self._calculate_statistics(data['nodes'], max_depth)
            x_coords = sorted(stats.keys())
            means = [stats[x]['mean'] for x in x_coords]
            ci_lower = [stats[x]['ci_lower'] for x in x_coords]
            ci_upper = [stats[x]['ci_upper'] for x in x_coords]
            ax1.plot(x_coords, means, color=color, label=data['label'], linewidth=1.5)
            ax1.fill_between(x_coords, ci_lower, ci_upper, color=color, alpha=0.2)
            if means:
                target_step = 2950
                if self.x_axis == 'steps':
                    target_idx = next((idx for idx, x in enumerate(x_coords) if x >= target_step), -1)
                else:
                    target_idx = next((idx for idx, x in enumerate(x_coords) if x >= target_step), -1)
                if target_idx != -1:
                    final_values.append(means[target_idx])
                    final_cis.append((ci_lower[target_idx], ci_upper[target_idx]))
                else:
                    final_values.append(means[-1])
                    final_cis.append((ci_lower[-1], ci_upper[-1]))
            achievements_by_depth = defaultdict(list)
            for achievement in self.achievements[version]:
                if self.x_axis == 'ticks':
                    x_coord = achievement.ticks
                    bucket = round(np.log(x_coord) / np.log(self.x_base) * 10) / 10
                else:
                    x_coord = achievement.depth
                    bucket = x_coord
                if x_coord > 0:
                    achievements_by_depth[bucket].append(achievement)
            positions = self.organize_achievement_positions(achievements_by_depth, stats, ax1, idx, used_positions)
            for (item_name, orig_x), (x, y) in positions.items():
                if x > 0 and y > 0:
                    x_min, x_max = ax1.get_xlim()
                    y_min, y_max = ax1.get_ylim()
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        self._add_achievement_icon(ax1, item_name, x, y, color)
        ax1.grid(True, which='major', linestyle='-', color='gray', alpha=0.2)
        ax1.grid(True, which='minor', linestyle='--', color='gray', alpha=0.1)
        ax1.tick_params(axis='both', which='major', labelsize=9)
        ax1.tick_params(axis='both', which='minor', labelsize=7)
        ax1.set_axisbelow(True)
        ax1.legend(loc='lower right', fontsize=10)
        x_pos = np.arange(len(final_values))
        x_margin = 0.5
        ax2.set_xlim(-x_margin, len(final_values) - 1 + x_margin)
        for i, (value, (ci_lower, ci_upper), color) in enumerate(zip(final_values, final_cis, colors)):
            ax2.vlines(i, ci_lower, ci_upper, color=color, alpha=0.5)
            ax2.scatter(i, value, color=color, s=50, zorder=5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([''] * len(x_pos))
        ax2.set_yscale('log', base=self.y_base)
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
        ax2.set_xlabel('Final Step', fontsize=12, labelpad=15)
        ax2.grid(True, which='major', axis='y', linestyle='-', color='gray', alpha=0.2)
        ax2.grid(True, which='minor', axis='y', linestyle='--', color='gray', alpha=0.1)
        ax2.set_ylim(30000.0, 500000.0)
        min_val = max(1e-10, min((ci[0] for ci in final_cis)) * 0.9)
        max_val = max((ci[1] for ci in final_cis)) * 1.1
        ax2.set_ylim(min_val, max_val)
        ax2.set_yticks([min_val, max_val])
        ax2.set_yticklabels([f'{int(min_val / 1000):,}k', f'{int(max_val / 1000):,}k'])
        ax2.yaxis.set_minor_locator(LogLocator(base=self.y_base, subs=np.arange(2, 10) * 0.1, numticks=10))
        self.add_connecting_lines(ax1, ax2)
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

    def _calculate_statistics(self, roots: List[Node], max_depth: int) -> Dict:
        """Calculate statistics with cumulative achievement tracking"""
        values_by_x = defaultdict(list)
        for root in roots:
            values_by_x[0].append(0)
            stack = [(root, 0, 0, 0, defaultdict(int))]
            while stack:
                node, depth, prev_ticks, prev_value, prev_achievements = stack.pop()
                if depth > max_depth:
                    continue
                current_achievements = prev_achievements.copy()
                for item, quantity in node.static_achievements.items():
                    current_achievements[item] += quantity
                for item, quantity in node.dynamic_achievements.items():
                    current_achievements[item] += quantity
                if self.use_value_gdp:
                    current_value = 0
                    for item, quantity in current_achievements.items():
                        item_value = self.value_calculator.get_value(item)
                        current_value += item_value * quantity
                else:
                    current_value = prev_value + node.metrics['value']
                ticks = prev_ticks + node.metrics['ticks']
                x_coord = ticks if self.x_axis == 'ticks' else depth
                values_by_x[x_coord].append(current_value)
                for child in node.children:
                    stack.append((child, depth + 1, ticks, current_value, current_achievements))
        stats = {0: {'mean': 0, 'ci_lower': 0, 'ci_upper': 0, 'std': 0}}
        x_coords = sorted((x for x in values_by_x.keys() if x > 0))
        if self.x_axis == 'ticks':
            eval_points = np.logspace(np.log(min(x_coords)) / np.log(self.x_base), np.log(max(x_coords)) / np.log(self.x_base), 500, base=self.x_base)
            window = 0.1
            for x in eval_points:
                nearby_values = []
                for orig_x, values in values_by_x.items():
                    if orig_x > 0:
                        log_diff = abs(np.log(x) / np.log(self.x_base) - np.log(orig_x) / np.log(self.x_base))
                        if log_diff < window:
                            weight = np.exp(-(log_diff / window) ** 2)
                            nearby_values.extend(((v, weight) for v in values))
                if nearby_values:
                    values, weights = zip(*nearby_values)
                    stats[x] = {'mean': np.average(values, weights=weights), 'std': np.std(values), 'ci_lower': self.weighted_percentile(values, weights, 2.5), 'ci_upper': self.weighted_percentile(values, weights, 97.5)}
        else:
            for x in x_coords:
                values = values_by_x[x]
                if values:
                    stats[x] = {'mean': np.mean(values), 'std': np.std(values), 'ci_lower': np.percentile(values, 2.5), 'ci_upper': np.percentile(values, 97.5)}
        prev_stats = {'mean': 0, 'ci_lower': 0, 'ci_upper': 0, 'std': 0}
        for x in sorted(stats.keys()):
            stats[x] = {'mean': max(stats[x]['mean'], prev_stats['mean']), 'ci_lower': max(stats[x]['ci_lower'], prev_stats['ci_lower']), 'ci_upper': max(stats[x]['ci_upper'], prev_stats['ci_upper']), 'std': stats[x]['std']}
            prev_stats = stats[x]
        return stats

    def _add_achievement_icon(self, ax, item_name: str, x: float, y: float, color: str):
        """Add achievement icon with background circle"""
        try:
            self._prepare_icon(item_name)
            icon_path = f'icons/{item_name}.png'
            if not os.path.exists(icon_path):
                return
            circle_img = self.create_circle_background(color)
            circle_box = OffsetImage(circle_img, zoom=0.2)
            circle_box.image.axes = ax
            ab_circle = AnnotationBbox(circle_box, (x, y), frameon=False, box_alignment=(0.5, 0.5), pad=0)
            ax.add_artist(ab_circle)
            icon = plt.imread(icon_path)
            icon_box = OffsetImage(icon, zoom=0.1)
            ab = AnnotationBbox(icon_box, (x, y), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f'Failed to add icon for {item_name}: {e}')

class ProgressionVisualizer:
    """Creates publication-quality visualizations of agent progression"""
    VERTICAL_SPACING_PIXELS = 12
    HORIZONTAL_OFFSET_PIXELS = 0

    def __init__(self, db_client, icons_path: str, x_axis: Literal['steps', 'ticks']='steps', cache_file: str='viz_cache_combined.pkl', x_base: float=10, y_base: float=10, use_value_gdp=False, recipes_file='recipes.jsonl', use_log_scale: bool=True):
        self.db_client = db_client
        self.icons_path = icons_path
        self.x_axis = x_axis
        self.cache_file = cache_file
        self.x_base = x_base
        self.y_base = y_base
        self.model_groups = {}
        self.versions = {}
        self.labels_count = {}
        self.achievements = defaultdict(list)
        self.colors = ['#8fd7d7', '#FFCD8E', '#00b0be', '#ff8ca1', '#f45f74', '#bdd373', '#98c127', '#ffb255']
        self.use_value_gdp = use_value_gdp
        self.value_calculator = ValueCalculator(recipes_file) if use_value_gdp else None
        self.use_log_scale = use_log_scale

    def _serialize_version_data(self):
        """Convert version data to cacheable format"""
        serialized = {}
        for version, data in self.versions.items():
            nodes_dict = {}
            for root in data['nodes']:
                stack = [root]
                while stack:
                    node = stack.pop()
                    nodes_dict[node.id] = node.to_dict()
                    stack.extend(node.children)
            serialized[version] = {'nodes_dict': nodes_dict, 'root_ids': [root.id for root in data['nodes']], 'label': data['label']}
        return serialized

    def _deserialize_version_data(self, serialized):
        """Restore version data from cached format"""
        for version, data in serialized.items():
            nodes_dict = {}
            for node_id, node_data in data['nodes_dict'].items():
                nodes_dict[node_id] = Node(id=node_data['id'], parent_id=node_data['parent_id'], metrics=node_data['metrics'], static_achievements=node_data['static_achievements'], dynamic_achievements=node_data['dynamic_achievements'], children=[])
            for node_id, node_data in data['nodes_dict'].items():
                nodes_dict[node_id].children = [nodes_dict[child_id] for child_id in node_data['children_ids']]
            self.versions[version] = {'nodes': [nodes_dict[root_id] for root_id in data['root_ids']], 'label': data['label']}

    def load_data(self, model_groups: Dict[str, List[int]], model_labels: Dict[str, str]):
        """
        Load and process data for multiple model groups

        Args:
            model_groups: Dict mapping model names to lists of version IDs
            model_labels: Dict mapping model names to display labels
        """
        print('\nStarting data load process...')
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if all([cached_data.get('model_groups')[model].version_ids == model_groups[model] for model in model_groups.keys()]):
                        print('Found matching cache data, loading...')
                        self._deserialize_version_data(cached_data['data'])
                        self.achievements = cached_data['achievements']
                        self.model_groups = cached_data['model_groups']
                        for model_name, group in self.model_groups.items():
                            print(f'\nModel: {model_name}')
                            self.labels_count[model_name] = len(group.version_ids)
                            for version_id in group.version_ids:
                                if version_id in self.versions:
                                    print(f'  Version {version_id}: {len(self.versions[version_id]['nodes'])} nodes')
                                else:
                                    print(f'  Version {version_id}: Not found in data')
                        return
            except Exception as e:
                print(f'Error loading cache: {e}')
                os.remove(self.cache_file)
        print('\nNo valid cache found, loading from database...')
        for idx, (model_name, version_ids) in enumerate(model_groups.items()):
            color = self.colors[idx % len(self.colors)]
            self.model_groups[model_name] = ModelVersionGroup(model_name=model_name, version_ids=version_ids, color=color, label=model_labels[model_name])
        total_versions = sum((len(group.version_ids) for group in self.model_groups.values()))
        versions_loaded = 0
        for model_name, group in self.model_groups.items():
            print(f'\nLoading data for model: {model_name}')
            for version in group.version_ids:
                print(f'  Loading version {version}...')
                nodes = self._load_version_from_db(version)
                versions_loaded += 1
                print(f'  Progress: {versions_loaded}/{total_versions} versions')
                if nodes:
                    print(f'  Found {len(nodes)} root nodes')
                    self.versions[version] = {'nodes': nodes, 'label': f'{group.label} (v{version})'}
                    if group.label not in self.labels_count:
                        self.labels_count[group.label] = 0
                    self.labels_count[group.label] += len(nodes)
                    self._process_achievements(version)
                else:
                    print(f'  No nodes found for version {version}')
        print('\nData loading complete. Summary:')
        for model_name, group in self.model_groups.items():
            print(f'\nModel: {model_name}')
            success_count = sum((1 for v in group.version_ids if v in self.versions))
            print(f'Successfully loaded {success_count}/{len(group.version_ids)} versions')
        try:
            print('\nSaving to cache...')
            with open(self.cache_file, 'wb') as f:
                pickle.dump({'data': self._serialize_version_data(), 'achievements': self.achievements, 'model_groups': self.model_groups}, f)
            print('Cache saved successfully')
        except Exception as e:
            print(f'Error saving cache: {e}')
        print('\nAll data loading operations complete')

    def _load_version_from_db(self, version: int) -> List[Node]:
        """Load all trajectories for a version from database"""
        with self.db_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('\n                    SELECT id, parent_id, achievements_json, value, ticks \n                    FROM programs WHERE version = %s\n                ', (version,))
                rows = cur.fetchall()
        nodes = {}
        roots = []
        for id, parent_id, achievements, value, ticks in rows:
            node = Node(id=id, parent_id=parent_id, metrics={'value': value or 0, 'ticks': ticks or 0}, static_achievements=achievements.get('static', {}), dynamic_achievements=achievements.get('dynamic', {}), children=[])
            nodes[id] = node
        for node in nodes.values():
            if node.parent_id is None:
                roots.append(node)
            elif node.parent_id in nodes:
                nodes[node.parent_id].children.append(node)
        return roots

    def _calculate_gdp(self, root: Node) -> float:
        """Calculate GDP for a trajectory using either method"""
        if not self.use_value_gdp:
            'Calculate final GDP for a trajectory'
            total = 0
            stack = [root]
            while stack:
                node = stack.pop()
                total += node.metrics['value']
                stack.extend(node.children)
            return total
        total_value = 0
        stack = [root]
        all_achievements = defaultdict(int)
        while stack:
            node = stack.pop()
            for item, quantity in node.static_achievements.items():
                all_achievements[item] += quantity
            for item, quantity in node.dynamic_achievements.items():
                all_achievements[item] += quantity
            stack.extend(node.children)
        print('\nValue-based GDP calculation:')
        for item, quantity in all_achievements.items():
            item_value = self.value_calculator.get_value(item)
            contribution = item_value * quantity
            total_value += contribution
            print(f'  {item}: {quantity} x {item_value:.2f} = {contribution:.2f}')
        print(f'Total value-based GDP: {total_value}')
        return total_value

    def _process_achievements(self, version: int):
        """Process achievements with correct tick accumulation"""
        print(f'\nProcessing achievements for version {version}')
        with open('recipes.jsonl', 'r') as f:
            recipes = {r['name']: r for r in map(json.loads, f)}
        seen = set()
        version_achievements = []
        for root in self.versions[version]['nodes']:
            current_path = []
            stack = [(root, 0, [])]
            while stack:
                node, depth, path = stack.pop()
                current_path = path + [node]
                path_ticks = sum((n.metrics['ticks'] for n in current_path))
                for achievements_dict, is_dynamic in [(node.static_achievements, False), (node.dynamic_achievements, True)]:
                    for item, quantity in achievements_dict.items():
                        if item not in seen:
                            print(f'\nProcessing achievement: {item}')
                            print(f'Original ticks: {path_ticks}, depth: {depth}')
                            version_achievements.append(Achievement(depth=depth, ticks=path_ticks, item_name=item, ingredients=self._count_ingredients(recipes.get(item, {})), is_dynamic=is_dynamic))
                            seen.add(item)
                for child in reversed(node.children):
                    stack.append((child, depth + 1, current_path))
        self.achievements[version] = version_achievements
        print(f'Total achievements processed for version {version}: {len(seen)}')

    def _count_ingredients(self, recipe: Dict) -> int:
        """Count total unique ingredients in recipe"""
        seen = set()
        if not recipe:
            return 1

        def traverse(item):
            seen.add(item['name'])
            for ingredient in item.get('ingredients', []):
                traverse(ingredient)
        traverse(recipe)
        return len(seen) - 1

    def add_complexity_brackets(self, fig, icon_ax, achievements, x_positions, complexities, fontsize=12):
        bracket_ax = fig.add_axes(icon_ax.get_position())
        bracket_ax.set_xlim(icon_ax.get_xlim())
        bracket_ax.set_ylim(0, 1)
        bracket_ax.axis('off')
        current_complexity = None
        start_x = None
        for i, (achievement, x) in enumerate(zip(achievements, x_positions)):
            complexity = complexities.get(achievement, 0)
            if complexity != current_complexity:
                if current_complexity is not None:
                    self.draw_bracket(bracket_ax, start_x, x_positions[i - 1], current_complexity, fontsize)
                current_complexity = complexity
                start_x = x
        if current_complexity is not None:
            self.draw_bracket(bracket_ax, start_x, x_positions[-1], current_complexity, fontsize)

    def draw_bracket(self, ax, start_x, end_x, complexity, fontsize=12):
        mid_x = (start_x + end_x) / 2
        bracket_height = 0.4
        text_height = -0.4
        ax.plot([start_x, start_x, end_x, end_x], [0, -bracket_height, -bracket_height, 0], color='black', linewidth=1)
        ax.text(mid_x, text_height, f'{complexity}', ha='center', va='top', fontsize=fontsize - 2, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        ax.set_ylim(min(ax.get_ylim()[0], text_height - 0.1), ax.get_ylim()[1])

    def export_combined_visualization(self, output_file: str, model_names: List[str], achievement_params: dict=None, production_params: dict=None, layout: Literal['side-by-side', 'stacked']='stacked'):
        """
        Create a combined visualization with achievement stack and production volumes.
        """
        if achievement_params is None:
            achievement_params = {'render_complexity': False, 'minimum_complexity': 2}
        if production_params is None:
            production_params = {'step_size': 50, 'step_proportion': 0.9, 'show_fractions': False, 'use_log_scale': False, 'min_total_volume': 1e-06, 'min_complexity': 2, 'cumulative': True, 'groupby_complexity': False, 'unified_y_axis': True, 'chart_type': 'line'}
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        if layout == 'side-by-side':
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(3, 3, width_ratios=[2, 1, 1], height_ratios=[1.5, 0.75, 0.75], left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.4)
            achievement_ax = fig.add_subplot(gs[:, 0])
            icon_ax = fig.add_axes(achievement_ax.get_position())
            icon_ax.set_position([achievement_ax.get_position().x0, achievement_ax.get_position().y0 - 0.05, achievement_ax.get_position().width, 0.05])
            production_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]
        else:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 0.75, 0.75], left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.4)
            achievement_ax = fig.add_subplot(gs[0, :])
            icon_ax = fig.add_axes(achievement_ax.get_position())
            icon_ax.set_position([achievement_ax.get_position().x0, achievement_ax.get_position().y0 - 0.05, achievement_ax.get_position().width, 0.05])
            production_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]
        self._plot_achievement_stack(achievement_ax, icon_ax, model_names, layout=layout, **achievement_params)
        self._plot_production_volumes(production_axes, model_names, layout=layout, **production_params)
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_achievement_stack(self, ax, icon_ax, model_names, render_complexity=False, minimum_complexity=2, fontsize=16, layout='stacked'):
        """Helper method to plot achievement stack on given axes"""
        from collections import defaultdict
        RAW_RESOURCES = {'coal', 'copper-ore', 'iron-ore', 'water', 'stone', 'wood'}
        model_achievement_counts = defaultdict(dict)
        all_achievements = set()
        achievement_complexities = {}
        for model_name in model_names:
            counts = defaultdict(int)
            group = self.model_groups[model_name]
            for version in group.version_ids:
                if version not in self.versions:
                    continue
                for root in self.versions[version]['nodes']:
                    run_achievements = defaultdict(int)
                    stack = [root]
                    while stack:
                        node = stack.pop()
                        for item, quantity in node.static_achievements.items():
                            run_achievements[item] += quantity
                        for item, quantity in node.dynamic_achievements.items():
                            run_achievements[item] += quantity
                        stack.extend(node.children)
                    for item, quantity in run_achievements.items():
                        counts[item] += quantity
                        all_achievements.add(item)
                        if item not in model_achievement_counts[model_name]:
                            model_achievement_counts[model_name][item] = 0
                        model_achievement_counts[model_name][item] += quantity
                for achievement in self.achievements[version]:
                    if achievement.item_name in RAW_RESOURCES:
                        achievement_complexities[achievement.item_name] = 0
                    else:
                        achievement_complexities[achievement.item_name] = achievement.ingredients
        if minimum_complexity > 0:
            filtered_achievements = {item for item in all_achievements if achievement_complexities.get(item, 0) >= minimum_complexity}
            all_achievements = filtered_achievements
            for version in model_achievement_counts:
                filtered_counts = {item: count for item, count in model_achievement_counts[version].items() if item in filtered_achievements}
                model_achievement_counts[version] = filtered_counts

        def sort_key(item):
            if item in RAW_RESOURCES and achievement_complexities.get(item, 0) < minimum_complexity:
                return (0, item)
            return (1, achievement_complexities.get(item, 0), item)
        sorted_achievements = sorted(all_achievements, key=sort_key)
        x_positions = np.arange(len(sorted_achievements))
        bar_width = 0.15
        spacing_factor = 1
        for idx, (version, counts) in enumerate(model_achievement_counts.items()):
            color = self.colors[idx % len(self.colors)]
            x_offset = (idx - (len(model_names) - 1) / 2) * bar_width * spacing_factor
            heights = [counts.get(ach, 0) for ach in sorted_achievements]
            nonzero_mask = np.array(heights) > 0
            if any(nonzero_mask):
                ax.bar(x_positions + x_offset, heights, bar_width, color=color, alpha=0.7, label=version)
        raw_resources_shown = len([r for r in RAW_RESOURCES if r in sorted_achievements])
        if raw_resources_shown > 0:
            ax.axvline(raw_resources_shown - 0.5, color='gray', linestyle='-', alpha=0.5)
        prev_complexity = None
        for i, ach in enumerate(sorted_achievements[raw_resources_shown:], raw_resources_shown):
            complexity = achievement_complexities.get(ach, 0)
            if prev_complexity is not None and complexity != prev_complexity:
                ax.axvline(i - 0.5, color='gray', linestyle='--', alpha=0.3)
            prev_complexity = complexity
        ax.set_yscale('log')
        ax.set_ylabel('Item Production', fontsize=fontsize - 2)
        ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
        ax.set_xticks([])
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0.9)
        num_versions = len(model_names)
        total_width = bar_width * num_versions * 1.1
        x_min = -total_width / 2
        x_max = len(sorted_achievements) - 1 + total_width / 2
        ax.set_xlim(x_min, x_max)
        icon_ax.set_xlim(ax.get_xlim())
        icon_ax.set_ylim(0, 1)
        icon_ax.axis('off')
        for x, achievement in zip(x_positions, sorted_achievements):
            try:
                icon_path = f'icons/{achievement}.png'
                if os.path.exists(icon_path):
                    icon = plt.imread(icon_path)
                    height, width = icon.shape[:2]
                    height += 1
                    width += 1
                    center = (width // 2, height // 2)
                    radius = min(width, height) // 2
                    Y, X = np.ogrid[:height, :width]
                    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
                    circular_mask = dist_from_center <= radius
                    bg = np.zeros((height, width, 4))
                    bg[circular_mask] = [1, 1, 1, 0.7]
                    bg_box = OffsetImage(bg, zoom=0.25)
                    bg_box.image.axes = icon_ax
                    bg_ab = AnnotationBbox(bg_box, (x, 0.95 if layout == 'stacked' else 0.5), frameon=False, box_alignment=(0.5, 0.5), zorder=2)
                    icon_ax.add_artist(bg_ab)
                    icon_box = OffsetImage(icon, zoom=0.25)
                    ab = AnnotationBbox(icon_box, (x, 0.95 if layout == 'stacked' else 0.5), frameon=False, box_alignment=(0.5, 0.5), zorder=3)
                    icon_ax.add_artist(ab)
            except Exception as e:
                print(f'Failed to add icon for {achievement}: {e}')
        self.add_complexity_brackets(ax.figure, icon_ax, sorted_achievements, x_positions, achievement_complexities, fontsize)

    def _plot_production_volumes(self, axes, model_names, step_size=25, step_proportion=0.9, show_fractions=False, use_log_scale=False, min_total_volume=1e-06, min_complexity=2, cumulative=True, groupby_complexity=False, unified_y_axis=True, chart_type='line', fontsize=16, layout='stacked'):
        """Helper method to plot production volumes on given axes using median values"""
        from collections import namedtuple, defaultdict
        import numpy as np
        StackItem = namedtuple('StackItem', ['node', 'step'])
        all_items = set()
        model_versions_found = defaultdict(list)
        for model_name in model_names:
            if model_name not in self.model_groups:
                print(f'Model {model_name} not found in model_groups')
                continue
            model_group = self.model_groups[model_name]
            for version_id in model_group.version_ids:
                if version_id not in self.versions:
                    print(f'Version {version_id} for model {model_name} not found in data')
                    continue
                model_versions_found[model_name].append(version_id)
                for root in self.versions[version_id]['nodes']:
                    stack = [StackItem(node=root, step=0)]
                    while stack:
                        current = stack.pop()
                        all_items.update(current.node.static_achievements.keys())
                        all_items.update(current.node.dynamic_achievements.keys())
                        for child in current.node.children:
                            stack.append(StackItem(node=child, step=current.step + 1))
        all_items = {item for item in all_items if item not in ('water', 'steam')}
        item_complexities = {}
        for model_name in model_names:
            if model_name not in model_versions_found:
                continue
            for version in model_versions_found[model_name]:
                if version in self.achievements:
                    for achievement in self.achievements[version]:
                        if achievement.item_name in all_items:
                            item_complexities[achievement.item_name] = achievement.ingredients
        sorted_items = sorted([item for item in all_items if item_complexities.get(item, 0) >= min_complexity], key=lambda x: item_complexities.get(x, 0))
        if not sorted_items:
            print(f'No items found with complexity >= {min_complexity}')
            return
        if groupby_complexity:
            complexity_groups = {}
            for item in sorted_items:
                complexity = item_complexities[item]
                if complexity not in complexity_groups:
                    complexity_groups[complexity] = []
                complexity_groups[complexity].append(item)
            unique_complexities = sorted(complexity_groups.keys())
            color_map = plt.cm.viridis(np.linspace(0, 1, len(unique_complexities)))
            complexity_colors = dict(zip(unique_complexities, color_map))
        else:
            color_map = plt.cm.viridis(np.linspace(0, 1, len(sorted_items)))
        global_ymin = float('inf')
        global_ymax = float('-inf')
        for ax_idx, model_name in enumerate(model_names):
            print(f'\nPlotting for axis {ax_idx}, model {model_name}')
            if ax_idx >= len(axes):
                print(f'ERROR: Not enough axes for model {model_name}')
                continue
            model_group = self.model_groups[model_name]
            production_by_step = defaultdict(lambda: defaultdict(list))
            all_steps = set()
            for version in model_group.version_ids:
                if version not in self.versions:
                    continue
                for root in self.versions[version]['nodes']:
                    stack = [StackItem(node=root, step=0)]
                    while stack:
                        current = stack.pop()
                        step_bucket = current.step // step_size * step_size
                        all_steps.add(step_bucket)
                        for child in current.node.children:
                            stack.append(StackItem(node=child, step=current.step + 1))
            all_steps = sorted(all_steps)
            if cumulative:
                for version in model_group.version_ids:
                    if version not in self.versions:
                        continue
                    for root in self.versions[version]['nodes']:
                        trajectory_cumulative = defaultdict(int)
                        stack = [StackItem(node=root, step=0)]
                        while stack:
                            current = stack.pop()
                            step_bucket = current.step // step_size * step_size
                            for achievements_dict in [current.node.static_achievements, current.node.dynamic_achievements]:
                                for item, quantity in achievements_dict.items():
                                    if item in sorted_items:
                                        trajectory_cumulative[item] += quantity
                            for item, total in trajectory_cumulative.items():
                                production_by_step[step_bucket][item].append(total)
                                for future_step in range(step_bucket + step_size, max(all_steps) + step_size, step_size):
                                    production_by_step[future_step][item].append(total)
                            for child in current.node.children:
                                stack.append(StackItem(node=child, step=current.step + 1))
            else:
                for version in model_group.version_ids:
                    if version not in self.versions:
                        continue
                    for root in self.versions[version]['nodes']:
                        stack = [StackItem(node=root, step=0)]
                        while stack:
                            current = stack.pop()
                            step_bucket = current.step // step_size * step_size
                            step_production = defaultdict(int)
                            for achievements_dict in [current.node.static_achievements, current.node.dynamic_achievements]:
                                for item, quantity in achievements_dict.items():
                                    if item in sorted_items:
                                        step_production[item] += quantity
                            for item, quantity in step_production.items():
                                production_by_step[step_bucket][item].append(quantity)
                            for child in current.node.children:
                                stack.append(StackItem(node=child, step=current.step + 1))
            total_values = {}
            for step in all_steps:
                total_values[step] = {}
                for item in sorted_items:
                    values = production_by_step[step][item]
                    if values:
                        total_values[step][item] = max(values)
                    else:
                        total_values[step][item] = 0
            processed_values = {}
            if groupby_complexity:
                for complexity in complexity_groups:
                    values = np.zeros(len(all_steps))
                    for item in complexity_groups[complexity]:
                        item_values = [total_values[step].get(item, 0) for step in all_steps]
                        values += item_values
                    processed_values[complexity] = values
            else:
                for item in sorted_items:
                    values = [total_values[step].get(item, 0) for step in all_steps]
                    if use_log_scale:
                        values = [np.log10(v + 1) for v in values]
                    processed_values[item] = values
            ax = axes[ax_idx]
            ax.set_aspect('auto')

            def format_large_number(x, pos):
                """Format large numbers with commas for readability"""
                if abs(x) >= 1000000.0:
                    return f'{int(x / 1000000.0):,}M'
                elif abs(x) >= 1000.0:
                    return f'{int(x / 1000.0):,}K'
                else:
                    return f'{int(x):,}'
            if groupby_complexity:
                values_for_stack = [processed_values[complexity] for complexity in unique_complexities]
                labels = [f'Complexity {complexity}' for complexity in unique_complexities]
                colors = [complexity_colors[complexity] for complexity in unique_complexities]
            else:
                values_for_stack = [processed_values[item] for item in sorted_items]
                labels = [f'{item}' for item in sorted_items]
                colors = color_map
            if chart_type.lower() == 'line':
                stack_plot = ax.stackplot(all_steps, values_for_stack, labels=labels if ax_idx == len(model_names) - 1 else [], colors=colors, alpha=0.7)
            else:
                bottom = np.zeros(len(all_steps))
                for values, label, color in zip(values_for_stack, labels, colors):
                    ax.bar(all_steps, values, bottom=bottom, label=label if ax_idx == len(model_names) - 1 else None, color=color, alpha=0.7, width=step_size * step_proportion)
                    bottom += values
            for ax_idx, ax in enumerate(axes):
                if use_log_scale:
                    ax.set_yscale('log')
                    ax.yaxis.set_major_formatter(plt.LogFormatter(base=10))
                    ax.yaxis.set_major_locator(plt.LogLocator(base=10.0))
                    ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10)))
                ax.set_title('')
                ymin, ymax = ax.get_ylim()
                global_ymin = min(global_ymin, ymin)
                global_ymax = max(global_ymax, ymax)
                ax.grid(True, axis='y', linestyle='--', alpha=0.3)
                ax.set_axisbelow(True)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(format_large_number))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_number))
                ax.set_xlim(min(all_steps), max(all_steps))
                if layout == 'stacked':
                    if ax_idx == 0:
                        ax.set_ylabel('Item Production', fontsize=fontsize - 2)
                elif ax_idx in [0, 2, 4]:
                    ax.set_ylabel('Item Production', fontsize=fontsize - 2)
                ax.set_xlabel('Steps', fontsize=fontsize - 2)
                if ax_idx < len(model_names):
                    title_text = self.model_groups[model_names[ax_idx]].label
                    color = self.model_groups[model_names[ax_idx]].color
                    ax_pos = ax.get_position()
                    title_y = ax_pos.y1 - 0.03
                    ax.figure.text(ax_pos.x0 + ax_pos.width * 0.02, title_y, '—', color=color, fontsize=fontsize + 4, fontweight='bold', horizontalalignment='left', verticalalignment='bottom', zorder=10)
                    ax.figure.text(ax_pos.x0 + ax_pos.width * 0.08, title_y + 0.003, title_text, fontsize=fontsize - 4, horizontalalignment='left', verticalalignment='bottom', zorder=10)
            ax.grid(True, which='major', linestyle='-', alpha=0.2)
            ax.grid(True, which='minor', linestyle='--', alpha=0.1)
            ax.set_xlim(min(all_steps), max(all_steps))
            if show_fractions:
                ax.set_ylim(0, 1)
            title_text = model_group.label
            color = model_group.color
            title_y = ax.get_position().y1 - 0.04
            if use_log_scale:
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(plt.LogFormatter(base=10))
                ax.yaxis.set_major_locator(plt.LogLocator(base=10.0))
                ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10)))
            ymin, ymax = ax.get_ylim()
            global_ymin = min(global_ymin, ymin)
            global_ymax = max(global_ymax, ymax)
            ax.margins(y=0.01)
        if unified_y_axis and (not show_fractions):
            padding = (global_ymax - global_ymin) * 0.1
            for ax in axes:
                ax.set_ylim(global_ymin, global_ymax + padding)
        if not groupby_complexity:
            if layout == 'stacked':
                create_custom_legend_with_icons(axes[-1], sorted_items, item_complexities, colors, fontsize, layout == 'stacked')
            else:
                create_custom_legend_with_icons(axes[-3], sorted_items, item_complexities, colors, fontsize, layout == 'stacked')

class ProgressionVisualizerWithTicks:
    """Creates publication-quality visualizations of agent progression"""
    VERTICAL_SPACING_PIXELS = 12
    HORIZONTAL_OFFSET_PIXELS = 0
    BASE_STACK_TOLERANCE = 4
    STACK_VERTICAL_SPACING = 12
    MAX_STACK_SIZE = 5

    def __init__(self, db_client, icons_path: str, x_axis: Literal['steps', 'ticks']='steps', cache_file: str='viz_cache.pkl', x_base: float=10, y_base: float=10, use_value_gdp=False, recipes_file='recipes.jsonl', use_log_scale: bool=True):
        self.db_client = db_client
        self.icons_path = icons_path
        self.x_axis = x_axis
        self.cache_file = cache_file
        self.x_base = x_base
        self.y_base = y_base
        self.versions = {}
        self.achievements = defaultdict(list)
        self.colors = ['#8fd7d7', '#FFCD8E', '#00b0be', '#ff8ca1', '#f45f74', '#bdd373', '#98c127', '#ffb255']
        self.use_value_gdp = use_value_gdp
        self.value_calculator = ValueCalculator(recipes_file) if use_value_gdp else None
        self.use_log_scale = use_log_scale

    def _serialize_version_data(self):
        """Convert version data to cacheable format"""
        serialized = {}
        for version, data in self.versions.items():
            nodes_dict = {}
            for root in data['nodes']:
                stack = [root]
                while stack:
                    node = stack.pop()
                    nodes_dict[node.id] = node.to_dict()
                    stack.extend(node.children)
            serialized[version] = {'nodes_dict': nodes_dict, 'root_ids': [root.id for root in data['nodes']], 'label': data['label']}
        return serialized

    def _deserialize_version_data(self, serialized):
        """Restore version data from cached format"""
        for version, data in serialized.items():
            nodes_dict = {}
            for node_id, node_data in data['nodes_dict'].items():
                nodes_dict[node_id] = Node(id=node_data['id'], parent_id=node_data['parent_id'], metrics=node_data['metrics'], static_achievements=node_data['static_achievements'], dynamic_achievements=node_data['dynamic_achievements'], children=[])
            for node_id, node_data in data['nodes_dict'].items():
                nodes_dict[node_id].children = [nodes_dict[child_id] for child_id in node_data['children_ids']]
            self.versions[version] = {'nodes': [nodes_dict[root_id] for root_id in data['root_ids']], 'label': data['label']}

    def load_data(self, version_groups: Dict[str, List[int]], labels: Dict[str, str]):
        """
        Load and process data for multiple version groups, using cache if available

        Args:
            version_groups: Dict mapping model names to lists of version numbers
            labels: Dict mapping model names to display labels
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data.get('version_groups') == version_groups:
                        self._deserialize_version_data(cached_data['data'])
                        self.achievements = cached_data['achievements']
                        return
            except Exception as e:
                print(f'Error loading cache: {e}')
                os.remove(self.cache_file)
        for model_name, versions in version_groups.items():
            print(f'\nLoading model {model_name} (versions: {versions})')
            all_nodes = []
            for version in versions:
                nodes = self._load_version_from_db(version)
                if nodes:
                    all_nodes.extend(nodes)
            if all_nodes:
                gdps = [self._calculate_gdp(root) for root in all_nodes]
                print(f'Mean GDP across all versions: {np.mean(gdps):.1f}')
                print(f'STD GDP across all versions: {np.std(gdps):.1f}')
                self.versions[model_name] = {'nodes': all_nodes, 'label': f'{labels[model_name]}'}
                self._process_merged_achievements(model_name, versions)
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({'version_groups': version_groups, 'data': self._serialize_version_data(), 'achievements': self.achievements}, f)
        except Exception as e:
            print(f'Error saving cache: {e}')

    def _process_merged_achievements(self, model_name: str, versions: List[int]):
        """Process achievements across multiple versions of the same model"""
        print(f'\nProcessing achievements for model {model_name}')
        with open('recipes.jsonl', 'r') as f:
            recipes = {r['name']: r for r in map(json.loads, f)}
        seen = set()
        model_achievements = []
        for version in versions:
            for root in self._load_version_from_db(version):
                current_path = []
                stack = [(root, 0, [])]
                while stack:
                    node, depth, path = stack.pop()
                    current_path = path + [node]
                    path_ticks = sum((n.metrics['ticks'] for n in current_path))
                    for achievements_dict, is_dynamic in [(node.static_achievements, False), (node.dynamic_achievements, True)]:
                        for item, quantity in achievements_dict.items():
                            achievement_key = (version, item)
                            if achievement_key not in seen:
                                print(f'\nProcessing achievement: {item} (version {version})')
                                print(f'Original ticks: {path_ticks}, depth: {depth}')
                                model_achievements.append(Achievement(depth=depth, ticks=path_ticks, item_name=item, ingredients=self._count_ingredients(recipes.get(item, {})), is_dynamic=is_dynamic))
                                seen.add(achievement_key)
                    for child in reversed(node.children):
                        stack.append((child, depth + 1, current_path))
        earliest_achievements = {}
        for achievement in model_achievements:
            key = achievement.item_name
            if key not in earliest_achievements or (achievement.ticks < earliest_achievements[key].ticks or (achievement.ticks == earliest_achievements[key].ticks and achievement.depth < earliest_achievements[key].depth)):
                earliest_achievements[key] = achievement
        self.achievements[model_name] = list(earliest_achievements.values())
        print(f'Total unique achievements processed for model {model_name}: {len(earliest_achievements)}')

    def organize_achievement_positions(self, achievements_by_depth, depth_stats, ax, series_index, used_positions):
        """Organize achievement positions with improved stacking logic and priority items"""
        final_positions = {}
        stacks = defaultdict(list)
        PRIORITY_ACHIEVEMENTS = {'lab', 'steam'}
        for x_coord, achievements_list in achievements_by_depth.items():
            if x_coord in depth_stats and depth_stats[x_coord]['mean'] > 0:
                base_position = depth_stats[x_coord]['mean']
                if self.x_axis == 'ticks':
                    log_x = np.log10(x_coord)
                    tolerance = self.BASE_STACK_TOLERANCE * log_x
                    bucket_key = round(x_coord / (x_coord * tolerance)) * (x_coord * tolerance)
                else:
                    if x_coord < 50:
                        tolerance = self.BASE_STACK_TOLERANCE
                    elif x_coord < 100:
                        tolerance = self.BASE_STACK_TOLERANCE * 2
                    elif x_coord < 400:
                        tolerance = self.BASE_STACK_TOLERANCE * 4
                    else:
                        tolerance = self.BASE_STACK_TOLERANCE * 8
                    bucket_key = round(x_coord / tolerance) * tolerance

                def sort_key(achievement):
                    is_priority = achievement.item_name in PRIORITY_ACHIEVEMENTS
                    return (int(is_priority), -achievement.ingredients, achievement.item_name)
                achievements_list.sort(key=sort_key)
                stacks[bucket_key].extend([(x_coord, achievement, base_position) for achievement in achievements_list])
        for bucket_key, stack in stacks.items():
            if not stack:
                continue
            base_x_coord, _, base_y = stack[0]
            base_display_coords = ax.transData.transform([[base_x_coord, base_y]])[0]
            base_display_x, base_display_y = base_display_coords
            x_key = round(bucket_key, -3 if self.x_axis == 'ticks' else 1)
            if x_key in used_positions:
                base_display_x += self.HORIZONTAL_OFFSET_PIXELS * (series_index + 1)
            else:
                used_positions[x_key] = True
            stack_size = min(len(stack), self.MAX_STACK_SIZE)
            for i in range(stack_size):
                x_coord, achievement, base_y = stack[i]
                stack_center_offset = (stack_size - 1) * self.STACK_VERTICAL_SPACING / 2
                vertical_offset = i * self.STACK_VERTICAL_SPACING - stack_center_offset
                display_y = base_display_y + vertical_offset
                data_coords = ax.transData.inverted().transform([[base_display_x, display_y]])[0]
                data_x, data_y = data_coords
                achievement_key = (achievement.item_name, x_coord)
                final_positions[achievement_key] = (data_x, data_y)
        return final_positions

    def _get_stack_vertical_spacing(self, ax):
        """Calculate appropriate vertical spacing based on plot dimensions"""
        bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        fig_height_pixels = bbox.height * ax.figure.dpi
        base_spacing = fig_height_pixels * 0.02
        return max(10, min(base_spacing, 20))

    def _prepare_achievement_stack(self, achievements, max_stack_size=5):
        """Prepare achievement stack with priority based on complexity"""
        sorted_achievements = sorted(achievements, key=lambda a: (a.ingredients, a.item_name))
        if len(sorted_achievements) > max_stack_size:
            return sorted_achievements[:max_stack_size]
        return sorted_achievements

    def organize_achievement_positions2(self, achievements_by_depth, depth_stats, ax, series_index, used_positions):
        """Organize achievement positions with improved tick mapping"""
        positions = {}
        final_positions = {}
        for x_coord, achievements_list in achievements_by_depth.items():
            if x_coord in depth_stats and depth_stats[x_coord]['mean'] > 0:
                base_position = depth_stats[x_coord]['mean']
                positions[x_coord] = {'achievements': achievements_list, 'base_y': base_position}
        for x_coord, group_data in positions.items():
            achievements = group_data['achievements']
            base_y = group_data['base_y']
            achievements.sort(key=lambda a: (a.ingredients, a.item_name))
            display_coords = ax.transData.transform([[x_coord, base_y]])[0]
            base_display_x, base_display_y = display_coords
            x_key = round(x_coord, -3 if self.x_axis == 'ticks' else 1)
            if x_key in used_positions:
                base_display_x += self.HORIZONTAL_OFFSET_PIXELS * (series_index + 1)
            else:
                used_positions[x_key] = True
            for i, achievement in enumerate(achievements):
                vertical_offset = i * self.VERTICAL_SPACING_PIXELS
                display_y = base_display_y + vertical_offset
                data_coords = ax.transData.inverted().transform([[base_display_x, display_y]])[0]
                data_x, data_y = data_coords
                achievement_key = (achievement.item_name, x_coord)
                final_positions[achievement_key] = (data_x, data_y)
        return final_positions

    def _load_version_from_db(self, version: int) -> List[Node]:
        """Load all trajectories for a version from database"""
        with self.db_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('\n                    SELECT id, parent_id, achievements_json, value, ticks \n                    FROM programs WHERE version = %s\n                ', (version,))
                rows = cur.fetchall()
        nodes = {}
        roots = []
        for id, parent_id, achievements, value, ticks in rows:
            node = Node(id=id, parent_id=parent_id, metrics={'value': value or 0, 'ticks': ticks or 0}, static_achievements=achievements.get('static', {}), dynamic_achievements=achievements.get('dynamic', {}), children=[])
            nodes[id] = node
        for node in nodes.values():
            if node.parent_id is None:
                roots.append(node)
            elif node.parent_id in nodes:
                nodes[node.parent_id].children.append(node)
        return roots

    def _calculate_gdp(self, root: Node) -> float:
        """Calculate GDP for a trajectory using either method"""
        if not self.use_value_gdp:
            'Calculate final GDP for a trajectory'
            total = 0
            stack = [root]
            while stack:
                node = stack.pop()
                total += node.metrics['value']
                stack.extend(node.children)
            return total
        total_value = 0
        stack = [root]
        all_achievements = defaultdict(int)
        while stack:
            node = stack.pop()
            for item, quantity in node.static_achievements.items():
                all_achievements[item] += quantity
            for item, quantity in node.dynamic_achievements.items():
                all_achievements[item] += quantity
            stack.extend(node.children)
        print('\nValue-based GDP calculation:')
        for item, quantity in all_achievements.items():
            item_value = self.value_calculator.get_value(item)
            contribution = item_value * quantity
            total_value += contribution
            print(f'  {item}: {quantity} x {item_value:.2f} = {contribution:.2f}')
        print(f'Total value-based GDP: {total_value}')
        return total_value

    def _process_achievements(self, version: int):
        """Process achievements with correct tick accumulation"""
        print(f'\nProcessing achievements for version {version}')
        with open('recipes.jsonl', 'r') as f:
            recipes = {r['name']: r for r in map(json.loads, f)}
        seen = set()
        version_achievements = []
        for root in self.versions[version]['nodes']:
            current_path = []
            stack = [(root, 0, [])]
            while stack:
                node, depth, path = stack.pop()
                current_path = path + [node]
                path_ticks = sum((n.metrics['ticks'] for n in current_path))
                for achievements_dict, is_dynamic in [(node.static_achievements, False), (node.dynamic_achievements, True)]:
                    for item, quantity in achievements_dict.items():
                        if item not in seen:
                            print(f'\nProcessing achievement: {item}')
                            print(f'Original ticks: {path_ticks}, depth: {depth}')
                            version_achievements.append(Achievement(depth=depth, ticks=path_ticks, item_name=item, ingredients=self._count_ingredients(recipes.get(item, {})), is_dynamic=is_dynamic))
                            seen.add(item)
                for child in reversed(node.children):
                    stack.append((child, depth + 1, current_path))
        self.achievements[version] = version_achievements
        print(f'Total achievements processed for version {version}: {len(seen)}')

    def _count_ingredients(self, recipe: Dict) -> int:
        """Count total unique ingredients in recipe"""
        seen = set()
        if not recipe:
            return 1

        def traverse(item):
            seen.add(item['name'])
            for ingredient in item.get('ingredients', []):
                traverse(ingredient)
        traverse(recipe)
        return len(seen) - 1

    def _prepare_icon(self, item_name: str):
        """Prepare achievement icon for visualization with size limits"""
        src = os.path.join(self.icons_path, f'{item_name}.png')
        dst = f'icons/{item_name}.png'
        if os.path.exists(src) and (not os.path.exists(dst)):
            os.makedirs('icons', exist_ok=True)
            with Image.open(src) as img:
                if hasattr(img, 'n_frames'):
                    img.seek(0)
                size = min(img.width, img.height)
                tile = img.crop((0, 0, size, size))
                if size > 256:
                    tile = tile.resize((256, 256), Image.Resampling.LANCZOS)
                if tile.mode != 'RGBA':
                    tile = tile.convert('RGBA')
                tile.save(dst, format='PNG')

    def weighted_percentile(self, values, weights, q):
        """Calculate weighted percentile for confidence intervals"""
        order = np.argsort(values)
        values = np.array(values)[order]
        weights = np.array(weights)[order]
        cumsum = np.cumsum(weights)
        cumsum = cumsum / cumsum[-1]
        return np.interp(q / 100, cumsum, values)

    def create_circle_background(self, color, alpha=0.7):
        """Create a circular background image with transparency"""
        size = 50
        img = np.zeros((size, size, 4))
        y, x = np.ogrid[-size / 2:size / 2, -size / 2:size / 2]
        mask = x * x + y * y <= size / 2 * (size / 2)
        img[..., :3][mask] = 1
        img[..., 3][mask] = alpha
        edge = np.zeros_like(mask)
        edge_width = 3
        for i in range(edge_width):
            edge_mask = (x * x + y * y <= (size / 2 - i) * (size / 2 - i)) & (x * x + y * y >= (size / 2 - i - 1) * (size / 2 - i - 1))
            edge |= edge_mask
        rgb_color = np.array([int(color[1:][i:i + 2], 16) / 255 for i in (0, 2, 4)])
        img[..., :3][edge] = rgb_color
        img[..., 3][edge] = 0.7
        return img

    def _get_step_at_ticks(self, version, target_ticks):
        """Find the step number at which a version reaches the target ticks"""
        if target_ticks <= 0 or np.isinf(target_ticks) or np.isnan(target_ticks):
            return None
        nodes = self.versions[version]['nodes']
        min_step = float('inf')
        for root in nodes:
            current_ticks = 0
            stack = [(root, 0, 0)]
            while stack:
                node, step, prev_ticks = stack.pop()
                current_ticks = prev_ticks + node.metrics['ticks']
                if current_ticks >= target_ticks:
                    min_step = min(min_step, step)
                    break
                for child in node.children:
                    stack.append((child, step + 1, current_ticks))
        return min_step if min_step != float('inf') else None

    def add_connecting_lines(self, ax1, ax2, step=990):
        y_min, y_max = ax2.get_ylim()
        fig = ax1.get_figure()
        ax1_bbox = ax1.get_position()
        ax2_bbox = ax2.get_position()
        x1 = ax1_bbox.x1
        x2 = ax2_bbox.x0
        y1_min = ax1.transData.transform([[0, y_min]])[0, 1]
        y1_min = fig.transFigure.inverted().transform([[0, y1_min]])[0, 1]
        y1_max = ax1.transData.transform([[0, y_max]])[0, 1]
        y1_max = fig.transFigure.inverted().transform([[0, y1_max]])[0, 1]
        y2_min = ax2.transData.transform([[0, y_min]])[0, 1]
        y2_min = fig.transFigure.inverted().transform([[0, y2_min]])[0, 1]
        y2_max = ax2.transData.transform([[0, y_max]])[0, 1]
        y2_max = fig.transFigure.inverted().transform([[0, y2_max]])[0, 1]
        polygon = plt.Polygon([[x1, y1_min], [x2, y2_min], [x2, y2_max], [x1, y1_max]], transform=fig.transFigure, facecolor='gray', alpha=0.15, edgecolor='#404040', linestyle='--', linewidth=1)
        fig.add_artist(polygon)

    def add_adjacent_connecting_lines(self, ax1, ax2):
        """Add connecting lines between two adjacent axes"""
        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()
        fig = ax1.get_figure()
        ax1_bbox = ax1.get_position()
        ax2_bbox = ax2.get_position()
        x1 = ax1_bbox.x1
        x2 = ax2_bbox.x0
        y1_min = ax1.transData.transform([[0, y1_min]])[0, 1]
        y1_min = fig.transFigure.inverted().transform([[0, y1_min]])[0, 1]
        y1_max = ax1.transData.transform([[0, y1_max]])[0, 1]
        y1_max = fig.transFigure.inverted().transform([[0, y1_max]])[0, 1]
        y2_min = ax2.transData.transform([[0, y2_min]])[0, 1]
        y2_min = fig.transFigure.inverted().transform([[0, y2_min]])[0, 1]
        y2_max = ax2.transData.transform([[0, y2_max]])[0, 1]
        y2_max = fig.transFigure.inverted().transform([[0, y2_max]])[0, 1]
        polygon = plt.Polygon([[x1, y1_min], [x2, y2_min], [x2, y2_max], [x1, y1_max]], transform=fig.transFigure, facecolor='gray', alpha=0.15, edgecolor='#404040', linestyle='--', linewidth=1)
        fig.add_artist(polygon)

    def export_split_visualization(self, output_file: str, max_depth: int=4990):
        """Export a visualization with main progression chart, final GDP scatter plot, and final time plot"""
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        fig = plt.figure(figsize=(14, 5))
        gs = plt.GridSpec(3, 3, height_ratios=[0.85, 0.075, 0.075], width_ratios=[15, 1, 1], wspace=0.12, hspace=0.2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax_reward_label = fig.add_subplot(gs[1, 1])
        ax_time_label = fig.add_subplot(gs[1, 2])
        ax_shared = fig.add_subplot(gs[2, 1:])
        ax_reward_label.axis('off')
        ax_time_label.axis('off')
        ax_shared.axis('off')
        ax1.set_xscale('log', base=self.x_base)
        ax1.set_yscale('log', base=self.y_base)
        if self.x_axis == 'ticks':
            ax1.set_xlim(1000.0, 100000000.0)
            ax1.set_xlabel('Ticks', fontsize=12)
        else:
            ax1.set_xlim(100, 5000.0)
            ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylim(1000.0, 400000.0)
        ax1.set_ylabel('Reward\nCumulative Production Score', fontsize=12, labelpad=10)
        ylabel = ax1.yaxis.label
        lines = ylabel.get_text().split('\n')
        ax1.set_ylabel(lines[0], fontsize=12, labelpad=20)
        subtitle = ax1.text(-0.05, 0.5, lines[1], transform=ax1.transAxes, rotation=90, ha='center', va='center', fontsize=9)
        final_values = []
        final_cis = []
        final_minutes = []
        final_minutes_cis = []
        colors = []
        ticks_per_unit = 60 * 60 * 60
        used_positions = {}
        for idx, (version, data) in enumerate(self.versions.items()):
            color = self.colors[idx % len(self.colors)]
            colors.append(color)
            stats = self._calculate_statistics(data['nodes'], max_depth)
            x_coords = sorted(stats.keys())
            means = [stats[x]['mean'] for x in x_coords]
            sem_lower = [stats[x]['sem_lower'] for x in x_coords]
            sem_upper = [stats[x]['sem_upper'] for x in x_coords]
            ax1.plot(x_coords, means, color=color, label=data['label'], linewidth=1.5)
            ax1.fill_between(x_coords, sem_lower, sem_upper, color=color, alpha=0.2)
            target_step = 4990
            if means:
                if self.x_axis == 'steps':
                    target_idx = next((idx for idx, x in enumerate(x_coords) if x >= target_step), -1)
                else:
                    target_idx = next((idx for idx, x in enumerate(x_coords) if x >= target_step), -1)
                if target_idx != -1:
                    final_values.append(means[target_idx])
                    final_cis.append((sem_lower[target_idx], sem_upper[target_idx]))
                else:
                    final_values.append(means[-1])
                    final_cis.append((sem_lower[-1], sem_upper[-1]))
                final_ticks_stats = self._calculate_final_ticks_statistics(data['nodes'], target_step)
                final_minutes.append(final_ticks_stats['median'] / ticks_per_unit)
                final_minutes_cis.append((final_ticks_stats['ci_lower'] / ticks_per_unit, final_ticks_stats['ci_upper'] / ticks_per_unit))
            achievements_by_depth = defaultdict(list)
            for achievement in self.achievements[version]:
                if self.x_axis == 'ticks':
                    x_coord = achievement.ticks
                    bucket = round(np.log(x_coord) / np.log(self.x_base) * 10) / 10
                else:
                    x_coord = achievement.depth
                    bucket = x_coord
                if x_coord > 0:
                    achievements_by_depth[bucket].append(achievement)
            positions = self.organize_achievement_positions(achievements_by_depth, stats, ax1, idx, used_positions)
            for (item_name, orig_x), (x, y) in positions.items():
                if x > 0 and y > 0:
                    x_min, x_max = ax1.get_xlim()
                    y_min, y_max = ax1.get_ylim()
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        self._add_achievement_icon(ax1, item_name, x, y, color)
        ax1.grid(True, which='major', linestyle='-', color='gray', alpha=0.2)
        ax1.grid(True, which='minor', linestyle='--', color='gray', alpha=0.1)
        ax1.tick_params(axis='both', which='major', labelsize=9)
        ax1.tick_params(axis='both', which='minor', labelsize=7)
        ax1.set_axisbelow(True)
        ax1.legend(loc='lower right', fontsize=10)
        x_margin = 0.5
        ax2.set_xlim(-x_margin, len(final_values) - 1 + x_margin)
        for i, (value, (ci_lower, ci_upper), color) in enumerate(zip(final_values, final_cis, colors)):
            ax2.vlines(i, ci_lower, ci_upper, color=color, alpha=0.5)
            ax2.scatter(i, value, color=color, s=50, zorder=5)
        ax2.set_xticks([])
        ax2.set_yscale('log', base=self.y_base)
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_minor_locator(LogLocator(base=10, subs='all'))
        ax2.grid(True, which='major', axis='y', linestyle='-', color='gray', alpha=0.2)
        ax2.grid(True, which='minor', axis='y', linestyle='--', color='gray', alpha=0.1)
        ax2.tick_params(axis='y', which='major', labelsize=8)
        ax3.set_xlim(-x_margin, len(final_minutes) - 1 + x_margin)
        for i, (value, (ci_lower, ci_upper), color) in enumerate(zip(final_minutes, final_minutes_cis, colors)):
            ax3.vlines(i, ci_lower, ci_upper, color=color, alpha=0.5)
            ax3.scatter(i, value, color=color, s=50, zorder=5)
        ax3.set_xticks([])
        ax3.set_yscale('log', base=10)
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.tick_right()
        ax3.yaxis.set_minor_locator(LogLocator(base=10, subs='all'))
        ax3.grid(True, which='major', axis='y', linestyle='-', color='gray', alpha=0.2)
        ax3.grid(True, which='minor', axis='y', linestyle='--', color='gray', alpha=0.1)
        ax3.tick_params(axis='y', which='major', labelsize=8)
        y2_min = min((ci[0] for ci in final_cis))
        y2_max = max((ci[1] for ci in final_cis))
        ax2.set_ylim(min(0, y2_min * 0.9), y2_max * 1.1)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x / 1000):,}k'))
        ax2.yaxis.set_minor_locator(plt.NullLocator())
        y3_min = min((ci[0] for ci in final_minutes_cis))
        y3_max = max((ci[1] for ci in final_minutes_cis))
        ax3.set_ylim(min(0, y3_min * 0.9), y3_max * 1.1)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax3.yaxis.set_minor_locator(plt.NullLocator())
        ax_reward_label.text(0.5, 1.4, 'Reward', ha='center', va='center', fontsize=10)
        ax_time_label.text(0.5, 1.4, 'Elapsed (hrs)', ha='center', va='center', fontsize=10)
        ax_shared.text(0.5, 2.4, 'Final', ha='center', va='center', fontsize=12)
        self.add_connecting_lines(ax1, ax2)
        self.add_adjacent_connecting_lines(ax2, ax3)
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

    def _calculate_final_ticks_statistics(self, roots: List[Node], target_step: int) -> Dict:
        """Calculate statistics for final ticks at target step"""
        final_ticks = []
        for root in roots:
            stack = [(root, 0, 0)]
            path_ticks = []
            while stack:
                node, step, prev_ticks = stack.pop()
                if step == target_step:
                    path_ticks.append(prev_ticks + node.metrics['ticks'])
                    continue
                if step < target_step:
                    for child in node.children:
                        stack.append((child, step + 1, prev_ticks + node.metrics['ticks']))
            if path_ticks:
                final_ticks.append(np.median(path_ticks))
        if final_ticks:
            return {'median': np.median(final_ticks), 'ci_lower': np.percentile(final_ticks, 2.5), 'ci_upper': np.percentile(final_ticks, 97.5)}
        return {'median': 0, 'ci_lower': 0, 'ci_upper': 0}

    def _calculate_statistics(self, roots: List[Node], max_depth: int) -> Dict:
        """Calculate statistics with standard error of mean instead of CI"""
        values_by_x = defaultdict(list)
        for root in roots:
            values_by_x[0].append(0)
            stack = [(root, 0, 0, 0, defaultdict(int))]
            while stack:
                node, depth, prev_ticks, prev_value, prev_achievements = stack.pop()
                if depth > max_depth:
                    continue
                current_achievements = prev_achievements.copy()
                for item, quantity in node.static_achievements.items():
                    current_achievements[item] += quantity
                for item, quantity in node.dynamic_achievements.items():
                    current_achievements[item] += quantity
                if self.use_value_gdp:
                    current_value = 0
                    for item, quantity in current_achievements.items():
                        item_value = self.value_calculator.get_value(item)
                        current_value += item_value * quantity
                else:
                    current_value = prev_value + node.metrics['value']
                ticks = prev_ticks + node.metrics['ticks']
                x_coord = ticks if self.x_axis == 'ticks' else depth
                values_by_x[x_coord].append(current_value)
                for child in node.children:
                    stack.append((child, depth + 1, ticks, current_value, current_achievements))
        stats = {0: {'mean': 0, 'sem_lower': 0, 'sem_upper': 0, 'std': 0}}
        x_coords = sorted((x for x in values_by_x.keys() if x > 0))
        if self.x_axis == 'ticks':
            eval_points = np.logspace(np.log(min(x_coords)) / np.log(self.x_base), np.log(max(x_coords)) / np.log(self.x_base), 500, base=self.x_base)
            window = 0.1
            for x in eval_points:
                nearby_values = []
                for orig_x, values in values_by_x.items():
                    if orig_x > 0:
                        log_diff = abs(np.log(x) / np.log(self.x_base) - np.log(orig_x) / np.log(self.x_base))
                        if log_diff < window:
                            weight = np.exp(-(log_diff / window) ** 2)
                            nearby_values.extend(((v, weight) for v in values))
                if nearby_values:
                    values, weights = zip(*nearby_values)
                    mean = np.average(values, weights=weights)
                    weighted_var = np.average((np.array(values) - mean) ** 2, weights=weights)
                    n_effective = np.sum(weights) ** 2 / np.sum(np.array(weights) ** 2)
                    sem = np.sqrt(weighted_var / n_effective)
                    stats[x] = {'mean': mean, 'std': np.sqrt(weighted_var), 'sem_lower': mean - sem, 'sem_upper': mean + sem}
        else:
            for x in x_coords:
                values = values_by_x[x]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    sem = std / np.sqrt(len(values))
                    stats[x] = {'mean': mean, 'std': std, 'sem_lower': mean - sem, 'sem_upper': mean + sem}
        prev_stats = {'mean': 0, 'sem_lower': 0, 'sem_upper': 0, 'std': 0}
        for x in sorted(stats.keys()):
            stats[x] = {'mean': max(stats[x]['mean'], prev_stats['mean']), 'sem_lower': max(stats[x]['sem_lower'], prev_stats['sem_lower']), 'sem_upper': max(stats[x]['sem_upper'], prev_stats['sem_upper']), 'std': stats[x]['std']}
            prev_stats = stats[x]
        return stats

    def _calculate_statistics_ci(self, roots: List[Node], max_depth: int) -> Dict:
        """Calculate statistics with cumulative achievement tracking"""
        values_by_x = defaultdict(list)
        for root in roots:
            values_by_x[0].append(0)
            stack = [(root, 0, 0, 0, defaultdict(int))]
            while stack:
                node, depth, prev_ticks, prev_value, prev_achievements = stack.pop()
                if depth > max_depth:
                    continue
                current_achievements = prev_achievements.copy()
                for item, quantity in node.static_achievements.items():
                    current_achievements[item] += quantity
                for item, quantity in node.dynamic_achievements.items():
                    current_achievements[item] += quantity
                if self.use_value_gdp:
                    current_value = 0
                    for item, quantity in current_achievements.items():
                        item_value = self.value_calculator.get_value(item)
                        current_value += item_value * quantity
                else:
                    current_value = prev_value + node.metrics['value']
                ticks = prev_ticks + node.metrics['ticks']
                x_coord = ticks if self.x_axis == 'ticks' else depth
                values_by_x[x_coord].append(current_value)
                for child in node.children:
                    stack.append((child, depth + 1, ticks, current_value, current_achievements))
        stats = {0: {'mean': 0, 'ci_lower': 0, 'ci_upper': 0, 'std': 0}}
        x_coords = sorted((x for x in values_by_x.keys() if x > 0))
        if self.x_axis == 'ticks':
            eval_points = np.logspace(np.log(min(x_coords)) / np.log(self.x_base), np.log(max(x_coords)) / np.log(self.x_base), 500, base=self.x_base)
            window = 0.1
            for x in eval_points:
                nearby_values = []
                for orig_x, values in values_by_x.items():
                    if orig_x > 0:
                        log_diff = abs(np.log(x) / np.log(self.x_base) - np.log(orig_x) / np.log(self.x_base))
                        if log_diff < window:
                            weight = np.exp(-(log_diff / window) ** 2)
                            nearby_values.extend(((v, weight) for v in values))
                if nearby_values:
                    values, weights = zip(*nearby_values)
                    stats[x] = {'mean': np.average(values, weights=weights), 'std': np.std(values), 'ci_lower': self.weighted_percentile(values, weights, 2.5), 'ci_upper': self.weighted_percentile(values, weights, 97.5)}
        else:
            for x in x_coords:
                values = values_by_x[x]
                if values:
                    stats[x] = {'mean': np.mean(values), 'std': np.std(values), 'ci_lower': np.percentile(values, 2.5), 'ci_upper': np.percentile(values, 97.5)}
        prev_stats = {'mean': 0, 'ci_lower': 0, 'ci_upper': 0, 'std': 0}
        for x in sorted(stats.keys()):
            stats[x] = {'mean': max(stats[x]['mean'], prev_stats['mean']), 'ci_lower': max(stats[x]['ci_lower'], prev_stats['ci_lower']), 'ci_upper': max(stats[x]['ci_upper'], prev_stats['ci_upper']), 'std': stats[x]['std']}
            prev_stats = stats[x]
        return stats

    def _add_achievement_icon(self, ax, item_name: str, x: float, y: float, color: str):
        """Add achievement icon with background circle"""
        try:
            self._prepare_icon(item_name)
            icon_path = f'icons/{item_name}.png'
            if not os.path.exists(icon_path):
                return
            circle_img = self.create_circle_background(color)
            circle_box = OffsetImage(circle_img, zoom=0.2)
            circle_box.image.axes = ax
            ab_circle = AnnotationBbox(circle_box, (x, y), frameon=False, box_alignment=(0.5, 0.5), pad=0)
            ax.add_artist(ab_circle)
            icon = plt.imread(icon_path)
            icon_box = OffsetImage(icon, zoom=0.1)
            ab = AnnotationBbox(icon_box, (x, y), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f'Failed to add icon for {item_name}: {e}')

