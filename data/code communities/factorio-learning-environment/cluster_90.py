# Cluster 90

def run_and_print_results(game: FactorioInstance, num_iterations: int=100):
    results = run_benchmark(game, num_iterations)
    print(f'Benchmark Results (iterations: {num_iterations}):')
    print('-' * 80)
    print(f'{'Operation':<20} {'Ops/Min':<10} {'Ops/Sec':<10} {'Duration':<10} {'Count':<10}')
    print('-' * 80)
    for name, data in results.items():
        print(f'{name:<20} {data['ops_per_minute']:.2f} {data['ops_per_second']:.2f} {data['duration']:.2f}s {data['operations']}')
    total_ops = sum((data['operations'] for data in results.values()))
    total_duration = sum((data['duration'] for data in results.values()))
    total_ops_per_minute = total_ops / total_duration * 60
    print('-' * 80)
    print(f'{'Total':<20} {total_ops_per_minute:.2f} {total_ops / total_duration:.2f} {total_duration:.2f}s {total_ops}')

def run_benchmark(game: FactorioInstance, num_iterations: int=100):
    benchmarks = {'place_entity_next_to': benchmark_place_entity_next_to, 'place_entity': benchmark_place_entity, 'move_to': benchmark_move_to, 'harvest_resource': benchmark_harvest_resource, 'craft_item': benchmark_craft_item, 'connect_entities': benchmark_connect_entities, 'rotate_entity': benchmark_rotate_entity, 'insert_item': benchmark_insert_item, 'extract_item': benchmark_extract_item, 'inspect_inventory': benchmark_inspect_inventory, 'get_resource_patch': benchmark_get_resource_patch}
    results = {}
    game.set_speed(10)
    for name, func in benchmarks.items():
        start_time = time.time()
        count = func(game, num_iterations)
        end_time = time.time()
        game.reset()
        duration = end_time - start_time
        ops_per_second = count / duration
        ops_per_minute = ops_per_second * 60
        results[name] = {'operations': count, 'duration': duration, 'ops_per_second': ops_per_second, 'ops_per_minute': ops_per_minute}
    return results

