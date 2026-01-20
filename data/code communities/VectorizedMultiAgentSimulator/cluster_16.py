# Cluster 16

def get_reference_paths(map_data):
    """This function returns the (long-term) reference paths."""
    reference_paths_all = []
    reference_paths_intersection = []
    reference_paths_merge_in = []
    reference_paths_merge_out = []
    path_intersection = [[11, 25, 13], [11, 26, 52, 37], [11, 72, 91], [12, 18, 14], [12, 17, 43, 38], [12, 73, 92], [39, 51, 37], [39, 50, 102, 91], [39, 20, 63], [40, 44, 38], [40, 45, 97, 92], [40, 21, 64], [89, 103, 91], [89, 104, 78, 63], [89, 46, 13], [90, 96, 92], [90, 95, 69, 64], [90, 47, 14], [65, 77, 63], [65, 76, 24, 13], [65, 98, 37], [66, 70, 64], [66, 71, 19, 14], [66, 99, 38]]
    path_merge_in = [[34, 32], [33, 31], [35, 31], [36, 49]]
    path_merge_out = [[6, 8], [5, 7], [5, 9], [23, 10]]
    path_to_loop = {1: (1, 4), 2: (2, 1), 3: (3, 64), 4: (4, 42), 5: (5, 22), 6: (6, 39), 7: (7, 15), 8: (1, 8), 9: (2, 10), 10: (3, 75), 11: (4, 45), 12: (5, 59), 13: (6, 61), 14: (7, 5), 15: (1, 58), 16: (2, 17), 17: (3, 79), 18: (4, 92), 19: (5, 68), 20: (6, 55), 21: (7, 11), 22: (1, 54), 23: (2, 38), 24: (3, 88), 25: (4, 100), 26: (5, 19), 27: (6, 65), 28: (7, 93), 29: (1, 82), 30: (2, 49), 31: (3, 95), 32: (4, 33), 33: (5, 14), 34: (6, 35), 35: (7, 83), 36: (1, 86), 37: (6, 29), 38: (7, 89), 39: (1, 32), 40: (1, 28)}
    lanelets_share_same_boundaries_list = [[4, 3, 22], [6, 5, 23], [8, 7], [60, 59], [58, 57, 75], [56, 55, 74], [54, 53], [80, 79], [82, 81, 100], [84, 83, 101], [86, 85], [34, 33], [32, 31, 49], [30, 29, 48], [28, 27], [2, 1], [13, 14], [15, 16], [9, 10], [11, 12], [63, 64], [61, 62], [67, 68], [65, 66], [91, 92], [93, 94], [87, 88], [89, 90], [37, 38], [35, 36], [41, 42], [39, 40], [25, 18], [26, 17], [52, 43], [72, 73], [51, 44], [50, 45], [102, 97], [20, 21], [103, 96], [104, 95], [78, 69], [46, 47], [77, 70], [76, 71], [24, 19], [98, 99]]
    num_paths_all = len(path_to_loop)
    for ref_path_id in range(num_paths_all):
        reference_lanelets_index = get_reference_lanelet_index(ref_path_id + 1, path_to_loop)
        reference_path = calculate_reference_path(reference_lanelets_index, map_data, lanelets_share_same_boundaries_list)
        reference_paths_all.append(reference_path)
    for reference_lanelets_index in path_intersection:
        reference_path = calculate_reference_path(reference_lanelets_index, map_data, lanelets_share_same_boundaries_list)
        reference_paths_intersection.append(reference_path)
    for reference_lanelets_index in path_merge_in:
        reference_path = calculate_reference_path(reference_lanelets_index, map_data, lanelets_share_same_boundaries_list)
        reference_paths_merge_in.append(reference_path)
    for reference_lanelets_index in path_merge_out:
        reference_path = calculate_reference_path(reference_lanelets_index, map_data, lanelets_share_same_boundaries_list)
        reference_paths_merge_out.append(reference_path)
    return (reference_paths_all, reference_paths_intersection, reference_paths_merge_in, reference_paths_merge_out)

def get_reference_lanelet_index(ref_path_id, path_to_loop):
    """
    Get loop of lanelets used for reference_path_struct.

    Args:
    ref_path_id (int): Path ID.

    Returns:
    list: List of lanelets indices.
    """
    reference_lanelets_loops = [[4, 6, 8, 60, 58, 56, 54, 80, 82, 84, 86, 34, 32, 30, 28, 2], [1, 3, 23, 10, 12, 17, 43, 38, 36, 49, 29, 27], [64, 62, 75, 55, 53, 79, 81, 101, 88, 90, 95, 69], [40, 45, 97, 92, 94, 100, 83, 85, 33, 31, 48, 42], [5, 7, 59, 57, 74, 68, 66, 71, 19, 14, 16, 22], [41, 39, 20, 63, 61, 57, 55, 67, 65, 98, 37, 35, 31, 29], [3, 5, 9, 11, 72, 91, 93, 81, 83, 87, 89, 46, 13, 15]]
    loop_index, starting_lanelet = path_to_loop.get(ref_path_id, (None, None))
    if loop_index is not None:
        reference_lanelets_loop = reference_lanelets_loops[loop_index - 1]
        index_starting_lanelet = reference_lanelets_loop.index(starting_lanelet)
        lanelets_index = reference_lanelets_loop[index_starting_lanelet:] + reference_lanelets_loop[:index_starting_lanelet]
        return lanelets_index
    else:
        return []

def calculate_reference_path(reference_lanelets_index, map_data, lanelets_share_same_boundaries_list):
    left_boundaries = None
    right_boundaries = None
    left_boundaries_shared = None
    right_boundaries_shared = None
    center_lines = None
    for lanelet in reference_lanelets_index:
        lanelets_share_same_boundaries = next((group for group in lanelets_share_same_boundaries_list if lanelet in group), None)
        left_bound = map_data['lanelets'][lanelet - 1]['left_boundary']
        right_bound = map_data['lanelets'][lanelet - 1]['right_boundary']
        left_bound_shared = map_data['lanelets'][lanelets_share_same_boundaries[0] - 1]['left_boundary']
        right_bound_shared = map_data['lanelets'][lanelets_share_same_boundaries[-1] - 1]['right_boundary']
        if left_boundaries is None:
            left_boundaries = left_bound
            right_boundaries = right_bound
            left_boundaries_shared = left_bound_shared
            right_boundaries_shared = right_bound_shared
        else:
            if torch.norm(left_boundaries[-1, :] - left_bound[0, :]) < 0.0001:
                left_boundaries = torch.cat((left_boundaries, left_bound[1:, :]), dim=0)
                left_boundaries_shared = torch.cat((left_boundaries_shared, left_bound_shared[1:, :]), dim=0)
            else:
                left_boundaries = torch.cat((left_boundaries, left_bound), dim=0)
                left_boundaries_shared = torch.cat((left_boundaries_shared, left_bound_shared), dim=0)
            if torch.norm(right_boundaries[-1, :] - right_bound[0, :]) < 0.0001:
                right_boundaries = torch.cat((right_boundaries, right_bound[1:, :]), dim=0)
                right_boundaries_shared = torch.cat((right_boundaries_shared, right_bound_shared[1:, :]), dim=0)
            else:
                right_boundaries = torch.cat((right_boundaries, right_bound), dim=0)
                right_boundaries_shared = torch.cat((right_boundaries_shared, right_bound_shared), dim=0)
    center_lines = (left_boundaries + right_boundaries) / 2
    is_loop = (center_lines[0, :] - center_lines[-1, :]).norm() <= 0.0001
    center_lines_vec = torch.diff(center_lines, dim=0)
    center_lines_vec_length = torch.norm(center_lines_vec, dim=1)
    center_lines_vec_mean_length = torch.mean(center_lines_vec_length)
    center_lines_vec_normalized = center_lines_vec / center_lines_vec_length.unsqueeze(1)
    center_line_yaw = torch.atan2(center_lines_vec[:, 1], center_lines_vec[:, 0])
    reference_path = {'reference_lanelets': reference_lanelets_index, 'left_boundary': left_boundaries, 'right_boundary': right_boundaries, 'left_boundary_shared': left_boundaries_shared, 'right_boundary_shared': right_boundaries_shared, 'center_line': center_lines, 'center_line_yaw': center_line_yaw, 'center_line_vec_normalized': center_lines_vec_normalized, 'center_line_vec_mean_length': center_lines_vec_mean_length, 'is_loop': is_loop}
    return reference_path

