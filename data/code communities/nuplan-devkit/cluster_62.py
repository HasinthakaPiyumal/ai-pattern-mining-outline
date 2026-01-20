# Cluster 62

def compare_map_objects(map_objects_1: List[GraphEdgeMapObject], map_objects_2: List[GraphEdgeMapObject]) -> None:
    """
    Compares two lists of GraphEdgeMapObjects. Note, only the first list will be rendered.
    :param map_objects_1: First list of GraphEdgeMapObjects.
    :param map_objects_2: Second list of GraphEdgeMapObjects.
    """
    assert type(map_objects_1) == type(map_objects_2), f'Map objects are not of the same type.Got {type(map_objects_1)} and {type(map_objects_2)}'
    map_object_2_dict = {lc.id: lc for lc in map_objects_2}
    assert {lc.id for lc in map_objects_1} == set(map_object_2_dict.keys())
    for map_object_1 in map_objects_1:
        map_object_2 = map_object_2_dict[map_object_1.id]
        blp_1 = map_object_1.baseline_path.discrete_path
        blp_2 = map_object_2.baseline_path.discrete_path
        compare_poses(blp_1[0], blp_2[0])
        compare_poses(blp_1[-1], blp_2[-1])

def compare_poses(pose1: StateSE2, pose2: StateSE2) -> None:
    """
    Compare x, y, and heading attribute of a StateSE2.
    :param pose1: first pose for comparing.
    :param pose2: second pose for comparing.
    """
    assert pose1.x == pytest.approx(pose2.x, 0.001)
    assert pose1.y == pytest.approx(pose2.y, 0.001)
    assert pose1.heading == pytest.approx(pose2.heading, 0.001)

