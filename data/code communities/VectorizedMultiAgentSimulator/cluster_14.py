# Cluster 14

def parse_bound(element, device):
    """Parses a bound (left boundary or right boundary) element to extract points and line marking."""
    points = [parse_point(point, device) for point in element.findall('point')]
    points = torch.vstack(points)
    line_marking = element.find('lineMarking').text if element.find('lineMarking') is not None else None
    return (points, line_marking)

def parse_point(element, device):
    """Parses a point element to extract x and y coordinates."""
    x = float(element.find('x').text) if element.find('x') is not None else None
    y = float(element.find('y').text) if element.find('y') is not None else None
    return torch.tensor([x, y], device=device)

def parse_lanelet(element, device):
    """Parses a lanelet element to extract detailed information."""
    lanelet_data = {'id': int(element.get('id')), 'left_boundary': [], 'left_boundary_center_points': [], 'left_boundary_lengths': [], 'left_boundary_yaws': [], 'left_line_marking': None, 'right_boundary': [], 'right_boundary_center_points': [], 'right_boundary_lengths': [], 'right_boundary_yaws': [], 'right_line_marking': None, 'center_line': [], 'center_line_center_points': [], 'center_line_lengths': [], 'center_line_yaws': [], 'center_line_marking': 'dashed', 'predecessor': [], 'successor': [], 'adjacent_left': None, 'adjacent_right': None, 'lanelet_type': None}
    for child in element:
        if child.tag == 'leftBound':
            lanelet_data['left_boundary'], lanelet_data['left_line_marking'] = parse_bound(child, device)
        elif child.tag == 'rightBound':
            lanelet_data['right_boundary'], lanelet_data['right_line_marking'] = parse_bound(child, device)
        elif child.tag == 'predecessor':
            lanelet_data['predecessor'].append(int(child.get('ref')))
        elif child.tag == 'successor':
            lanelet_data['successor'].append(int(child.get('ref')))
        elif child.tag == 'adjacentLeft':
            lanelet_data['adjacent_left'] = {'id': int(child.get('ref')), 'drivingDirection': child.get('drivingDir')}
        elif child.tag == 'adjacentRight':
            lanelet_data['adjacent_right'] = {'id': int(child.get('ref')), 'drivingDirection': child.get('drivingDir')}
        elif child.tag == 'lanelet_type':
            lanelet_data['lanelet_type'] = child.text
    lanelet_data['center_line'] = (lanelet_data['left_boundary'] + lanelet_data['right_boundary']) / 2
    lanelet_data['center_line_center_points'], lanelet_data['center_line_lengths'], lanelet_data['center_line_yaws'], _ = get_center_length_yaw_polyline(polyline=lanelet_data['center_line'])
    lanelet_data['left_boundary_center_points'], lanelet_data['left_boundary_lengths'], lanelet_data['left_boundary_yaws'], _ = get_center_length_yaw_polyline(polyline=lanelet_data['left_boundary'])
    lanelet_data['right_boundary_center_points'], lanelet_data['right_boundary_lengths'], lanelet_data['right_boundary_yaws'], _ = get_center_length_yaw_polyline(polyline=lanelet_data['right_boundary'])
    return lanelet_data

def get_center_length_yaw_polyline(polyline: torch.Tensor):
    """This function calculates the center points, lengths, and yaws of all line segments of the given polyline."""
    center_points = polyline.unfold(0, 2, 1).mean(dim=2)
    polyline_vecs = polyline.diff(dim=0)
    lengths = polyline_vecs.norm(dim=1)
    yaws = torch.atan2(polyline_vecs[:, 1], polyline_vecs[:, 0])
    return (center_points, lengths, yaws, polyline_vecs)

def get_map_data(map_file_path, device=None):
    """This function returns the map data."""
    if device is None:
        device = torch.device('cpu')
    tree = ET.parse(map_file_path)
    root = tree.getroot()
    lanelets = []
    intersection_info = []
    for child in root:
        if child.tag == 'lanelet':
            lanelets.append(parse_lanelet(child, device))
        elif child.tag == 'intersection':
            intersection_info = parse_intersections(child)
    mean_lane_width = torch.mean(torch.norm(torch.vstack([lanelets[i]['left_boundary'] for i in range(len(lanelets))]) - torch.vstack([lanelets[i]['right_boundary'] for i in range(len(lanelets))]), dim=1))
    map_data = {'lanelets': lanelets, 'intersection_info': intersection_info, 'mean_lane_width': mean_lane_width}
    return map_data

def parse_intersections(element):
    """This function parses the lanes of the intersection."""
    intersection_info = []
    for incoming in element.findall('incoming'):
        incoming_info = {'incomingLanelet': int(incoming.find('incomingLanelet').get('ref')), 'successorsRight': int(incoming.find('successorsRight').get('ref')), 'successorsStraight': [int(s.get('ref')) for s in incoming.findall('successorsStraight')], 'successorsLeft': int(incoming.find('successorsLeft').get('ref'))}
        intersection_info.append(incoming_info)
    return intersection_info

