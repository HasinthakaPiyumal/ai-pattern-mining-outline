# Cluster 9

def bar(results_file, vid_width, vid_height, annotation_path):
    foo()
    print('bar')
    print(f'Export Function Check: results_file: {results_file} | vid_width: {vid_width} | vid_height: {vid_height} | annotation_path: {annotation_path}')
    return annotation_path

def foo():
    print('foo')

def baz(json_paths, annotation_path):
    foo()
    print('baz')
    print(f'Export Function Check: json_paths {json_paths} | annotation_path: {annotation_path}')
    return annotation_path

