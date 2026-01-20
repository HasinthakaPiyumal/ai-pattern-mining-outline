# Cluster 12

def print_all_tensors_name(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print(key)
    except Exception as e:
        print(str(e))
        if 'corrupted compressed block contents' in str(e):
            print("It's likely that your checkpoint file has been compressed with SNAPPY.")

