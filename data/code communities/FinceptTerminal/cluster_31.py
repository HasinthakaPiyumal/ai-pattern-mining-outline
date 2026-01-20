# Cluster 31

def keyboard_handler(sender, key_code):
    """Handle keyboard events"""
    global selected_node_for_deletion
    if key_code == 68:
        delete_selected_node()

def delete_selected_node():
    """Delete the currently selected node"""
    global selected_node_for_deletion
    if selected_node_for_deletion is not None:
        delete_node(selected_node_for_deletion)
        selected_node_for_deletion = None

