# Cluster 87

def set_model_params(loc, rot, rot_mode='XYZ', model_obj_name='Car', target_color=None):
    """
    Args:
        loc: list
            [x, y, z]
        rot: list
            [angle1, angle2, angle3] (rad.)
        rot_mode: str
            Euler angle order
        model_obj_name: str
            name of the entire model. New obj name.
        target_color: dict (optinoal)
            {"material_key":.., "color": ...}
    """
    model = bpy.data.objects[model_obj_name]
    model.location = loc
    model.rotation_mode = rot_mode
    model.rotation_euler = rot
    if target_color is not None:
        modify_car_color(model, target_color['material_key'], target_color['color'])

def modify_car_color(model: bpy.types.Object, material_key, color):
    """
    Args:
        model: bpy_types.Objct
            car model
        material_key: str
            key name in model.material_slots. Refer to the car paint material.
        color: list of float
            target base color, [R,G,B,alpha] 
    """
    material = model.material_slots[material_key].material
    material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color

def add_model_params(model_setting):
    """
    model_setting includes:
        - blender_file: path to object model file
        - insert_pos: list of len 3
        - insert_rot: list of len 3 
        - model_obj_name: object name within blender_file
        - new_obj_name: object name in this scene
        - target_color: optional .
    """
    blender_file = model_setting['blender_file']
    model_obj_name = model_setting['model_obj_name']
    new_obj_name = model_setting['new_obj_name']
    target_color = model_setting.get('target_color', None)
    with bpy.data.libraries.load(blender_file, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects
    for obj in data_to.objects:
        if obj.name == model_obj_name:
            bpy.context.collection.objects.link(obj)
    if model_obj_name in bpy.data.objects:
        imported_object = bpy.data.objects[model_obj_name]
        imported_object.name = new_obj_name
        print(f'rename {model_obj_name} to {new_obj_name}')
    for slot in imported_object.material_slots:
        material = slot.material
        if material:
            material.name = new_obj_name + '_' + material.name
    if target_color is not None:
        target_color['material_key'] = new_obj_name + '_' + target_color['material_key']
    set_model_params(model_setting['insert_pos'], model_setting['insert_rot'], rot_mode='XYZ', model_obj_name=new_obj_name, target_color=target_color)

