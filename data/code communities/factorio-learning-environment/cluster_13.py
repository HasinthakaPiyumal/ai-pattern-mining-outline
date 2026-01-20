# Cluster 13

def build_ingredient_tree(recipe_name, graph, visited=None):
    if visited is None:
        visited = set()
    if recipe_name not in graph:
        return {'name': recipe_name, 'ingredients': []}
    if recipe_name in visited:
        return {'name': recipe_name, 'ingredients': []}
    visited.add(recipe_name)
    recipe = graph[recipe_name]
    ingredient_tree = {'name': recipe_name, 'ingredients': []}
    for ingredient in recipe:
        sub_tree = build_ingredient_tree(ingredient['name'], graph, visited)
        sub_tree['amount'] = ingredient['amount']
        sub_tree['type'] = ingredient['type']
        ingredient_tree['ingredients'].append(sub_tree)
    visited.remove(recipe_name)
    return ingredient_tree

