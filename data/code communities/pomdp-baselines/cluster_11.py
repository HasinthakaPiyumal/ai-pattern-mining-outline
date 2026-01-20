# Cluster 11

def sample_textures(textures, sides=True, sectors=True):
    """Perform texture sampling.

    :param textures: A set of textures to sample from
    :param sides: Update all side textures (walls)
    :param sectors: Update all sector textures (floor, ceiling)
    """

    @sampler_with_map_editor
    def sampler(env, config, editor):
        """Perform texture sampling.

        :param env: Environment instance
        :param config: Configuration dictionary
        :param editor: Map editor
        """
        if sides:
            for side in editor.sidedefs:
                side.texturemiddle = str(env.np_random.choice(textures))
        if sectors:
            for sector in editor.sectors:
                sector.texturefloor = str(env.np_random.choice(textures))
                sector.textureceiling = str(env.np_random.choice(textures))
    return sampler

def sample_things(things, modify_things):
    """Perform thing sampling.

    :param things: A set of things to sample from
    :param modify_things: A set of things that can be modified
    """

    @sampler_with_map_editor
    def sampler(env, config, editor):
        """Perform thing sampling.

        :param env: Environment instance
        :param config: Configuration dictionary
        :param editor: Map editor
        """
        for thing in editor.things:
            if thing.type not in modify_things:
                continue
            thing.type = int(env.np_random.choice(things))
    return sampler

