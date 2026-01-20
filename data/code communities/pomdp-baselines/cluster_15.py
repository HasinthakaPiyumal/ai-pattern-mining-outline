# Cluster 15

class RandomWeakPushCartPole(ModifiableCartPoleEnv):

    def __init__(self):
        super(RandomWeakPushCartPole, self).__init__()
        self.force_mag = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FORCE_MAG, self.EXTREME_UPPER_FORCE_MAG, self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.force_mag = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FORCE_MAG, self.EXTREME_UPPER_FORCE_MAG, self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomWeakPushCartPole, self).parameters
        parameters.update({'force': self.force_mag})
        return parameters

def uniform_exclude_inner(np_uniform, a, b, a_i, b_i):
    """Draw sample from uniform distribution, excluding an inner range"""
    if not (a < a_i and b_i < b):
        raise ValueError('Bad range, inner: ({},{}), outer: ({},{})'.format(a, b, a_i, b_i))
    while True:
        result = np_uniform(a, b)
        if a <= result and result < a_i or (b_i <= result and result < b):
            return result

class RandomShortPoleCartPole(ModifiableCartPoleEnv):

    def __init__(self):
        super(RandomShortPoleCartPole, self).__init__()
        self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        self._followup()

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomShortPoleCartPole, self).parameters
        parameters.update({'length': self.length})
        return parameters

class RandomLightPoleCartPole(ModifiableCartPoleEnv):

    def __init__(self):
        super(RandomLightPoleCartPole, self).__init__()
        self.masspole = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASSPOLE, self.EXTREME_UPPER_MASSPOLE, self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)
        self._followup()

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.masspole = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASSPOLE, self.EXTREME_UPPER_MASSPOLE, self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomLightPoleCartPole, self).parameters
        parameters.update({'mass': self.masspole})
        return parameters

class RandomExtremeCartPole(ModifiableCartPoleEnv):

    def __init__(self):
        super(RandomExtremeCartPole, self).__init__()
        '\n        self.force_mag = self.np_random.uniform(self.LOWER_FORCE_MAG, self.UPPER_FORCE_MAG)\n        self.length = self.np_random.uniform(self.LOWER_LENGTH, self.UPPER_LENGTH)\n        self.masspole = self.np_random.uniform(self.LOWER_MASSPOLE, self.UPPER_MASSPOLE)\n        '
        self.force_mag = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FORCE_MAG, self.EXTREME_UPPER_FORCE_MAG, self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)
        self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        self.masspole = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASSPOLE, self.EXTREME_UPPER_MASSPOLE, self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)
        self._followup()

    def reset(self, new=True):
        self.nsteps = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        '\n        self.force_mag = self.np_random.uniform(self.LOWER_FORCE_MAG, self.UPPER_FORCE_MAG)\n        self.length = self.np_random.uniform(self.LOWER_LENGTH, self.UPPER_LENGTH)\n        self.masspole = self.np_random.uniform(self.LOWER_MASSPOLE, self.UPPER_MASSPOLE)\n        '
        if new:
            self.force_mag = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FORCE_MAG, self.EXTREME_UPPER_FORCE_MAG, self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)
            self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
            self.masspole = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASSPOLE, self.EXTREME_UPPER_MASSPOLE, self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomExtremeCartPole, self).parameters
        parameters.update({'force_mag': self.force_mag, 'length': self.length, 'masspole': self.masspole, 'total_mass': self.total_mass, 'polemass_length': self.polemass_length})
        return parameters

class RandomWeakForceMountainCar(ModifiableMountainCarEnv):

    def reset(self, new=True):
        if new:
            self.force = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FORCE, self.EXTREME_UPPER_FORCE, self.RANDOM_LOWER_FORCE, self.RANDOM_UPPER_FORCE)
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomWeakForceMountainCar, self).parameters
        parameters.update({'force': self.force})
        return parameters

class RandomLightCarMountainCar(ModifiableMountainCarEnv):

    def reset(self, new=True):
        if new:
            self.mass = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASS, self.EXTREME_UPPER_MASS, self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomLightCarMountainCar, self).parameters
        parameters.update({'mass': self.mass})
        return parameters

class RandomExtremeMountainCar(ModifiableMountainCarEnv):

    def reset(self, new=True):
        self.nsteps = 0
        if new:
            self.force = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FORCE, self.EXTREME_UPPER_FORCE, self.RANDOM_LOWER_FORCE, self.RANDOM_UPPER_FORCE)
            self.mass = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASS, self.EXTREME_UPPER_MASS, self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomExtremeMountainCar, self).parameters
        parameters.update({'force': self.force, 'mass': self.mass})
        return parameters

class RandomLightPendulum(ModifiablePendulumEnv):

    def __init__(self):
        super(RandomLightPendulum, self).__init__()
        self.mass = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASS, self.EXTREME_UPPER_MASS, self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)

    def reset(self, new=True):
        if new:
            self.mass = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASS, self.EXTREME_UPPER_MASS, self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        return super(RandomLightPendulum, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomLightPendulum, self).parameters
        parameters.update({'mass': self.mass})
        return parameters

class RandomShortPendulum(ModifiablePendulumEnv):

    def __init__(self):
        super(RandomShortPendulum, self).__init__()
        self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)

    def reset(self, new=True):
        if new:
            self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        return super(RandomShortPendulum, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomShortPendulum, self).parameters
        parameters.update({'length': self.length})
        return parameters

class RandomExtremePendulum(ModifiablePendulumEnv):

    def __init__(self):
        super(RandomExtremePendulum, self).__init__()
        self.mass = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASS, self.EXTREME_UPPER_MASS, self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)

    def reset(self, new=True):
        if new:
            self.mass = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASS, self.EXTREME_UPPER_MASS, self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
            self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        return super(RandomExtremePendulum, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremePendulum, self).parameters
        parameters.update({'mass': self.mass, 'length': self.length})
        return parameters

class RandomLightAcrobot(ModifiableAcrobotEnv):

    def __init__(self):
        super(RandomLightAcrobot, self).__init__()
        self.mass = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASS, self.EXTREME_UPPER_MASS, self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)

    def reset(self, new=True):
        if new:
            self.mass = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASS, self.EXTREME_UPPER_MASS, self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        return super(RandomLightAcrobot, self).reset(new)

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def parameters(self):
        parameters = super(RandomLightAcrobot, self).parameters
        parameters.update({'mass': self.mass})
        return parameters

class RandomShortAcrobot(ModifiableAcrobotEnv):

    def __init__(self):
        super(RandomShortAcrobot, self).__init__()
        self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)

    def reset(self, new=True):
        if new:
            self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        return super(RandomShortAcrobot, self).reset(new)

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def parameters(self):
        parameters = super(RandomShortAcrobot, self).parameters
        parameters.update({'length': self.length})
        return parameters

class RandomLowInertiaAcrobot(ModifiableAcrobotEnv):

    def __init__(self):
        super(RandomLowInertiaAcrobot, self).__init__()
        self.inertia = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_INERTIA, self.EXTREME_UPPER_INERTIA, self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA)

    def reset(self, new=True):
        if new:
            self.inertia = self.np_random.uniform(self.np_random.uniform, self.EXTREME_LOWER_INERTIA, self.EXTREME_UPPER_INERTIA, self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA)
        return super(RandomLowInertiaAcrobot, self).reset(new)

    @property
    def LINK_MOI(self):
        return self.inertia

    @property
    def parameters(self):
        parameters = super(RandomLowInertiaAcrobot, self).parameters
        parameters.update({'inertia': self.inertia})
        return parameters

class RandomExtremeAcrobot(ModifiableAcrobotEnv):

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def LINK_MOI(self):
        return self.inertia

    def __init__(self):
        super(RandomExtremeAcrobot, self).__init__()
        self.mass = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASS, self.EXTREME_UPPER_MASS, self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        self.inertia = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_INERTIA, self.EXTREME_UPPER_INERTIA, self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA)

    def reset(self, new=True):
        if new:
            self.mass = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASS, self.EXTREME_UPPER_MASS, self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
            self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH, self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
            self.inertia = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_INERTIA, self.EXTREME_UPPER_INERTIA, self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA)
        return super(RandomExtremeAcrobot, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremeAcrobot, self).parameters
        parameters.update({'mass': self.mass, 'length': self.length, 'inertia': self.inertia})
        return parameters

