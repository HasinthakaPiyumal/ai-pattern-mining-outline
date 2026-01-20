# Cluster 19

class Category(Base):
    """
    A category within our taxonomy. Includes both things (e.g. cars) or stuff (e.g. lanes, sidewalks).
    Subcategories are delineated by a period.
    """
    __tablename__ = 'category'
    token = Column(sql_types.HexLen8, primary_key=True)
    name = Column(String(64))
    description = Column(Text)

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    def __repr__(self) -> str:
        """
        Return the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def color(self) -> Tuple[int, int, int]:
        """
        Get category color.
        :return: The category color tuple.
        """
        c: Tuple[int, int, int] = default_color(self.name)
        return c

    @property
    def color_np(self) -> npt.NDArray[np.float64]:
        """
        Get category color in numpy.
        :return: The category color in numpy.
        """
        c: npt.NDArray[np.float64] = default_color_np(self.name)
        return c

def default_color(category_name: str) -> Tuple[int, int, int]:
    """
    Get the default color for a category.

    :param category_name: Category name.
    :return: Default RGB color tuple.
    """
    if 'cycle' in category_name:
        return MotionalColor.RADICAL_RED.to_rgb_tuple()
    elif 'vehicle' in category_name:
        return MotionalColor.PUMPKIN.to_rgb_tuple()
    elif 'human.pedestrian' in category_name:
        return MotionalColor.BLUE.to_rgb_tuple()
    elif 'cone' in category_name or 'barrier' in category_name:
        return MotionalColor.BLACK.to_rgb_tuple()
    elif category_name == 'flat.driveable_surface':
        return MotionalColor.ORANGE.to_rgb_tuple()
    elif category_name == 'flat':
        return MotionalColor.SPRING_GREEN.to_rgb_tuple()
    elif category_name == 'vehicle.ego':
        return MotionalColor.ELECTRIC_VIOLET.to_rgb_tuple()
    else:
        return MotionalColor.MAGENTA.to_rgb_tuple()

def default_color_np(category_name: str) -> npt.NDArray[np.float64]:
    """
    Get the default color for a category in numpy.

    :param category_name: Category name.
    :return: <np.float: 3> RGB color.
    """
    return np.array(default_color(category_name)) / 255.0

