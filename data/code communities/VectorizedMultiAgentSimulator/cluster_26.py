# Cluster 26

class Viewer(object):

    def __init__(self, width, height, display=None, visible=True):
        display = get_display(display)
        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height, display=display, visible=visible)
        self.window.on_close = self.window_closed_by_user
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()
        self.bounds = None
        glEnable(GL_BLEND)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        self.bounds = torch.tensor([left, right, bottom, top], device=left.device)
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def add_onetime_list(self, geoms):
        self.onetime_geoms.extend(geoms)

    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        text_lines = []
        for geom in chain(self.geoms, self.onetime_geoms):
            if isinstance(geom, TextLine):
                text_lines.append(geom)
            else:
                geom.render()
        self.transform.disable()
        for text in text_lines:
            text.render()
        pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
        pyglet.gl.glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        arr = None
        if return_rgb_array:
            arr = self.get_array()
        self.window.flip()
        self.onetime_geoms = []
        return arr

    def get_array(self):
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape((buffer.height, buffer.width, 4))
        arr = arr[::-1, :, 0:3]
        return arr

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise RuntimeError('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

