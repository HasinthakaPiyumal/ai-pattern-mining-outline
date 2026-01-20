# Cluster 71

class FrameInfoTest(unittest.TestCase):

    def test_frame_info_equality(self):
        info1 = FrameInfo(height=250, width=250, channels=3, color_space=ColorSpace.GRAY)
        info2 = FrameInfo(250, 250, 3, color_space=ColorSpace.GRAY)
        self.assertEqual(info1, info2)

