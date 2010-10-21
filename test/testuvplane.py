import unittest
from pyautoplot.uvplane import *



class UvplaneTest(unittest.TestCase):
    def test_quadrant(self):
        self.assertEquals(quadrant(0.0), 0)
        self.assertEquals(quadrant(-1e-9), 3)
        self.assertEquals(quadrant(pi), 2)
        self.assertEquals(quadrant(3*pi-1e-9), 1)
        self.assertEquals(quadrant(-4*pi-1e-9), 3)
        self.assertEquals(quadrant(-4*pi+1e-9), 0)
        pass

    def test_fixup_rgb(self):
        pass

    def test_color_from_angle(self):
        pass

    def test_rgb_scale_palette(self):
        pass

    def test_phase_palette(self):
        pass

    def test_rgb_from_complex_image(self):
        pass

    def test_read_fits_image(self):
        pass
        
    pass


if __name__ == '__main__':
    unittest.main()
    pass
