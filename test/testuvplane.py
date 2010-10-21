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
        a = array([[0.99, 1.0, 1.001],[0.001, 0.0, -0.001]])
        f = fixup_rgb(a)

        empty=fixup_rgb(array([]))
        self.assertEquals(empty.shape, (0,))
        
        self.assertEquals(len(a.shape), 2)
        self.assertEquals(len(f.shape), 2)
        self.assertEquals(a.shape[0], f.shape[0])
        self.assertEquals(a.shape[1], f.shape[1])

        self.assertAlmostEquals(a[0,0], 0.99)
        self.assertAlmostEquals(a[0,1], 1.0)
        self.assertAlmostEquals(a[0,2], 1.001)
        self.assertAlmostEquals(a[1,0], 0.001)
        self.assertAlmostEquals(a[1,1], 0.0)
        self.assertAlmostEquals(a[1,2], -0.001)
        
        self.assertAlmostEquals(f[0,0], 0.99)
        self.assertAlmostEquals(f[0,1], 1.0)
        self.assertAlmostEquals(f[0,2], 1.0)
        self.assertAlmostEquals(f[1,0], 0.001)
        self.assertAlmostEquals(f[1,1], 0.0)
        self.assertAlmostEquals(f[1,2], 0.0)
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
