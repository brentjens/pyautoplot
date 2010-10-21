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
        rgb_pr = color_from_angle(0.0)
        rgb_pi = color_from_angle(pi/2)
        rgb_nr = color_from_angle(pi)
        rgb_ni = color_from_angle(pi*1.5)

        self.assertEquals(len(rgb_pr.shape), 1)
        self.assertEquals(rgb_pr.shape[0], 3)

        self.assertAlmostEquals(rgb_pr[0], 1.0)
        self.assertAlmostEquals(rgb_pr[1], 0.0)
        self.assertAlmostEquals(rgb_pr[2], 0.0)
        
        self.assertAlmostEquals(rgb_pi[0], 0.0)
        self.assertAlmostEquals(rgb_pi[1], 1.0)
        self.assertAlmostEquals(rgb_pi[2], 0.0)

        self.assertAlmostEquals(rgb_nr[0], 1.0)
        self.assertAlmostEquals(rgb_nr[1], 0.0)
        self.assertAlmostEquals(rgb_nr[2], 0.75)

        self.assertAlmostEquals(rgb_ni[0], 0.0)
        self.assertAlmostEquals(rgb_ni[1], 0.75)
        self.assertAlmostEquals(rgb_ni[2], 0.75)
        pass

    def test_rgb_scale_palette(self):
        self.assertAlmostEquals(rgb_scale_palette(0.0, 0.75), 1.0)
        self.assertAlmostEquals(rgb_scale_palette(pi/2, 0.5), 1.0)
        self.assertAlmostEquals(rgb_scale_palette(pi, 0.25), 1.0)
        self.assertAlmostEquals(rgb_scale_palette(3*pi/2, 1.2), 1.0)

        one_eight=pi/4.0
        self.assertAlmostEquals(rgb_scale_palette(0.0+one_eight, 0.75), 2.5**0.8)
        self.assertAlmostEquals(rgb_scale_palette(pi/2+one_eight, 0.5), 2.0**0.8)
        self.assertAlmostEquals(rgb_scale_palette(pi+one_eight, 0.25), 1.5**0.8)
        self.assertAlmostEquals(rgb_scale_palette(3*pi/2+one_eight, 1.2), 3.4**0.8)
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
