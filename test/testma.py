import unittest
import pyautoplot.ma as ma
import os


class TestMaskedArrays (unittest.TestCase):
    def test_module_type(self):
        self.assertEquals(type(ma), type(os))
        pass

    def test_module_contents(self):
        expected_symbols=['abs', 'absolute', 'add', 'all', 'allclose',
    'allequal', 'alltrue', 'amax', 'amin', 'anom', 'anomalies', 'any',
    'apply_along_axis', 'arange', 'arccos', 'arccosh', 'arcsin',
    'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin',
    'argsort', 'around', 'array', 'asanyarray', 'asarray',
    'atleast_1d', 'atleast_2d', 'atleast_3d', 'average',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bool_', 'ceil',
    'choose', 'clip', 'clump_masked', 'clump_unmasked',
    'column_stack', 'common_fill_value', 'compress', 'compress_cols',
    'compress_rowcols', 'compress_rows', 'compressed', 'concatenate',
    'conjugate', 'copy', 'core', 'corrcoef', 'cos', 'cosh', 'count',
    'count_masked', 'cov', 'cumprod', 'cumsum', 'default_fill_value',
    'diag', 'diagflat', 'diagonal', 'diff', 'divide', 'dot', 'dstack',
    'dump', 'dumps', 'ediff1d', 'empty', 'empty_like', 'equal', 'exp',
    'expand_dims', 'extras', 'fabs', 'filled', 'fix_invalid',
    'flatnotmasked_contiguous', 'flatnotmasked_edges', 'flatten_mask',
    'flatten_structured_array', 'floor', 'floor_divide', 'fmod',
    'frombuffer', 'fromflex', 'fromfunction', 'getdata', 'getmask',
    'getmaskarray', 'greater', 'greater_equal', 'harden_mask',
    'hsplit', 'hstack', 'hypot', 'identity', 'ids', 'in1d', 'indices',
    'inner', 'innerproduct', 'intersect1d', 'intersect1d_nu', 'isMA',
    'isMaskedArray', 'is_mask', 'is_masked', 'isarray', 'left_shift',
    'less', 'less_equal', 'load', 'loads', 'log', 'log10',
    'logical_and', 'logical_not', 'logical_or', 'logical_xor',
    'make_mask', 'make_mask_descr', 'make_mask_none', 'mask_cols',
    'mask_or', 'mask_rowcols', 'mask_rows', 'masked', 'masked_all',
    'masked_all_like', 'masked_array', 'masked_equal',
    'masked_greater', 'masked_greater_equal', 'masked_inside',
    'masked_invalid', 'masked_less', 'masked_less_equal',
    'masked_not_equal', 'masked_object', 'masked_outside',
    'masked_print_option', 'masked_singleton', 'masked_values',
    'masked_where', 'max', 'maximum', 'maximum_fill_value', 'mean',
    'median', 'min', 'minimum', 'minimum_fill_value', 'mod', 'mr_',
    'multiply', 'negative', 'nomask', 'nonzero', 'not_equal',
    'notmasked_contiguous', 'notmasked_edges', 'ones', 'outer',
    'outerproduct', 'polyfit', 'power', 'prod', 'product', 'ptp',
    'put', 'putmask', 'rank', 'ravel', 'remainder', 'repeat',
    'reshape', 'resize', 'right_shift', 'round', 'round_',
    'row_stack', 'set_fill_value', 'setdiff1d', 'setmember1d',
    'setxor1d', 'shape', 'sin', 'sinh', 'size', 'soften_mask',
    'sometrue', 'sort', 'sqrt', 'squeeze', 'std', 'subtract', 'sum',
    'swapaxes', 'take', 'tan', 'tanh', 'trace', 'transpose',
    'true_divide', 'union1d', 'unique', 'unique1d', 'vander', 'var',
    'vstack', 'where', 'zeros']

        for s in expected_symbols:
            self.assertTrue(s in dir(ma)) 
    pass

#
#  M A I N 
#

if __name__ == '__main__':
    unittest.main()

#
#  E O F
#
