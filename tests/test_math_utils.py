import unittest
from utils.math_utils import get_factors

class TestMathUtils(unittest.TestCase):

    def test_get_factors(self):
        self.assertEqual(get_factors(1), [1])
        self.assertEqual(get_factors(2), [1, 2])
        self.assertEqual(get_factors(3), [1, 3])
        self.assertEqual(get_factors(16), [1, 2, 4, 8, 16])
        self.assertEqual(get_factors(28), [1, 2, 4, 7, 14, 28])
        self.assertEqual(get_factors(7), [1, 7])
        self.assertEqual(get_factors(12), [1, 2, 3, 4, 6, 12])
        self.assertEqual(get_factors(100), [1, 2, 4, 5, 10, 20, 25, 50, 100])

        # invalid cases
        self.assertRaises(ValueError, get_factors, 0)
        self.assertRaises(ValueError, get_factors, -10)

if __name__ == '__main__':
    unittest.main()