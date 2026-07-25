import unittest

from miyarai import Tensor


class TestTensorAddition(unittest.TestCase):

    def test_add_1d_tensors(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])

        result = a + b

        self.assertEqual(
            result,
            Tensor([5, 7, 9])
        )


    def test_add_2d_tensors(self):
        a = Tensor([
            [1, 2],
            [3, 4]
        ])

        b = Tensor([
            [5, 6],
            [7, 8]
        ])

        result = a + b

        self.assertEqual(
            result,
            Tensor([
                [6, 8],
                [10, 12]
            ])
        )


    def test_add_floats(self):
        a = Tensor([1.5, 2.5])
        b = Tensor([0.5, 1.5])

        result = a + b

        self.assertEqual(
            result,
            Tensor([2.0, 4.0])
        )


    def test_add_negative_numbers(self):
        a = Tensor([-1, -2, 3])
        b = Tensor([5, -6, 7])

        result = a + b

        self.assertEqual(
            result,
            Tensor([4, -8, 10])
        )


    def test_original_tensors_unchanged(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])

        result = a + b

        self.assertEqual(a, Tensor([1, 2, 3]))
        self.assertEqual(b, Tensor([4, 5, 6]))
        self.assertEqual(result, Tensor([5, 7, 9]))


    def test_add_shape_mismatch(self):
        a = Tensor([1, 2])
        b = Tensor([1, 2, 3])

        with self.assertRaises(ValueError):
            a + b


    def test_add_dimension_mismatch(self):
        a = Tensor([[1, 2]])
        b = Tensor([1, 2])

        with self.assertRaises(ValueError):
            a + b


    def test_add_empty_tensors(self):
        a = Tensor([])
        b = Tensor([])

        result = a + b

        self.assertEqual(result, Tensor([]))


if __name__ == "__main__":
    unittest.main()