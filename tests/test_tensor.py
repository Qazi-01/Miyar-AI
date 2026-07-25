import unittest
from miyarai import Tensor

class TestTensor(unittest.TestCase):
    def test_shape(self):
        tensor = Tensor([
            [1,2],
            [3,4]
        ])

        self.assertEqual(tensor.shape, (2,2))

    def test_ndim(self):
        tensor = Tensor([
            [1,2],
            [3,4]
        ])

        self.assertEqual(tensor.ndim, 2)

    def test_size(self):
        tensor = Tensor([
            [1,2],
            [3,4]
        ])

        self.assertEqual(tensor.size, 4)

    def test_dtype(self):
        tensor = Tensor([
            [1,2,3]
        ])

        self.assertEqual(tensor.dtype, "int")

    def test_clone(self):
        a = Tensor([1,2,3])
        b = a.clone()

        b[0] = 999

        self.assertEqual(a.tolist(), [1,2,3])
        self.assertEqual(b.tolist(), [999,2,3])

    def test_reshape(self):
        tensor = Tensor([1,2,3,4])
        reshaped = tensor.reshape((2,2))

        self.assertEqual(
            reshaped.tolist(),
            [[1,2],[3,4]]
        )

    def test_transpose(self):
        tensor = Tensor([
            [1,2],
            [3,4]
        ])

        transposed = tensor.transpose()

        self.assertEqual(
            transposed.tolist(),
            [
                [1,3],
                [2,4]
            ]
        )

if __name__ == "__main__":
    unittest.main()