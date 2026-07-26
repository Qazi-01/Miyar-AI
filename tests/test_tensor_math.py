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

#res means result

class TestTensorSubtraction(unittest.TestCase):

    def test_sub_1d_tensors(self):
        a = Tensor([5,7,9])
        b = Tensor([1,2,3])

        res = a - b 
        self.assertEqual(
            res,
            Tensor([4,5,6])
            )

    def test_sub_2d_tensors(self):
        a = Tensor ([
            [10,20],
            [30,40]
        ])
        b = Tensor([
            [1,2],
            [3,4]
        ])

        res = a - b
        self.assertEqual(
            res,
            Tensor([
                [9,18],
                [27,36]
            ])
        )

    def test_sub_floats(self):
        a = Tensor([2.5, 5.5])
        b = Tensor([0.5,1.5])
        res = a - b 

        self.assertEqual(
            res,
            Tensor([2.0,4.0])
        )

    def test_sub_negative_numbers(self):
        a = Tensor([-5,2,10])
        b = Tensor([-2,4,3])
        res = a - b 

        self.assertEqual(
            res,
            Tensor([-3,-2,7])
        )

    def test_original_tensors_unchange_after_sub(self):
        a = Tensor ([5,7,9])
        b = Tensor ([1,2,3])
        res = a - b

        self.assertEqual(a ,Tensor([5,7,9]))
        self.assertEqual(b ,Tensor([1,2,3]))
        self.assertEqual(res ,Tensor([4,5,6]))

    def test_sub_shape_mismatch(self):
        a = Tensor ([1,2])
        b = Tensor ([1,2,3])

        with self.assertRaises(ValueError):
            a - b

class TestTensorMultiplication(unittest.TestCase):

    def test_mul_1d_tensors(self):
        a = Tensor([2,3,4])
        b = Tensor([5,6,7])
        res = a * b 

        self.assertEqual(
            res,
            Tensor([10,18,28])
        )

    def test_mul_2d_tensors(self):
        a = Tensor([
            [1,2],
            [3,4]
        ])

        b = Tensor([
            [5,6],
            [7,8]
        ])

        res = a * b

        self.assertEqual(
            res,
            Tensor([
                [5,12],
                [21,32]
            ])
        )

    def test_mul_floats(self):
        a = Tensor([1.5,2.5])
        b = Tensor([2.0,4.0])
        res = a * b

        self.assertEqual(
            res,
            Tensor([3.0,10.0])
        )

    def test_mul_negative_numbers(self):
        a = Tensor([-2,3,-4])
        b = Tensor([5,-6,7])
        res = a * b

        self.assertEqual(
            res,
            Tensor([-10,-18,-28])
        )

    def test_original_tensors_unchanged_after_mul(self):
        a = Tensor([2,3,4])
        b = Tensor([5,6,7])
        res = a * b

        self.assertEqual(a, Tensor([2,3,4]))
        self.assertEqual(b, Tensor([5,6,7]))
        self.assertEqual(res, Tensor([10,18,28]))

    def test_mul_shape_mismatch(self):
        a = Tensor([1,2])
        b = Tensor([1,2,3])

        with self.assertRaises(ValueError):
            a * b

    def test_mul_empty_tensors(self):
        a = Tensor([])
        b = Tensor([])

        self.assertEqual(a * b, Tensor([]))

class TestTensorDivision(unittest.TestCase):

    def test_div_1d_tensors(self):
        a = Tensor([10,18,28])
        b = Tensor([2,3,4])
        res = a / b

        self.assertEqual(
            res,
            Tensor([5.0,6.0,7.0])
        )

    def test_div_2d_tensors(self):
        a = Tensor([
            [10,20],
            [30,40]
        ])
        b = Tensor([
            [2,4],
            [5,8]
        ])

        res = a / b

        self.assertEqual(
            res,
            Tensor([
                [5.0,5.0],
                [6.0,5.0]
            ])
        )

    def test_div_floats(self):
        a = Tensor([3.0,10.0])
        b = Tensor([1.5,2.5])
        res = a / b

        self.assertEqual(
            res,
            Tensor([2.0,4.0])
        )

    def test_div_negative_numbers(self):
        a = Tensor([-10,-18,28])
        b = Tensor([5,-6,7])
        res = a / b

        self.assertEqual(
            res,
            Tensor([-2.0,3.0,4.0])
        )

    def test_original_tensors_unchange_after_div(self):
        a = Tensor([10,18,28])
        b = Tensor([2,3,4])
        res = a / b

        self.assertEqual(a, Tensor([10,18,28]))
        self.assertEqual(b, Tensor([2,3,4]))
        self.assertEqual(res, Tensor([5.0,6.0,7.0]))

    def test_div_shape_mismatch(self):
        a = Tensor([1,2])
        b = Tensor([1,2,3])

        with self.assertRaises(ValueError):
            a / b

    def test_div_by_zero(self):
        a = Tensor([1,2,3])
        b = Tensor([1,0,1])

        with self.assertRaises(ZeroDivisionError):
            a / b 

    def test_div_empty_tensors(self):
        a = Tensor([])
        b = Tensor([])

        self.assertEqual(a / b, Tensor([]))

class TestTensorScalarAddition(unittest.TestCase):

    def test_add_scalar_int(self):
        a = Tensor([1,2,3])
        res = a + 5

        self.assertEqual(
            res,
            Tensor([6,7,8])
        )

    def test_add_scalar_float(self):
        a = Tensor([1.5, 2.5])
        res = a + 2.0

        self.assertEqual(
            res,
            Tensor([3.5,4.5])
        )

    def test_add_scalar_2d(self):
        a = Tensor([
            [1,2],
            [3,4]
        ])
        res = a + 10

        self.assertEqual(
            res,
            Tensor([
                [11,12],
                [13,14]
            ])
        )

    def test_add_scalar_negative_values(self):
        a = Tensor([-2,5])
        res = a + 3

        self.assertEqual(
            res,
            Tensor([1,8])
        )

    def test_add_scalar_empty_tensor(self):
        a = Tensor([])

        self.assertEqual(
            a + 5,
            Tensor([])
        )

    def test_original_tensor_unchanged_after_scalar_add(self):
        a = Tensor([1,2,3])
        res = a + 5

        self.assertEqual(a, Tensor([1,2,3]))
        self.assertEqual(res, Tensor([6,7,8]))

    def test_tensor_addition_still_works(self):
        a = Tensor([1,2,3])
        b = Tensor([4,5,6])

        self.assertEqual(
            a + b,
            Tensor([5,7,9])
        )

class TestTensorScalarSubtraction(unittest.TestCase):

    def test_sub_scalar_int(self):
        a = Tensor([5,6,7])
        res = a - 2

        self.assertEqual(
            res,
            Tensor([3,4,5])
        )

    def test_sub_scalar_float(self):
        a = Tensor([3.5,4.5])
        res = a - 1.5

        self.assertEqual(
            res,
            Tensor([2.0,3.0])
        )

    def test_sub_scalar_2d(self):
        a = Tensor([
            [10,20],
            [30,40]
        ])
        res = a - 5

        self.assertEqual(
            res,
            Tensor([
                [5,15],
                [25,35]
            ])
        )

    def test_sub_scalar_negative_values(self):
        a = Tensor([-2,5])
        res = a - 3

        self.assertEqual(
            res,
            Tensor([-5,2])
        )

    def test_sub_scalar_empty_tensor(self):
        a = Tensor([])
        self.assertEqual(
            a - 5,
            Tensor([])
        )

    def test_original_tensor_unchanged_after_scalar_sub(self):
        a = Tensor([5,6,7])
        res = a - 2

        self.assertEqual(a, Tensor([5,6,7]))
        self.assertEqual(res, Tensor([3,4,5]))

    def test_tensor_subtraction_still_works(self):
        a = Tensor([5,7,9])
        b = Tensor([1,2,3])

        self.assertEqual(
            a - b,
            Tensor([4,5,6])
        )

