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

class TestTensorScalarMultiplication(unittest.TestCase):
    def test_mul_scalar_int(self):
        a = Tensor([2,3,4])
        res = a * 5

        self.assertEqual(
            res,
            Tensor([10,15,20])
        )

    def test_mul_scalar_float(self):
        a = Tensor([1.5, 2.5])
        res = a * 2.0

        self.assertEqual(
            res,
            Tensor([3.0,5.0])
        )

    def test_mul_scalar_2d(self):
        a = Tensor([
            [1,2],
            [3,4]
        ])
        res = a * 10

        self.assertEqual(
            res,
            Tensor([
                [10,20],
                [30,40]
            ])
        )

    def test_mul_scalar_negative_numbers(self):
        a = Tensor([-2,5])
        res = a * 3

        self.assertEqual(
            res,
            Tensor([-6,15])
        )

    def test_mul_scalar_zero(self):
        a = Tensor ([5,6,7])
        self.assertEqual(
            a * 0,
            Tensor([0,0,0])
        )

    def test_mul_scalar_empty_tensor(self):
        a = Tensor([])
        self.assertEqual(
            a * 5,
            Tensor([])
        )

    def test_original_tensor_unchanged_after_scalar_mul(self):
        a = Tensor([2,3,4])
        res = a * 5

        self.assertEqual(a, Tensor([2,3,4]))
        self.assertEqual(res, Tensor([10,15,20]))

    def test_tensor_multiplication_still_works(self):
        a = Tensor([2,3,4])
        b = Tensor([5,6,7])

        self.assertEqual(
            a*b,
            Tensor([10,18,28])
        )

class TestTensorScalarDivision(unittest.TestCase):

    def test_div_scalar_int(self):
        a = Tensor([10,20,30])
        res = a / 10

        self.assertEqual(
            res,
            Tensor([1.0,2.0,3.0])
        )

    def test_div_scalar_float(self):
        a = Tensor([3.0,6.0])
        res = a / 1.5

        self.assertEqual(
            res,
            Tensor([2.0,4.0])
        )

    def test_div_scalar_2d(self):
        a = Tensor([
            [20,40],
            [60,80]
        ])
        res = a / 20

        self.assertEqual(
            res,
            Tensor([
                [1.0,2.0],
                [3.0,4.0]
            ])
        )

    def test_div_scalar_negative_values(self):
        a = Tensor([-10,20])
        res = a / 5

        self.assertEqual(
            res,
            Tensor([-2.0, 4.0])
        )

    def test_div_scalar_empty_tensor(self):
        a = Tensor([])

        self.assertEqual(
            a / 5,
            Tensor([])
        )

    def test_div_scalar_by_zero(self):
        a = Tensor([1,2,3])

        with self.assertRaises(ZeroDivisionError):
            a / 0

    def test_original_tensor_unchanged_after_scalar_div(self):
       a = Tensor([10,20,30])

       res = a / 10

       self.assertEqual(a, Tensor([10,20,30]))
       self.assertEqual(res, Tensor([1.0,2.0,3.0]))

    def test_tensor_division_still_works(self):
        a = Tensor ([10,18,28])
        b = Tensor ([2,3,4])

        self.assertEqual(
            a / b,
            Tensor([5.0,6.0,7.0])
        )

class TestTensorSum(unittest.TestCase):

    def test_sum_1d_tensor(self):
        a = Tensor([1,2,3,4])

        self.assertEqual(
            a.sum(),
            10
        )

    def test_sum_2d_tensor(self):
        a = Tensor([
            [1,2],
            [3,4]
        ])

        self.assertEqual(
            a.sum(),
            10
        )

    def test_sum_sloat_tensor(self):
        a = Tensor([1.5,2.5,3.0])

        self.assertEqual(
            a.sum(),
            7.0
        )

    def test_sum_negative_value(self):
        a = Tensor([-5,2,3])

        self.assertEqual(
            a.sum(),
            0
        )

    def test_sum_empty_tensor(self):
        a = Tensor([])

        self.assertEqual(
            a.sum(),
            0
        )

    def test_sum_does_not_modify_tensor(self):
        a = Tensor([1,2,3])
        res = a.sum()

        self.assertEqual(
            a,
            Tensor([1,2,3])
        )
        self.assertEqual(
            res,
            6
        )

class TestTensorMean(unittest.TestCase):

    def test_mean_1d_tensor(self):
        a = Tensor([2,4,6])

        self.assertEqual(
            a.mean(),
            4
        )

    def test_mean_float_tensor(self):
        a = Tensor([1.5,2.5,3.5])

        self.assertEqual(
            a.mean(),
            2.5
        )

    def test_mean_2d_tensor(self):
        a = Tensor([
            [1,2],
            [3,4]
        ])

        self.assertEqual(
            a.mean(),
            2.5
        )

    def test_mean_negative_value(self):
        a = Tensor([-2,4,6])

        self.assertEqual(
            a.mean(),
            2.6666666666666665
        )

    def test_mean_empty_tensor(self):
        a = Tensor([])

        self.assertEqual(
            a.mean(),
            0
        )

    def test_mean_does_not_modify_tensor(self):
        a = Tensor([1,2,3])
        res = a.mean()

        self.assertEqual(a, Tensor([1,2,3]))
        self.assertEqual(res, 2.0)

class TestTensorMax(unittest.TestCase):

    def test_max_1d_tensor(self):
        a = Tensor([1,5,3])
        self.assertEqual(
            a.max(),
            5
        )

    def test_max_2d_tensor(self):
        a = Tensor([
            [1,8],
            [3,4]
        ])
        self.assertEqual(
            a.max(),
            8
        )

    def test_max_float_tensor(self):
        a = Tensor([1.5,7.2,3.8])
        self.assertEqual(
            a.max(),
            7.2
        )

    def test_max_negative_values(self):
        a = Tensor([-10,-3,-7])
        self.assertEqual(
            a.max(),
            -3
        )

    def test_max_empty_tensor(self):
        a = Tensor([])
        self.assertEqual(
            a.max(),
            None
        )

    def test_max_does_not_modify_original_tensor(self):
        a = Tensor([1,5,3])
        res = a.max()

        self.assertEqual(a, Tensor([1,5,3]))
        self.assertEqual(res, 5)

class TestTensorMin(unittest.TestCase):

    def test_min_1d_tensor(self):
        a = Tensor([5,2,8])
        self.assertEqual(
            a.min(),
            2
        )

    def test_min_2d_tensor(self):
        a = Tensor([
            [5,1],
            [8,3]
        ])
        self.assertEqual(
            a.min(),
            1
        )

    def test_min_float_tensor(self):
        a = Tensor([3.5,1.2,7.8])
        self.assertEqual(
            a.min(),
            1.2
        )

    def test_min_negative_values(self):
        a = Tensor([-5,-2,-9])
        self.assertEqual(
            a.min(),
            -9
        )

    def test_min_empty_tensor(self):
        a = Tensor([])
        self.assertEqual(
            a.min(),
            None
        )

    def test_min_does_not_modify_original_tensor(self):
        a = Tensor([5,2,8])
        res = a.min()

        self.assertEqual(a, Tensor([5,2,8]))
        self.assertEqual(res, 2)

class TestTensorMatMul(unittest.TestCase):

    def test_matmul_2x2(self):
        a = Tensor([
            [1,2],
            [3,4]
        ])

        b = Tensor([
            [5,6],
            [7,8]
        ])

        self.assertEqual(
            a@b, 
            Tensor([
                [19,22],
                [43,50]
            ])
        )

    def test_matmul_2x3_3x2(self):
        a = Tensor([
            [1,2,3],
            [4,5,6]
        ])

        b = Tensor([
            [7,8],
            [9,10],
            [11,12]
        ])

        self.assertEqual(
            a@b,
            Tensor([
                [58,64],
                [139,154]
            ])
        )

    def test_matmul_identity(self):
        a = Tensor([
            [1,2],
            [3,4]
        ])

        identity = Tensor([
            [1,0],
            [0,1]
        ])

        self.assertEqual(
            a@ identity,
            Tensor([
                [1,2],
                [3,4]
            ])
        )

    def test_matmul_invalid_dimensions(self):
        a = Tensor([1,2,3])
        b = Tensor([4,5,6])

        with self.assertRaises(ValueError):
            a @ b

    def test_matmul_shape_mismatch(self):
        a = Tensor([
            [1,2]
        ])
        b = Tensor([
            [1,2]
        ])

        with self.assertRaises(ValueError):
            a @ b

    def test_matmul_does_not_modify_operands(self):
        a = Tensor([
            [1,2],
            [3,4]
        ])
        b = Tensor([
            [5,6],
            [7,8]
        ])

        res = a @ b

        self.assertEqual(
            a,
            Tensor([
                [1,2],
                [3,4]
            ])
        )
        self.assertEqual(
            b,
            Tensor([
                [5,6],
                [7,8]
            ])
        )
        self.assertEqual(
            res,
            Tensor([
                [19,22],
                [43,50]
            ])
        )

