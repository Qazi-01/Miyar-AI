from copy import deepcopy


class Tensor:

    def __init__(self, data, dtype=None, requires_grad=False):
        self._validate_shape(data)

        self.data = data
        self.shape = self._infer_shape(data)
        self.ndim = len(self.shape)
        self.size = self._calculate_size(self.shape)
        self.dtype = self._infer_dtype(data)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self.parents = ()


    def __repr__(self):
        return f"Tensor({self.data})"

    def _validate_shape(self, data):

        if not isinstance(data, list):
            return
        if len(data) == 0:
            return

        # Either all elements are list, or none are.
        are_lists = [isinstance(item,list) for item in data]

        if any(are_lists) and not all(are_lists):
            raise ValueError("inconsistent tensor shape")

        if all(are_lists):
            expected_length = len(data[0])

            for item in data:
                if len(item) != expected_length:
                    raise ValueError("Inconsistent tensor shape")

                self._validate_shape(item)


    def _infer_shape(self,data):
        shape = []

        while isinstance(data,list):
            shape.append(len(data))

            if len(data) == 0:
                break
            data = data[0]
        return tuple(shape)
    
    def _calculate_size(self,shape):

        size = 1

        for dimension in shape:
            size *= dimension

        return size

    def _infer_dtype(self,data):
        values = []

        def collect(obj):
            if isinstance(obj, list):
                for item in obj:
                    collect(item)
            else:
                values.append(type(obj))

        collect(data)

        if not values:
            return None 

        first = values[0]

        for value in values:
            if value != first:
                raise TypeError("Tensor element must have same data type.")

        return first.__name__

    def _flatten(self, data):

        result = []
        def flatten(item):
            if isinstance(item, list):
                for value in item:
                    flatten(value)
            else:
                result.append(item)

        flatten(data)

        return result

    def _build_shape(self, flat, shape):

        flat = flat.copy()

        def build(current_shape):
            if len(current_shape) == 1:
                result = flat[:current_shape[0]]
                del flat[:current_shape[0]]
                return result 
            return[
                build(current_shape[1:])
                for _ in range(current_shape[0])
            ]
        return build(shape)

    def _check_same_shape(self, other):
        if self.shape != other.shape:
            raise ValueError(
                f"Cannot operate on tensors with shapes {self.shape} and {other.shape}."
            )

    def _broadcast_shape(self, shape1, shape2):
        shape1 = (1,) * (len(shape2) - len(shape1)) + shape1
        shape2 = (1,) * (len(shape1) - len(shape2)) + shape2

        result = []

        for dim1, dim2 in zip(shape1,shape2):
            if dim1 == dim2:
                result.append(dim1)
            elif dim1 == 1:
                result.append(dim2)
            elif dim2 == 1:
                result.append(dim1)
            else:
                raise ValueError(
                    f"Cannot broadcast shapes {shape1} and {shape2}."
                )
        return tuple(result)

    def _expand_to_shape(self,data,current_shape,target_shape):

        if current_shape == target_shape:
            return data

        # prepend leading dimension of size 1
        while len(current_shape) < len(target_shape):
            data = [data]
            current_shape = (1,) + current_shape

        def expand(d, current, target):
            if len(current) == 0:
                return d

            if current[0] == target[0]:
                return [ 
                    expand(item, current[1:], target[1:])
                    for item in d
                ]

            if current[0] == 1:
                expanded = expand(d[0], current[1:], target[1:])
                return [
                    expanded
                    for _ in range(target[0])
                ]

            raise ValueError(
                f"Cannot expand shape {current_shape} to {target_shape}."
            )
        return expand(data, current_shape, target_shape)


    def _elementwise_operation(self, other, operation):
        target_shape = self._broadcast_shape(
            self.shape,
            other.shape
        )

        left = self._expand_to_shape(
            self.data,
            self.shape,
            target_shape
        )

        right = self._expand_to_shape(
            other.data,
            other.shape,
            target_shape
        )

        flat_left = self._flatten(left)
        flat_right = self._flatten(right)

        result = [
            operation(a,b)
            for a,b in zip(flat_left, flat_right)
        ]

        new_data = self._build_shape(
            result,
            target_shape
        )

        return Tensor(new_data)
    

    def _scalar_operation(self, data, scalar, operation):

        if isinstance (data, list):
            return[
                self._scalar_operation(item, scalar, operation)
                for item in data
            ]

        return operation(data, scalar)

    def _tuple_index(self, data, indices):

        if len(indices) == 0:
            return data

        if not isinstance(data, list):
            raise IndexError("Too many indices for tensor.")
        index = indices[0]

        if not isinstance(index, int):
            raise TypeError(
                "Tuple indexing currently supports only integers."
            )

        return self._tuple_index(
            data[index],
            indices[1:]
        )

         
    def __getitem__(self,index):
        if isinstance(index, tuple):
           result = self._tuple_index(
               self.data,
               index
           )

        elif isinstance(index, int):
           result = self.data[index]

        elif isinstance(index, slice):
            result = self.data[index]

        elif isinstance(index, list):

            if not all(isinstance(i, int) for i in index):
                raise TypeError(
                    "Advance indexing currently only supports only integer lists."
                )

            result = [
                self.data[i]
                for i in index
            ]

        else:
            raise TypeError(
                'Tensor indices must be integers, slices, tuples, or integer lists.'
            )
        
        if isinstance(result, list):
            return Tensor(result)

        return result
    
    
    def __setitem__(self, index, value):
        if not isinstance(index, int):
            raise TypeError(
                "Tensor indices must be integers."
            )
        if isinstance(value, Tensor):
            value = value.data

        self.data[index] = value
    
    
    def __len__(self):
        return len(self.data)

    def clone(self):
        return Tensor(deepcopy(self.data))

    def tolist(self):
        return deepcopy(self.data)

    def reshape(self, shape):

        new_size = self._calculate_size(shape)

        if new_size != self.size:
            raise ValueError(
                f"Cannot reshape tensor size {self.size} into shape {shape}."
            )

        flat = self._flatten(self.data)
        new_data = self._build_shape(flat, shape)

        return Tensor(new_data)

    def transpose(self):

        if self.ndim != 2:
            raise ValueError("Transpose is only supported for 2d tensors ")

        rows = self.shape[0]
        cols = self.shape[1]

        transposed = []
        
        for col in range(cols):
            new_row = []
            for row in range(rows):
                new_row.append(self.data[row][col])

            transposed.append(new_row)

        return Tensor(transposed)
    
    def __iter__(self):
        return iter(self.data)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False

        return self.data == other.data

    def __add__(self, other):
        if isinstance(other, Tensor):
            return self._elementwise_operation(
                other,
                lambda a, b: a + b
            )

        if isinstance(other,(int, float)):
            new_data = self._scalar_operation(
                self.data,
                other,
                lambda a, b: a + b
            )

            return Tensor(new_data)

        return NotImplemented
        
    def __sub__(self, other):
        if isinstance(other, Tensor):
        
            return self._elementwise_operation(
                other,
                lambda a, b: a - b
            )

        if isinstance(other,(int, float)):
            new_data = self._scalar_operation(
                self.data,
                other,
                lambda a, b: a - b
            )

            return Tensor(new_data)
        
        return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return self._elementwise_operation(
                other,
                lambda a, b: a*b
            )

        if isinstance(other,(int,float)):
            new_data = self._scalar_operation(
                self.data,
                other,
                lambda a, b: a*b
            )

            return Tensor(new_data)
        return NotImplemented
    
    def __truediv__(self, other):

        if isinstance(other, Tensor):
            return self._elementwise_operation(
                other,
                lambda a, b: a / b
            )

        if isinstance(other,(int,float)):
            new_data = self._scalar_operation(
                self.data,
                other,
                lambda a, b: a / b
            )

            return Tensor(new_data)
        return NotImplemented

    def flatten(self):
        return Tensor(
            self._flatten(self.data)
        )

    def squeeze(self):
        #return copy of tensor with all dimension of size 1 removed

        def squeeze_recursive(data):
            while isinstance (data,list) and len(data) == 1:
                data = data[0]

            if isinstance(data, list):
                return[
                    squeeze_recursive(item)
                    for item in data
                ]
            return data
        return Tensor(
            squeeze_recursive(self.data)
        )

    def unsqueeze(self, axis):

        if axis < -(self.ndim + 1):
            raise ValueError("Axis out of range")

        if axis < 0:
            axis += self.ndim + 1

        if axis > self.ndim:
            raise ValueError("Axis out of range")
    
        def insert_axis(data, current_axis):
            if current_axis == 0:
                return [data]

            if isinstance(data, list):
                return [
                    insert_axis(item, current_axis - 1)
                    for item in data
                ]

            return [data]
        new_data = insert_axis(self.data, axis)

        return Tensor(new_data)


    def repeat(self, repeats):

        if not isinstance(repeats, int):
            raise TypeError("Repeats must be an integer.")

        if repeats <= 0:
            raise ValueError("Repeats must be getter than zero.")

        def repeat_recursive(data):
            if isinstance(data, list):
                return [
                    item
                    for element in data
                    for item in(
                        [repeat_recursive(element)] * repeats
                        if not isinstance(element, list)
                        else [repeat_recursive(element)]
                    )
                ]
            return data
        new_data = repeat_recursive(self.data)

        return Tensor(new_data)

    def tile(self, repeats):
        if not isinstance(repeats, int):
            raise TypeError("Repeats must be integer.")
        if repeats <= 0:
            raise ValueError("Repeats must be greater than zero.")

        def deep_copy(data):
            if isinstance(data,list):
                return [deep_copy(item) for item in data]
            return data

        if isinstance(self.data, list):
            new_data = []

            for i in range(repeats):
                copied = deep_copy(self.data)

                if self.ndim == 1:
                    new_data.extend(copied)
                else:
                    new_data.extend(copied)

        else:
            new_data = [self.data for i in range(repeats)]

        return Tensor(new_data)

    @staticmethod
    def stack(tensors, axis=0):

        if not tensors:
            raise ValueError("Cannot stack an empty list of tensors")

        if not all(isinstance(tensor,Tensor) for tensor in tensors):
            raise TypeError("All elements must be tensor object.")

        reference_shape = tensors[0].shape

        for tensor in tensors:
            if tensor.shape != reference_shape:
                raise ValueError("All tensors must have the same shape.")

        if axis != 0:
            raise NotImplementedError(
                "stack() currently supports only axis = 0"
            )

        new_data = [tensor.data for tensor in tensors]
        return Tensor(new_data)

    @staticmethod
    def concat(tensors, axis=0):

        if not tensors:
            raise ValueError("Cannot concatenate an empty list of tensors.")

        if not all(isinstance(tensor, Tensor)for tensor in tensors):
            raise TypeError("All elements must be Tensor objects")

        reference_shape = tensors[0].shape
        ndim = len(reference_shape)

        if axis < -ndim or axis >= ndim:
            raise ValueError("Axis out of range.")

        if axis < 0:
            axis += ndim 

        for tensor in tensors:
            if len(tensor.shape) != ndim:
                raise ValueError("All tensors must have same number of dimensions.")

            for i in range(ndim):
                if i != axis and tensor.shape[i] != reference_shape[i]:
                    raise ValueError(
                        "All tensors must have the same shape except along the concatenation axis."
                    )
        def concat_recursive(data_list, current_axis):
            if current_axis == 0:
                result = []
                for data in data_list:
                    result.extend(data)
                return result

            return [
                concat_recursive(
                    [data[i] for data in data_list],
                    current_axis - 1
                )
                for i in range(len(data_list[0]))
            ]
        new_data = concat_recursive(
            [tensor.data for tensor in tensors],
            axis
        )

        return Tensor(new_data)

    @staticmethod
    def split(tensor, sections, axis=0):

        if not isinstance(tensor, Tensor):
            raise TypeError("Input must be a Tensor.")

        if sections <= 0:
            raise ValueError("Sections must be greater than zero.")

        ndim = tensor.ndim

        if axis < -ndim or axis >= ndim:
            raise ValueError("Axis out of range.")

        if axis < 0:
            axis += ndim

        axis_size = tensor.shape[axis]

        if axis_size % sections != 0:
            raise ValueError(
                "Tensor cannot be evenly split into the requested number of sections."
            )
        
        chunk_size = axis_size // sections

        def split_recursive(data, current_axis):
            if current_axis == 0:
                return [
                    data[i:i + chunk_size]
                    for i in range(0, len(data), chunk_size)
                ]

            child_split = [
                split_recursive(item, current_axis - 1)
                for item in data
            ]

            result = []

            for section in range(sections):
                result.append([
                    child[section]
                    for child in child_split
                ])

            return result

        split_data = split_recursive(tensor.data, axis)
        return [Tensor(part) for part in split_data]



    def __matmul__(self, other):

        if not isinstance(other, Tensor):
            return NotImplemented

        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(
                "Matrix multiplication is only supported for 2D tensors."
            )

        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Cannot multiply matrices with shapes {self.shape} and {other.shape}."
            )

        rows = self.shape[0]
        cols = other.shape[1]
        common = self.shape[1]

        result = []

        for i in range(rows):
            new_row = []

            for j in range(cols):
                value = 0

                for k in range(common):
                    value += (
                        self.data[i][k]
                        * other.data[k][j]
                    )

                new_row.append(value)
            result.append(new_row)        
        return Tensor(result)
    

    def sum(self):
        total = 0
        values = self._flatten(self.data)
        for value in values:
            total += value

        return total

    def mean(self):
        if self.size == 0:
            return 0

        return self.sum() / self.size

    def max(self):
        if self.size == 0:
            return None

        flat = self._flatten(self.data)
        return max(flat)

    def min(self):
        if self.size == 0:
            return None

        flat = self._flatten(self.data)
        return min(flat)
    
    