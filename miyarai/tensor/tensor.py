from copy import deepcopy


class Tensor:

    def __init__(self, data):
        self._validate_shape(data)

        self.data = data
        self.shape = self._infer_shape(data)
        self.ndim = len(self.shape)
        self.size = self._calculate_size(self.shape)
        self.dtype = self._infer_dtype(data)

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


    def _elementwise_operation(self, other, operation):
        self._check_same_shape(other)

        flat_self = self._flatten(self.data)
        flat_other = self._flatten(other.data)

        result = [
            operation(a, b)
            for a, b in zip(flat_self, flat_other)
        ]

        new_data = self._build_shape(result, self.shape)

        return Tensor(new_data)

    def _scalar_operation(self, data, scalar, operation):

        if isinstance (data, list):
            return[
                self._scalar_operation(item, scalar, operation)
                for item in data
            ]

        return operation(data, scalar)
         
    def __getitem__(self,index):
        return self.data[index]

    def __setitem__(self, index, value):
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

    def sum(self):
        total = 0
        values = self._flatten(self.data)
        for value in values:
            total += value

        return total
    

    def __repr__(self):
        return f"Tensor({self.data})"