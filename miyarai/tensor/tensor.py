from copy import deepcopy


class Tensor:

    def __init__(self, data):
        self._validate_shape(data)

        self.data = data
        self.shape = self._infer_shape(data)
        self.ndim = len(self.shape)
        self.size =self._calculate_size(self.shape)

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

    def __getitem__(self,index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __len__(self):
        return len(self.data)

    def clone(self):
        return Tensor(deepcopy(self.data))
    
    def __repr__(self):
        return f"Tensor({self.data})"