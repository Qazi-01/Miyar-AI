class Tensor:

    def __init__(self, data):
        self.data = data
        self.shape = self._infer_shape(data)
        self.ndim = len(self.shape)
        self.size =self._calculate_size(self.shape)

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

    def __repr__(self):
        return f"Tensor({self.data})"