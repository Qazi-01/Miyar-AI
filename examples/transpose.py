from miyarai import Tensor

def main():
    tensor = Tensor([
        [1,2],
        [3,4]
    ])

    print("Original Tensor")
    print(tensor)

    transposed = tensor.transpose()

    print("\nTransposed Tensor:")
    print(transposed)

if __name__ == "__main__":
    main()