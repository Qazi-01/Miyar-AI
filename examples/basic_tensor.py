from miyarai import Tensor

def main():
    x = Tensor([
        [1,2],
        [3,4]
    ])

    print("Tensor:")
    print(x)

    print("\nShape:", x.shape)
    print("Dimensions:", x.ndim)
    print("Size:", x.size)
    print("Data Type:", x.dtype)

if __name__ == "__main__":
    main()