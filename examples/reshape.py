from miyarai import Tensor

def main():
    tensor = Tensor ([
        [1,2],
        [3,4]
    ])

    print("Original Tensor:")
    print(tensor)
    print("Shape:", tensor.shape)

    reshaped = tensor.reshape((4,))

    print("\nReshpaed Tensor:")
    print(reshaped)
    print("shape:", reshaped.shape)

if __name__ == "__main__":
    main()