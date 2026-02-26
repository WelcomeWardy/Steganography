import matplotlib.pyplot as plt
import numpy as np

def show_images(input_S, input_C, decoded_S, encoded_C, n=4):

    plt.figure(figsize=(12,8))

    for i in range(n):

        plt.subplot(n,4,i*4+1)
        plt.imshow(input_C[i])
        plt.title("Cover")
        plt.axis("off")

        plt.subplot(n,4,i*4+2)
        plt.imshow(input_S[i])
        plt.title("Secret")
        plt.axis("off")

        plt.subplot(n,4,i*4+3)
        plt.imshow(encoded_C[i])
        plt.title("Encoded Cover")
        plt.axis("off")

        plt.subplot(n,4,i*4+4)
        plt.imshow(decoded_S[i])
        plt.title("Decoded Secret")
        plt.axis("off")

    plt.tight_layout()
    plt.show()