import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

def pixel_errors(input_S, input_C, decoded_S, encoded_C):

    diff_S = np.sqrt(np.mean(np.square(255*(input_S - decoded_S))))
    diff_C = np.sqrt(np.mean(np.square(255*(input_C - encoded_C))))

    return diff_S, diff_C


def plot_histogram(diff_S, diff_C):

    sns.kdeplot(diff_S.flatten(), shade=True)
    plt.title("Distribution of errors in Secret Image")
    plt.show()

    sns.kdeplot(diff_C.flatten(), shade=True)
    plt.title("Distribution of errors in Cover Image")
    plt.show()