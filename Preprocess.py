import tensorflow as tf
# import numpy as np
# import pandas as pd
# import os, shutil
# import matplotlib.pyplot as plt
# import seaborn as sns

Model = tf.keras.models.load_model('saved_model/my_model')
# print(Model.summary())
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def predict(x):
    result = Model.predict([x])
    return result

ff = [0.84168337, 0.35135135, 0.6       , 0.57994781, 0.        ,
       0.        , 0.        , 0.33491522, 0.66921399, 0.33078601,
       0.        , 0.33078601, 0.66921399]

def main():
    # print('I am groot')
    res = predict(ff)
    print(res)
if __name__ == '__main__':
    main()