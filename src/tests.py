import pickle
import pandas as pd
import datetime


def get_data():
    data_file = open('data/data.pickle', 'rb')
    data = pickle.load(data_file)
    data_file.close()
    return data


d = get_data()

