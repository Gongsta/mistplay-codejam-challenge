import pandas as pd
import numpy as np

dataset = pd.read_csv('./book2.csv')

dataset = dataset.fillna(0)

X = dataset.drop(columns=[0, 'os_version'])