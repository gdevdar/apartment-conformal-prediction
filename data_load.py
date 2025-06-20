def data_load():
    import pandas as pd
    import os

    X_train = pd.read_json("model_dataset/X_train.json")
    X_test = pd.read_json("model_dataset/X_test.json")
    y_train = pd.read_json("model_dataset/y_train.json", typ="series")
    y_test = pd.read_json("model_dataset/y_test.json", typ = "series")

    return X_train,X_test,y_train,y_test