import pandas as pd


def combine_ts(series_of_labels, *args):
    data = pd.DataFrame(series_of_labels)

    for arg in args:
        data = data.join(arg, how="outer")

        for col in data.columns:
            if col == "label":
                data.loc[:, col] = data.loc[:, col].fillna(method="bfill")
            else:
                data.loc[:, col] = data.loc[:, col].fillna(method="ffill")

    data = data.dropna()
    label = data.label
    data = data.drop(columns="label")
    return data, label

########################################################################################################################
