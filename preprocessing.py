import pandas as pd


def load_data(path):
    df = pd.read_csv(path)

    df = df.drop(columns=["Unnamed: 0"])

    df = df.iloc[:10000]

    sensors = [col for col in df.columns if col.startswith("sensor")]

    X = df[sensors].values
    return df


def create_rul_categories(df):
    q10 = df["rul"].quantile(0.10)
    q50 = df["rul"].quantile(0.50)
    q90 = df["rul"].quantile(0.90)

    def categorize(r):
        if r < q10:
            return 0
        elif r < q50:
            return 1
        elif r < q90:
            return 2
        else:
            return 3

    df["condition"] = df["rul"].apply(categorize)
    return df
