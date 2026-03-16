import numpy as np
from preprocessing import load_data, create_rul_categories
from segmentation import run_segmentation
from clustering import recursive_clustering
from kadane import kadane

DATA_PATH = "../data/rul_hrs.csv"


def main():
    df = load_data(DATA_PATH)
    df = create_rul_categories(df)

    sensors = [col for col in df.columns if col.startswith("sensor")]

    # Task 1 segmentation example
    signal = df["sensor_00"].values
    segments = run_segmentation(signal, threshold=1)
    print("Segmentation complexity:", len(segments))

    # Task 2 clustering
    X = df[sensors].values
    clusters = recursive_clustering(X, 4)
    print("Clusters:", len(clusters))
    print("Data shape:", X.shape)

    # Task 3 Kadane example
    sensor = df["sensor_00"].values
    d = np.abs(np.diff(sensor))
    x = d - np.mean(d)
    s, e, score = kadane(x)
    print("Max deviation interval:", s, e, "score:", score)


if __name__ == "__main__":
    main()
