import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pdb

if __name__ == "__main__":
    example_s = pd.read_csv("./plotting_data/example_s.csv", index_col=0)
    example_m = pd.read_csv("./plotting_data/example_m.csv", index_col=0)
    example_s.replace("-",np.nan, inplace=True)
    mask = np.array(example_s.isna())
    masked_example_s = np.ma.masked_array(example_s, mask=mask)
    pdb.set_trace()
    #example_l = pd.read_csv("./plotting_data/example_l.csv")
    #example_s.columns = ["Unnamed: 0", "100000", "1000000", "10000000"]
    #example_m.columns = ["Unnamed: 0", "100000", "1000000", "10000000"]
    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(np.array(example_s), mask=example_s == np.nan, ax=ax, annot=True, fmt="0.2f", cmap="YlGnBu")
    plt.savefig("try.png")