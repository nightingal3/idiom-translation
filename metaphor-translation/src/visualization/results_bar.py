import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # data = {
    #     "Language": ["FI", "FI", "FI", "FI", "FI", "FI", "FI", "FI", "FI", "FI", "FI", "FI", "FI", "FI", "FI", "FI", "FR", "FR", "FR", "FR"],  # extend this list
    #     "Test Set": ["Idioms", "Idioms", "Idioms", "Idioms", "Literal", "Literal", "Literal", "Literal", "Random", "Random", "Random", "Random", "Idioms", "Idioms", "Idioms", "Idioms", "Literal", "Literal", "Literal", "Literal"],  # extend this list
    #     "Model": ["Normal", "Knn", "Upweight", "Knn + upweight", "Normal", "Knn", "Upweight", "Knn + upweight", "Normal", "Knn", "Upweight", "Knn + upweight", "Normal", "Knn", "Upweight", "Knn + upweight", "Normal", "Knn", "Upweight", "Knn + upweight"],  # extend this list
    #     "BLEU": [0.1608, 0.179, 0.1773, 0.182, 0.2093, 0.2257, 0.2314, 0.2362, 0.2365, 0.2557, 0.2374, 0.2498, 0.2001, 0.2154, 0.2174, 0.2309, 0.2778, 0.2824, 0.2923, 0.2883],  # extend this list
    #     "RougeL-sum": [0.3737, 0.3904, 0.3895, 0.4027, 0.5053, 0.535, 0.5281, 0.5335, 0.535, 0.5602, 0.5458, 0.564, 0.4329, 0.4547, 0.4398, 0.4568, 0.5261, 0.5318, 0.5332, 0.5325],  # extend this list
    #     "BertScore": [0.9126, 0.9152, 0.9154, 0.9172, 0.935, 0.937, 0.9364, 0.9387, 0.9145, 0.9183, 0.9163, 0.9193, 0.9211, 0.9235, 0.9235, 0.926, 0.9377, 0.9387, 0.9384, 0.9399],  # extend this list
    #     "Meteor": [0.3592, 0.3745, 0.3709, 0.3833, 0.505, 0.5161, 0.5258, 0.5217, 0.4971, 0.5178, 0.5102, 0.5261, 0.4393, 0.4493, 0.4419, 0.4581, 0.5504, 0.5544, 0.5564, 0.5549]  # extend this list
    # }
   #df = pd.DataFrame(data)
    df = pd.read_csv("./data/tab5_data.csv")
    palettes = {
        "Idioms": "viridis",
        "Literal": "plasma",
        "Random": "inferno"
    }

    df_melted = df.melt(id_vars=["Language", "Test Set", "Model"], var_name="Metric", value_name="Score")

    # Convert the melted dataframe back to wide format
    df_wide = df_melted.pivot_table(index=['Language', 'Test Set', 'Model'], columns='Metric', values='Score').reset_index()

    # Get unique languages, test sets, models and metrics
    languages = df_wide['Language'].unique()
    metrics = df_melted['Metric'].unique()
    test_sets = df_wide["Test Set"].unique()

    # Create a subplot for each language
    fig, axes = plt.subplots(nrows=len(languages), ncols=len(metrics), figsize=(15, 8))

    # Define the order and color mapping for the models
    model_order = ["Normal", "Upweight", "Knn", "Knn + upweight"]
    color_palette = sns.color_palette("Set2")
    color_mapping = dict(zip(model_order, color_palette))

    # Create legend handles manually
    legend_patches = [mpatches.Patch(color=color_mapping[model], label=model) for model in model_order]
    ybounds = {(0, 0, 0): (0.15, 0.19), (0, 0, 1): (0.15, 0.25), (0, 0, 2): (0.2, 0.27),
               (0, 1, 0): (0.3, 0.42), (0, 1, 1): (0.4, 0.55), (0, 1, 2): (0.5, 0.58),
               (0, 2, 0): (0.85, 0.95), (0, 2, 1): (0.9, 0.96), (0, 2, 2): (0.9, 0.93),
               (0, 3, 0): (0.3, 0.4), (0, 3, 1): (0.45, 0.55), (0, 3, 2): (0.45, 0.55),
               (1, 0, 0): (0.18, 0.25), (1, 0, 1): (0.25, 0.3), (1, 0, 2): (0.25, 0.35),
               (1, 1, 0): (0.4, 0.48), (1, 1, 1): (0.5, 0.54), (1, 1, 2): (0.44, 0.68),
               (1, 2, 0): (0.9, 0.93), (1, 2, 1): (0.9, 0.95), (1, 2, 2): (0.9, 0.95),
               (1, 3, 0): (0.4, 0.46), (1, 3, 1): (0.5, 0.56), (1, 3, 2): (0.45, 0.65),
               (2, 0, 0): (0.08, 0.1), (2, 0, 1): (0.12, 0.15), (2, 0, 2): (0.08, 0.1),
               (2, 1, 0): (0.25, 0.3), (2, 1, 1): (0.38, 0.4), (2, 1, 2): (0.32, 0.36),
               (2, 2, 0): (0.9, 0.906), (2, 2, 1): (0.9, 0.93), (2, 2, 2): (0.85, 0.9),
               (2, 3, 0): (0.25, 0.32), (2, 3, 1): (0.4, 0.45), (2, 3, 2): (0.3, 0.36)}
    # Iterate over languages (each row in the figure)
    for i, language in enumerate(languages):
        # Iterate over metrics (each column in the figure)
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            # Select the subset of the data for the current language and metric
            data_subset = df_wide[df_wide['Language'] == language]
            
            # Plot a grouped bar plot
            #sns.barplot(x='Test Set', y=metric, hue='Model', data=data_subset, ax=ax, hue_order=model_order, palette=color_mapping)
            widths = [0.2, 0.2, 0.2]
            offsets = [0, 0.35, 0.7]
            for k, (test_set, width, offset) in enumerate(zip(test_sets, widths, offsets)):
                subax = ax.inset_axes([offset, 0, width, 1])  # adjust as needed
                data_testset = data_subset[data_subset['Test Set'] == test_set]
                #import pdb; pdb.set_trace()
                sns.barplot(x='Model', y=metric, data=data_testset, ax=subax, order=model_order, palette=color_mapping)
                subax.set_title(test_set, y=-0.2)
                subax.set_xticks([])
                subax.set_xlabel("")
                subax.set_ylabel("")
                subax.set_xticklabels("")
                subax.yaxis.get_major_ticks()[0].label1.set_visible(False)  # hide the first y-tick label
                subax.tick_params(axis="y", pad=0.01)
                subax.tick_params(axis="y", labelsize=8, length=0)
                if (i, j, k) in ybounds:
                    subax.set_ylim(ybounds[(i, j, k)])

            # Set plot title and remove the legend
            ax.set_title(f'{language} - {metric}', y=1.1, fontsize=12)
            #ax.get_legend().remove()
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.axis('off')
            

    # Add a single legend for the whole figure
    legend_elements = [
        Patch(facecolor=color_mapping[model], label=model) for model in model_order
    ]

    # Adjust the layout so that legend is fully visible
    plt.subplots_adjust(right=0.85, hspace=0.5)
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.6), fontsize=12)

    #plt.tight_layout()
    plt.savefig("metric_bar.png")
    plt.savefig("metric_bar.pdf")