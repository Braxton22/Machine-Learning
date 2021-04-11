from sklearn.datasets import fetch_california_housing
import pandas as pd

california = fetch_california_housing()

california_df = pd.DataFrame(california.data, columns=california.feature_names)
california_df["MedHouseValue"] = pd.Series(california.target)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.1)
sns.set_style("whitegrid")
grid = sns.pairplot(california_df, vars=california_df.columns[0:2])
# I only did the first two columns because otherwise it takes absolutely forever to load.
plt.show()