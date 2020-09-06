import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = pd.read_csv("data.csv")

print(raw_data.info())
print(raw_data.isna().sum())
print(raw_data.head(10))
print(raw_data.columns.values)
"""
569 samples, 33 features, 1 int, 1 str, 31 float, 0 null
ID is float, diagnosis is str M(malignant) or B(benign), rest are only numerical continuous values
By now we can drop the ID as it is not significative to keep it
We can also drop the column Unamed:32 as it is full of NaN
We also need to map M and B on 0 and 1

Lucky me no null in the dataset
No categorical features to transform
"""

raw_df = pd.DataFrame(raw_data)
raw_df = raw_df.drop(["id"], axis=1)    #removing id
raw_df = raw_df.drop(["Unnamed: 32"], axis=1)
my_map = {"M" : 1, "B" : 0}
raw_df["diagnosis"] = raw_df["diagnosis"].map(my_map)
corr = raw_df.corr()
f, ax = plt.subplots(figsize=(10, 40))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, center=0, square=True, linewidths=.5)
plt.show()

"""
The correlation matrix indicates that the diagnosis is not correlated at all with symmetry_se, texture_se and fractal_dimension_mean
We decide to remove these features
Also we observe that radius_mean is hightly correlated with area_worst, perimeter_worst, radius_worst, area_mean and perimeter_mean
All these are all correlated together and it is normal as they are mesuring the same thing = the size of the tumor 
(as long as the tumor has a circle shape, removing the volume (if existing) could have been an error as we dont know the shape of the tumor)
We decide to drop area_worst, perimeter_worst, radius_worst, area_mean and perimeter_mean and only keep radius_mean
"""

raw_df = raw_df.drop(["area_worst", "perimeter_worst", "radius_worst", "area_mean", "perimeter_mean"], axis=1)

raw_df.to_csv("modified_data.csv", index=False)