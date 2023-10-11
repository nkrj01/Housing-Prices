import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
import numpy as np
from functions import Functions
import seaborn as sns
import matplotlib.pyplot as plt


# Set up the aesthetics and layout
sns.set_style("whitegrid")

# Plot distributions for each column
for column in df.columns:
    plt.figure(figsize=(10, 5))

    # Check if the column datatype is numeric before plotting
    if pd.api.types.is_numeric_dtype(df[column]):
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution for {column}')
        plt.show()
    else:
        print(f"Column {column} is not numeric and will not be plotted.")