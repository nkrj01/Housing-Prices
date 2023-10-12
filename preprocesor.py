import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA


class CyclicEncoder(TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.input_features_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [X.name]
            X = X.values
            X = X.reshape(-1, 1)
        self.maxValue = np.max(X)
        return self

    def transform(self, X, y=None):
        sin_component = np.sin(2 * np.pi * X / self.maxValue)
        cos_component = np.cos(2 * np.pi * X / self.maxValue)
        result = np.column_stack((sin_component, cos_component))
        return result

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.input_features_
        output_feature_names = []
        for feature in input_features:
            output_feature_names.extend([f"{feature}_sin", f"{feature}_cos"])
        return output_feature_names


class Preprocessor:

    @staticmethod
    def processor(
            numericalVariables, categoricalVariables, cyclicVariables,
            ctImputer, ctEncoder, pca_ct,
            nmImputer, nmEncoder, pca_nm,
            cyImputer, cyEncoder, pca_cy):

        if pca_ct:
            categoricalImputerTransformer = Pipeline([('imputer', ctImputer),
                                                      ('enc', ctEncoder),
                                                      ('pca', pca_ct)])
        else:
            categoricalImputerTransformer = Pipeline([('imputer', ctImputer),
                                                      ('enc', ctEncoder)])

        # Defines a pipeline for imputation and encoding for Continuous variables
        if pca_nm:
            numericalImputerTransformer = Pipeline([('imputer', nmImputer),
                                                    ("MMS", nmEncoder),
                                                    ('pca', pca_nm)])
        else:
            numericalImputerTransformer = Pipeline([('imputer', nmImputer),
                                                    ("MMS", nmEncoder)])

        if pca_cy:
            cyclicImputerTransformer = Pipeline([('imputer', cyImputer),
                                                 ("enc", cyEncoder),
                                                 ('pca', pca_cy)])
        else:
            cyclicImputerTransformer = Pipeline([('imputer', cyImputer),
                                                 ("enc", cyEncoder)])

        # Column transformer for numerical and categorical imputation
        ct = ColumnTransformer([
            ('NumT', numericalImputerTransformer, numericalVariables),
            ("CatT", categoricalImputerTransformer, list(categoricalVariables.keys())),
            ("CycT", cyclicImputerTransformer, cyclicVariables)
        ], remainder="passthrough", verbose_feature_names_out=False)
        return ct
