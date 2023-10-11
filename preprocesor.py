import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
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
            pca_on={"NV": False, "CV": False, "CyV": False}, pca_n=20):

        categoricalImputer = SimpleImputer(strategy='constant', fill_value="Missing")
        categories = [categoricalVariables[key] for key in categoricalVariables.keys()]
        OE = OrdinalEncoder(encoded_missing_value=-1, handle_unknown="use_encoded_value", unknown_value=-1)
        OHE = OneHotEncoder(categories=categories, handle_unknown="ignore", sparse=False)
        pca = PCA(n_components=pca_on)
        if pca_on["NV"]:
            categoricalImputerTransformer = Pipeline([('imputer', categoricalImputer),
                                                      ('OHE', OE),
                                                      ('pca', pca)])
        else:
            categoricalImputerTransformer = Pipeline([('imputer', categoricalImputer),
                                                      ('OHE', OE)])

        # Defines a pipeline for imputation and encoding for Continuous variables
        if pca_on["CV"]:
            numericalImputer = SimpleImputer(strategy='constant', fill_value=-1)
            numericalImputerTransformer = Pipeline([('imputer', numericalImputer),
                                                    ("MMS", StandardScaler()),
                                                    ('pca', pca)])
        else:
            numericalImputer = SimpleImputer(strategy='constant', fill_value=-1)
            numericalImputerTransformer = Pipeline([('imputer', numericalImputer),
                                                    ("MMS", StandardScaler())])

        if pca_on["CyV"]:
            cyclicImputer = SimpleImputer(strategy='constant', fill_value=-1)
            cyclicImputerTransformer = Pipeline([('imputer', cyclicImputer),
                                                 ("enc", CyclicEncoder()),
                                                 ('pca', pca)])
        else:
            cyclicImputer = SimpleImputer(strategy='constant', fill_value=-1)
            cyclicImputerTransformer = Pipeline([('imputer', cyclicImputer),
                                                 ("enc", CyclicEncoder())])

        # Column transformer for numerical and categorical imputation
        ct = ColumnTransformer([
            ('NumT', numericalImputerTransformer, numericalVariables),
            ("CatT", categoricalImputerTransformer, list(categoricalVariables.keys())),
            ("CycT", cyclicImputerTransformer, cyclicVariables)
        ], remainder="passthrough", verbose_feature_names_out=False)
        return ct
