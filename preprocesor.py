import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin


class CyclicEncoder(TransformerMixin):
    """
    Custom encoder defined using Mixin class to encode cyclical variable
    """
    def fit(self, X, y=None):
        """
        :param X: (Dataframe or Numpy array)
        :param y: None
        :return: max value of the cycle
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.input_features_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [X.name]
            X = X.astype(float)
            X = X.values
            X = X.reshape(-1, 1)
        self.maxValue = np.max(X)
        return self

    def transform(self, X, y=None):
        """
        :param X: (DataFrame of Numpy array)
        :param y: None
        :return: Returns encoded cyclic variable. Two column (sin x and cos x for each column)
        """
        sin_component = np.sin(2 * np.pi * X / self.maxValue)
        cos_component = np.cos(2 * np.pi * X / self.maxValue)
        result = np.column_stack((sin_component, cos_component))
        return result

    def get_feature_names_out(self, input_features=None):
        """
        :param input_features: list[string]
        :return: list[string], Name of the output features
        """
        if input_features is None:
            input_features = self.input_features_
        output_feature_names = []
        for feature in input_features:
            output_feature_names.extend([f"{feature}_sin", f"{feature}_cos"])
        return output_feature_names

    def get_params(self, deep=True):
        return {}


class MedianImputer(TransformerMixin):
    """
    Custom imputer to impute missing values with the median of the column
    """
    def fit(self, X, y=None):
        """
        :param X: (Dataframe or numpy array)
        :param y: None
        :return: (numpy array) of length equals to number of columns with median value of each column
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.input_features_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [X.name]
            X = X.astype(float)
            X = X.values
        self.median = np.nanmedian(X, axis=0)
        return self

    def transform(self, X, y=None):

        """
        :param X: (Numpy array or df)
        :param y: None
        :return: (numpy array) with missing values imputed with medians
        """
        X = X.values
        X = X.astype(float)
        for col_idx in range(X.shape[1]):
            nan_indices = np.isnan(X[:, col_idx])  # Identify NaN values in the column
            X[nan_indices, col_idx] = self.median[col_idx]
        return X

    def get_feature_names_out(self, input_features=None):

        """
        :param input_features: list[string]
        :return: list[string], Name of the output features
        """
        if input_features is None:
            input_features = self.input_features_
        output_feature_names = []
        for feature in input_features:
            output_feature_names.extend([f"{feature}"])
        return output_feature_names

    def get_params(self, deep=True):
        return {}


class CountEncoder(TransformerMixin):

    """
    Class that encodes categorical variable with the number of time they appear in the feature.
    For example, "x" appears 3 times in a column and "y" appears 2 times, then all the x's will be
    replaced by 3 and all the y's will be replaced by 2
    """
    def fit(self, X, y=None):
        """
        :param X: (numpy array or df)
        :param y: None
        :return: (list[dictionary]), one dictionary per column. Each dictionary contains
        the unique value of the column along with the number of times that unique value is occuring
        in the column.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.input_features_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [X.name]
            # X = X.astype(float)
            X = X.values
        unique_values_per_column = [np.unique(X[:, i]) for i in range(X.shape[1])]
        self.list_unique_dict = []
        for column_index in range(X.shape[1]):
            unique_dict = {}
            for j in unique_values_per_column[column_index]:
                unique_dict[j] = np.count_nonzero(X[:, column_index] == j)
            self.list_unique_dict.append(unique_dict)
        return self

    def transform(self, X, y=None):
        """
        :param X: (numpy array or df)
        :param y: None
        :return: (numpy array), with categorical variable encoded with the number of occurrences.
        If in test set, a category is encountered which was not seen during fit, then that category
        is encoded as 0.
        """
        for column_index in range(X.shape[1]):
            dict = self.list_unique_dict[column_index]
            mask_list = []
            for key in dict.keys():
                mask = X[:, column_index] == key
                X[mask, column_index] = dict[key]
                mask_list.append(mask)

            total_mask = np.zeros(X.shape[0])
            total_mask = total_mask.astype("bool")
            for mask in mask_list:
                total_mask = np.logical_or(total_mask, mask)
            total_mask = np.logical_not(total_mask)
            X[total_mask, column_index] = 0
        return X

    def get_feature_names_out(self, input_features=None):
        """
        :param input_features: list[string]
        :return: list[string], Name of the output features
        """
        if input_features is None:
            input_features = self.input_features_
        output_feature_names = []
        for feature in input_features:
            output_feature_names.extend([f"{feature}"])
        return output_feature_names

    def get_params(self, deep=True):
        return {}


class Preprocessor:
    """
    This class puts together all the encoders and imputes defined into pipelines and finally into
    a column transformer that is used for preprocessing all the variables together.
    """
    @staticmethod
    def processor(
            numericalVariables, categoricalVariables, cyclicVariables,
            ctImputer, ctEncoder, pca_ct,
            nmImputer, nmEncoder, pca_nm,
            cyImputer, cyEncoder, pca_cy):

        """
        :param numericalVariables: list of all the numerical variables
        :param categoricalVariables: list of all the categorical variables
        :param cyclicVariables: list of all the cyclic variable variables
        :param ctImputer: categorical variable imputer
        :param ctEncoder: categorical variable encoder
        :param pca_ct: PCA for categorical variable. Default: False
        :param nmImputer: numerical variable imputer
        :param nmEncoder: numerical variable encoder
        :param pca_nm: PCA for numerical variable. Default: False
        :param cyImputer: cyclic variable imputer
        :param cyEncoder: cyclic variable encoder
        :param pca_cy: PCA for cyclic variable. Default: False
        :return: Column transformer object on which fit and tranform function
                 can be called for variable tranformation.
        """
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
