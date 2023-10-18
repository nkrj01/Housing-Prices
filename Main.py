import pandas as pd
from preprocesor import Preprocessor, CyclicEncoder, MedianImputer, CountEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from xgboost import XGBRegressor
from plotting_functions import Plot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from functions import Functions
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


def dataTypeChange(X):
    """
    :param X: Dataframe
    :return: Dataframe with data type of various columns changed

    This function is only made for this specific database.
    Change this function accordingly.

    """

    for col in X.columns:
        if X[col].dtype == 'int64':
            X[col] = X[col].astype(str)
    X[cyclicVariables] = X[cyclicVariables].astype('int64')
    return X


# import data and define X and y variable
trainData = pd.read_csv(r"C:\Users\14702\PycharmProjects\pythonProject2\Kaggel_housing prices\train.csv")
y = trainData["SalePrice"]
X = trainData.drop(["SalePrice", "Id"], axis=1)


# train and test data when fitting on partial data and cross-validating on CV dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3) #3
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# train data and X_test when fitting on test data and predicting test data
# X_train = X
# y_train = y
# testData = pd.read_csv(r"C:\Users\14702\PycharmProjects\pythonProject2\Kaggel_housing prices\test.csv")
# X_test = testData.drop(["Id"], axis=1)


# return a dictionary with all the categorical type variable along with all the categorical values they can take
# Also return the name of all the numerical type variables.
filepath = r"C:\Users\14702\PycharmProjects\pythonProject2\Kaggel_housing prices\data_description.txt"
objFunctions = Functions(X)
categoricalVariables, numericalVariables, cyclicVariables, NAColumns = objFunctions.variableTypeSeparator(filepath)

X_train = dataTypeChange(X_train)
X_test = dataTypeChange(X_test)

# Defining imputer, encoder, and pca for all three variable types.
categoricalImputer = SimpleImputer(strategy='constant', fill_value="Missing")
categories = [categoricalVariables[key] for key in categoricalVariables.keys()]
OrdinalEncoder = OrdinalEncoder(encoded_missing_value=-1, handle_unknown="use_encoded_value", unknown_value=-1)
# OneHotEncoder = OneHotEncoder(categories=categories, handle_unknown="ignore", sparse=False)
pca_ct = PCA(n_components=10)

# numericalImputer = SimpleImputer(strategy='constant', fill_value=-1)
numericalImputer = MedianImputer()
numericalEncoder = MinMaxScaler()
pca_nm = PCA(n_components=10)

cyclicImputer = SimpleImputer(strategy='constant', fill_value=-1)
cyclicEncoder = CyclicEncoder()
pca_cy = PCA(n_components=10)

# Preprocess Data. Imputation, scaling, One-hot-encoding, ordinal encoding, PCA etc.
# See preprocessor class for more info.
processor = Preprocessor.processor(numericalVariables, categoricalVariables, cyclicVariables,
                                   categoricalImputer, CountEncoder(), False,
                                   numericalImputer, numericalEncoder, False,
                                   cyclicImputer, cyclicEncoder, False)

processor.fit(X_train)
X_train_trans = processor.transform(X_train)
df_train_trans = pd.DataFrame(X_train_trans, columns=processor.get_feature_names_out())
X_test_trans = processor.transform(X_test)
df_test_trans = pd.DataFrame(X_test_trans, columns=processor.get_feature_names_out())


# sample weights generation to pay more attention to high sale prices data
"""sample_weights = [0.1 if (y_train[i]>200000) else 1 for i in range(len(y_train))]
sample_weights_test = [0.1 if (y_test[i]>200000) else 1 for i in range(len(y_test))]
"""

# Fit of Gradient boost.
gbEstimator = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gbEstimator.fit(df_train_trans, y_train)
y_predict = gbEstimator.predict(df_test_trans)
score_test = gbEstimator.score(df_test_trans, y_test)
score_train = gbEstimator.score(df_train_trans, y_train)
print("gb test: ", score_test)
print("gb train: ", score_train)


# Fit of autogluon
'''train = pd.concat([df_train_trans, y_train], axis=1)
test = pd.concat([df_test_trans, y_test], axis=1)
# test = df_test_trans
excluded_models = ['KNN', 'NN']
predictor = TabularPredictor(label="SalePrice").fit(train, presets='best_quality')
predictions = predictor.predict(test.drop(columns=['SalePrice']))
y_predict = predictor.predict(test)
performance = predictor.evaluate(test)
table_result = predictor.leaderboard(test)
print(performance)'''


# Fit of XG boost
'''xgbEstimator = XGBRegressor(eval_metric="mlogloss", n_estimators=100, learning_rate=0.15, max_depth=5)
xgbEstimator.fit(df_train_trans, y_train)
y_predict = xgbEstimator.predict(df_test_trans)
score_test = xgbEstimator.score(df_test_trans, y_test)
score_train = xgbEstimator.score(df_train_trans, y_train)
print("xgb test: ", score_test)
print("xgb train: ", score_train)'''


# y_predict and y_actual with y=x line to visualize the accuracy of prediction
plt.scatter(y_test, y_predict)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='y=x line')
plt.title("Scatter Plots of X variables vs y")
plt.legend()
plt.show()


# Function to create CSV file for submission
"""submitFile(y_predict)"""
