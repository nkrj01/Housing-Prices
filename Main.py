import pandas as pd
from functions import Functions
from preprocesor import Preprocessor
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


def medianImputation(X, colName):
    """
    :param X: Dataframe
    :param colName: String, Name of the column for imputation
    :return: Dataframe, with column imputation
    """

    Median = X[X[colName].notna()][colName].median()
    X[colName] = X[colName].fillna(Median)
    return X


def clipping(X, colName, cutoff):
    """
    :param X: Dataframe
    :param colName: String, name of the column that is basis for clipping df
    :param cutoff: cut-off value for clipping
    :return: Dataframe, shortened df after clipping
    """
    X = X[X[colName] < cutoff]
    return X


def VIF(X, colNameList):
    """

    :param X: Dataframe
    :param colNameList: list["string"], list of column for VIF calculation
    :return: Dataframe, VIF of input columns
    """

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X[colNameList].columns
    vif_data["VIF"] = [variance_inflation_factor(X[colNameList].values, i) for i in range(X[colNameList].shape[1])]
    return vif_data


def submitFile(y_predict):

    """
    :param y_predict: Array
    :return: csv file

    This functions save a csv file of two columns "id" and "y_predict" for submission to Kaggle.

    """
    idx = np.array(range(1461, 2920, 1))
    array = np.vstack((idx, y_predict)).T
    df_submit = pd.DataFrame(array, columns=["id", "SalePrice"])
    df_submit["id"] = df_submit["id"].astype("int64")
    df_submit.to_csv("submission.csv", index=False)


trainData = pd.read_csv(r"C:\Users\14702\PycharmProjects\pythonProject2\Kaggel_housing prices\train.csv")

# Transformations.
'''trainData["SalePrice"] = np.log(trainData["SalePrice"])
trainData["OverallQual"] = np.power(trainData["OverallQual"], 1.5)
trainData["2ndFlrSF"] = np.power(trainData["2ndFlrSF"], 1.5)'''

y = trainData["SalePrice"]
X = trainData.drop(["SalePrice", "Id"], axis=1)
X = medianImputation(X, "LotFrontage")


# train and test data when fitting on partial data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# train data and X_test when fitting on whole data and predicting test data
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

# Preprocess Data. Imputation, scaling, One-hot-encoding, ordinal encoding, PCA.
# See preprocessor class for more info.
processor = Preprocessor.processor(numericalVariables, categoricalVariables, cyclicVariables)
processor.fit(X_train)
X_train_trans = processor.transform(X_train)
df_train_trans = pd.DataFrame(X_train_trans, columns=processor.get_feature_names_out())
X_test_trans = processor.transform(X_test)
df_test_trans = pd.DataFrame(X_test_trans, columns=processor.get_feature_names_out())


# drop columns that might not be useful.
'''drop = ["Utilities", "Street", "PoolQC", "Heating"]
drop = ["MiscFeature"]
df_train_trans.drop(drop, axis=1, inplace=True)
df_test_trans.drop(drop, axis=1, inplace=True)
'''


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
print(performance)
'''

# Fit of Gradient boost.
sample_weights = [1000 if (y_train[i]>400000) else 1 for i in range(len(y_train))]
gbEstimator = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
gbEstimator.fit(df_train_trans, y_train)
y_predict = gbEstimator.predict(df_test_trans)
score_test = gbEstimator.score(df_test_trans, y_test)
score_train = gbEstimator.score(df_train_trans, y_train)
print("gb test: ", score_test)
print("gb train: ", score_train)


# Fit of XG boost
'''xgbEstimator = XGBRegressor(eval_metric="mlogloss", n_estimators=150, learning_rate=0.1, max_depth=3)
xgbEstimator.fit(df_train_trans, y_train)
y_predict = xgbEstimator.predict(df_test_trans)
score_test = xgbEstimator.score(df_test_trans, y_test)
score_train = xgbEstimator.score(df_train_trans, y_train)
print("xgb test: ", score_test)
print("xgb train: ", score_train)'''


plt.scatter(y_test, y_predict)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='y=x line')
plt.title("Scatter Plots of X variables vs y")
plt.legend()
plt.show()

# submitFile(y_predict)
