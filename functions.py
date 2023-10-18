import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor


class Functions:
    def __init__(self, _df):
        self.df = _df

    # Return a dictionary with all the categorical type variable along with all the categorical values they can take.
    # Also return the name of all the numerical type variables.
    # Also returns the name of all the columns that contains at least on NA

    # Logic here is to collect the data that is between two empty lines in different lists in the txt file.
    def variableTypeSeparator(self, filepath):
        with open(filepath, 'r') as file:
            # Read the content of the file
            lineNumber = 0
            emptyLineList = []
            for line in file:
                lineNumber += 1
                if not line.strip():
                    emptyLineList.append(lineNumber)  # stores the line number of empty lines

        with open(filepath, 'r') as file:
            lines = file.readlines()
            textLinesList = []
            for j in range(0, len(emptyLineList) - 1):
                textLine = []
                for i in range(emptyLineList[j], emptyLineList[j + 1]):  # pick data between two empty lines
                    strippedLine = lines[i].strip()
                    tabPosition = strippedLine.find("\t")
                    textLine.append(strippedLine[0:tabPosition])  # only pick the characters not the description
                textLine = textLine[0:-1]
                textLinesList.append(textLine)

        # remove the texts that describes continuous variable
        tempList = textLinesList
        textLinesList = []
        for i, lst in enumerate(tempList):
            if len(lst) > 1 or (i < len(tempList) - 1 and len(tempList[i + 1]) > 1):
                textLinesList.append(lst)

        # List of numerical variables
        numericalVariables = [i for i in tempList if i not in textLinesList]
        numericalVariables = sum(numericalVariables, [])
        temp = []
        for i in numericalVariables:
            colonPosition = i.find(":")
            temp.append(i[:colonPosition])
        numericalVariables = temp

        # remove month sold from numerical variable
        cyclicalVariables = ["MoSold"]
        numericalVariables.remove("MoSold")

        # Dictionary of Categorical variables along with all the categories.
        # pair the name of the variable with the categorical values list
        range1 = range(0, len(textLinesList), 2)
        range2 = range(1, len(textLinesList), 2)
        value = []
        name = []
        for i, j in zip(range1, range2):
            colonPosition = textLinesList[i][0].find(":")
            name.append(textLinesList[i][0][0:colonPosition])
            value.append(textLinesList[j])
        categoricalVariables = dict(zip(name, value))

        totalNA = self.df.isna().sum()
        NAColumns = totalNA[totalNA != 0]

        for key in categoricalVariables.keys():
            if key in NAColumns:
                if "NA" in categoricalVariables[key]:
                    index = categoricalVariables[key].index("NA")
                    categoricalVariables[key][index] = "Missing"
                else:
                    categoricalVariables[key].append("Missing")

        return categoricalVariables, numericalVariables, cyclicalVariables, NAColumns

    @staticmethod
    def Rvalue(X, Y):
        R = np.corrcoef(X, Y)[0, 1]
        return R


    @staticmethod
    def medianImputation(X, colName):
        """
        :param X: Dataframe
        :param colName: String, Name of the column for imputation
        :return: Dataframe, with column imputation
        """

        Median = X[X[colName].notna()][colName].median()
        X[colName] = X[colName].fillna(Median)
        return X

    @staticmethod
    def clipping(X, colName, cutoff):
        """
        :param X: Dataframe
        :param colName: String, name of the column that is basis for clipping df
        :param cutoff: cut-off value for clipping
        :return: Dataframe, shortened df after clipping
        """
        X = X[X[colName] < cutoff]
        return X

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def regressorImputer(X, feature1, feature2):

        """
        :param X: DataFrame
        :return: DataFrame, with column imputed based on regression model

        This function first fits feature2 with feature 2 using non-nan values,
        This uses feature2 to predict nan values of feature1
        """

        df = X[[feature2, feature1]]
        im_X_test = df[df[feature1].isna()]
        im_X_test = im_X_test[feature2]
        df = df.dropna(subset=feature1)
        im_X = df[feature2].to_numpy().reshape(-1, 1)
        im_Y = df[feature1].to_numpy().reshape(-1, 1)
        reg = XGBRegressor(eval_metric="mlogloss", n_estimators=50, learning_rate=0.1, max_depth=2).fit(im_X, im_Y)
        im_y_pred = reg.predict(im_X_test.to_numpy().reshape(-1, 1))
        bool = X[feature1].isna()
        X.loc[bool, feature1] = im_y_pred
        return X

