import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
