from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


class Plot:

    @staticmethod
    def scatterPlot(X, Y):
        plt.scatter(X, Y)
        plt.title("Scatter Plots of X variables vs y")
        plt.xlabel("X Values")
        plt.ylabel("y Values")
        plt.legend()
        plt.xlim(0, 300)  # Set the x-axis limits from 0 to 6
        plt.ylim(0, 50000)  # Set the y-axis limits from 0 to 25
        plt.show()

    @staticmethod
    def histogram(X):
        plt.hist(X)
        plt.title("Scatter Plots of X variables vs y")
        plt.xlabel("X Values")
        plt.ylabel("y Values")
        plt.legend()
        plt.show()


    @staticmethod
    def heat_map(corr_matrix):
        sns.heatmap(corr_matrix,
                    cmap="coolwarm",
                    center=0,
                    annot=True,
                    fmt=" .1g"
                    )

