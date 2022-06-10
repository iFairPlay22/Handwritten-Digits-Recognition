import matplotlib.pyplot as plt
from datetime import datetime
import os
import random

def randomColors():
    return list(
        { "#ff0000", "#ff8800", "#fff700", "#95ff00", "#00ff26",
          "#00ffee", "#0022ff", "#aa00ff", "#ff00f2", "#ff0000"}
    )

class SmartPlot:

    def __init__(self, title="plot", x_label="Epochs", y_label="Score", path="./output"):

        self.__data = {}
        self.__title = title
        self.__x_label = x_label
        self.__y_label = y_label
        self.__path = path

    def addPoint(self, label, color, value):
        if label not in self.__data:
            self.__data[label] = { "color": color, "data": [] }

        self.__data[label]["data"].append(value)

    def build(self):
        self.__fig, self.__ax = plt.subplots()

        for label, k in self.__data.items():
            self.__ax.plot(range(1, len(k["data"])+1), k["data"], label=label, color=k["color"])

        self.__ax.set_xlabel(self.__x_label)
        self.__ax.set_ylabel(self.__y_label)
        self.__ax.set_title(self.__title)
        self.__ax.legend()

        if not os.path.exists(self.__path):
            os.makedirs(self.__path)

        fileName = self.__path + '/plot_' + str(datetime.now())[0:19] + "_" + self.__title 
        fileName = fileName.replace("-", "_").replace(":", "_").replace(" ", "_").replace('"', "_").replace("'", "_")
        self.__fig.savefig(fileName + '.png')

    def reset(self):
        self.__data = {}

    def show():
        plt.show()