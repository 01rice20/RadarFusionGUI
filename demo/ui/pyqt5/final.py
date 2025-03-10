# -*-coding:utf-8-*-
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtCore, QtGui
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys
import os
import numpy as np
import csv
from gui_setting import Ui_Form
import time
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
from PyQt5.QtWidgets import (QWidget, QPushButton, QApplication, QHBoxLayout, QVBoxLayout)
from random import randint
import subprocess

plt.rcParams['axes.unicode_minus'] = False

class MyFigure(FigureCanvas):
    def __init__(self, width=30, height=30, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MyFigure, self).__init__(self.fig)  # Display the plot
        self.axes1 = self.fig.add_subplot(311)
        self.axes2 = self.fig.add_subplot(312)

class MainDialogImgBW(QDialog, Ui_Form):
    def __init__(self):
        super(MainDialogImgBW, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Show Results")
        # self.setMinimumSize(0,0)

        self.F = MyFigure(width=30, height=30, dpi=100)
        self.gridlayout = QGridLayout(self.groupBox)
        self.gridlayout.addWidget(self.F, 0, 1)
        self.label2.setText("Static")
        # Update the plot per 2s
        self.counter = 0
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.plotcos)
        self.timer.start()

    def plotcos(self):
        self.F.axes1.clear()
        self.F.axes2.clear()
        self.F.axes1.set_title('Heart Rate')
        self.F.axes2.set_title('Breath Rate')

        self.time = []
        self.state = []
        self.hr = []
        self.br = []
        self.predict = []

        with open('../output.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            self.counter += 1

            for i in reader:
                self.time.append(float(i[0]))
                self.state.append(int(i[1]))
                self.predict.append(int(i[2]))
                self.hr.append(float(i[3]))
                self.br.append(float(i[4]))

                if (self.state[-1] == 0):
                    self.label1.setText("Static")
                elif (self.state[-1] == 1):
                    self.label1.setText("Moving")

                if (self.predict[-1] == -1):
                    self.label2.setText("NAN")
                elif (self.predict[-1] == 0):
                    self.label2.setText("convulsion")
                elif (self.predict[-1] == 1):
                    self.label2.setText("cough")
                elif (self.predict[-1] == 2):
                    self.label2.setText("drink")
                elif (self.predict[-1] == 3):
                    self.label2.setText("fall")
                elif (self.predict[-1] == 4):
                    self.label2.setText("wave")

                self.label3.setText(str(self.hr[-1])[:5])

        self.F.axes1.plot(self.time, self.hr)
        self.F.axes2.plot(self.time, self.br)
        self.F.draw()
        self.F.update()

        self.F.figure.suptitle('')
        self.F.figure.tight_layout()


if __name__ == "__main__":
    command1 = "del ..\output.csv"
    app = QApplication(sys.argv)
    main = MainDialogImgBW()
    main.show()
    main.showFullScreen()
    app.installEventFilter(main)
    if (app.exec_() == 0):
        # os.system(command1)
        sys.exit()