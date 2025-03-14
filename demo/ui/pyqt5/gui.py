from PyQt5 import QtCore, QtGui, QtWidgets
import csv
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(fig)
        self.axes = fig.add_subplot(111)
        self.axes.figure.tight_layout()

class Ui_Frame(object):
    def setupUi(self, Frame):
        Frame.setObjectName("Frame")
        Frame.resize(1174, 860)
        self.groupBox = QtWidgets.QGroupBox(Frame)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 1131, 241))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(260, 10, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(580, 10, 271, 41))
      
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(70, 10, 121, 41))
     
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(880, 10, 231, 41))
      
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_22 = QtWidgets.QLabel(self.groupBox)
        self.label_22.setGeometry(QtCore.QRect(320, 80, 146, 111))
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei")
        font.setPointSize(80)
        self.label_22.setFont(font)
        self.label_22.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_22.setAutoFillBackground(False)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.groupBox)
        self.label_23.setGeometry(QtCore.QRect(650, 80, 146, 111))
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei")
        font.setPointSize(80)
        self.label_23.setFont(font)
        self.label_23.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_23.setAutoFillBackground(False)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.label_26 = QtWidgets.QLabel(self.groupBox)
        self.label_26.setGeometry(QtCore.QRect(45, 55, 180, 180))
        self.label_26.setText("")
        self.label_26.setScaledContents(True)
        self.label_26.setObjectName("label_26")
        self.label_20 = QtWidgets.QLabel(self.groupBox)
        self.label_20.setGeometry(QtCore.QRect(900, 30, 200, 200))
        self.label_20.setText("")
        self.label_20.setScaledContents(True)
        self.label_20.setObjectName("label_20")
        self.groupBox_2 = QtWidgets.QGroupBox(Frame)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 270, 551, 571))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_24 = QtWidgets.QLabel(self.groupBox_2)
        self.label_24.setGeometry(QtCore.QRect(130, 10, 271, 41))
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_24.setFont(font)
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.label_28 = QtWidgets.QLabel(self.groupBox_2)
        self.label_28.setGeometry(QtCore.QRect(130, 290, 271, 41))
       
        self.label_28.setFont(font)
        self.label_28.setAlignment(QtCore.Qt.AlignCenter)
        self.label_28.setObjectName("label_28")

        self.widget = QtWidgets.QWidget(self.groupBox_2)
        self.widget.setGeometry(QtCore.QRect(10, 50, 531, 231))
        self.widget.setObjectName("widget")
        self.widget_2 = QtWidgets.QWidget(self.groupBox_2)
        self.widget_2.setGeometry(QtCore.QRect(10, 330, 531, 231))
        self.widget_2.setObjectName("widget_2")
        self.canvas1 = MplCanvas(self.widget, width=5, height=4, dpi=100)
        self.layout1 = QtWidgets.QVBoxLayout(self.widget)
        self.layout1.addWidget(self.canvas1)
        self.canvas2 = MplCanvas(self.widget_2, width=5, height=4, dpi=100)
        self.layout2 = QtWidgets.QVBoxLayout(self.widget_2)
        self.layout2.addWidget(self.canvas2)

        self.groupBox_3 = QtWidgets.QGroupBox(Frame)
        self.groupBox_3.setGeometry(QtCore.QRect(580, 270, 331, 571))
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(10, 50, 311, 231))
        self.label.setText("")
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_6 = QtWidgets.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(10, 330, 311, 231))
        self.label_6.setText("")
        self.label_6.setScaledContents(True)
        self.label_6.setObjectName("label_6")
        self.label_25 = QtWidgets.QLabel(self.groupBox_3)
        self.label_25.setGeometry(QtCore.QRect(30, 10, 271, 41))
      
        self.label_25.setFont(font)
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.label_29 = QtWidgets.QLabel(self.groupBox_3)
        self.label_29.setGeometry(QtCore.QRect(30, 290, 271, 41))
      
        self.label_29.setFont(font)
        self.label_29.setAlignment(QtCore.Qt.AlignCenter)
        self.label_29.setObjectName("label_29")
        self.groupBox_4 = QtWidgets.QGroupBox(Frame)
        self.groupBox_4.setGeometry(QtCore.QRect(920, 270, 231, 571))
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.label_9 = QtWidgets.QLabel(self.groupBox_4)
        self.label_9.setGeometry(QtCore.QRect(30, 10, 181, 41))
     
        self.label_9.setFont(font)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.groupBox_4)
        self.label_10.setGeometry(QtCore.QRect(0, 80, 191, 51))
      
        self.label_10.setFont(font)
        self.label_10.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_10.setAutoFillBackground(False)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.groupBox_4)
        self.label_11.setGeometry(QtCore.QRect(0, 180, 191, 51))
      
        self.label_11.setFont(font)
        self.label_11.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_11.setAutoFillBackground(False)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.groupBox_4)
        self.label_12.setGeometry(QtCore.QRect(0, 280, 191, 51))
       
        self.label_12.setFont(font)
        self.label_12.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_12.setAutoFillBackground(False)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.groupBox_4)
        self.label_13.setGeometry(QtCore.QRect(0, 380, 191, 51))
      
        self.label_13.setFont(font)
        self.label_13.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_13.setAutoFillBackground(False)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.groupBox_4)
        self.label_14.setGeometry(QtCore.QRect(0, 480, 191, 51))
      
        self.label_14.setFont(font)
        self.label_14.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_14.setAutoFillBackground(False)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.groupBox_4)
        self.label_15.setGeometry(QtCore.QRect(160, 80, 81, 51))
       
        self.label_15.setFont(font)
        self.label_15.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_15.setAutoFillBackground(False)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.groupBox_4)
        self.label_16.setGeometry(QtCore.QRect(160, 180, 81, 51))
       
        self.label_16.setFont(font)
        self.label_16.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_16.setAutoFillBackground(False)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.groupBox_4)
        self.label_17.setGeometry(QtCore.QRect(160, 280, 81, 51))
      
        self.label_17.setFont(font)
        self.label_17.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_17.setAutoFillBackground(False)
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.groupBox_4)
        self.label_18.setGeometry(QtCore.QRect(160, 380, 81, 51))
       
        self.label_18.setFont(font)
        self.label_18.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_18.setAutoFillBackground(False)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.groupBox_4)
        self.label_19.setGeometry(QtCore.QRect(160, 480, 81, 51))
     
        self.label_19.setFont(font)
        self.label_19.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_19.setAutoFillBackground(False)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")

        _translate = QtCore.QCoreApplication.translate
        Frame.setWindowTitle(_translate("Frame", "Show Results"))
        self.label_2.setText(_translate("Frame", "Heart Rate (bpm)"))
        self.label_3.setText(_translate("Frame", "Breath Rate (bpm)"))
        self.label_4.setText(_translate("Frame", "State"))
        self.label_5.setText(_translate("Frame", "Predict Class"))
        self.label_24.setText(_translate("Frame", "Heart Rate Curve"))
        self.label_28.setText(_translate("Frame", "Breath Rate Curve"))
        self.label_25.setText(_translate("Frame", "micro-Doppler"))
        self.label_29.setText(_translate("Frame", "Range-Time"))
        self.label_9.setText(_translate("Frame", "Count"))
        self.label_10.setText(_translate("Frame", "Drink"))
        self.label_11.setText(_translate("Frame", "Wave"))
        self.label_12.setText(_translate("Frame", "Fall"))
        self.label_13.setText(_translate("Frame", "Cough"))
        self.label_14.setText(_translate("Frame", "Convulsion"))

        # self.retranslateUi()
        # QtCore.QMetaObject.connectSlotsByName(Frame)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.retranslateUi)
        self.timer.start()
    
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        zero = "0"
        self.time = []
        self.hr = []
        self.br = []
        # self.canvas1.axes.clear()
        # self.canvas2.axes.clear()

        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)

        font2 = QtGui.QFont()
        font2.setFamily("Microsoft JhengHei")
        font2.setPointSize(30)
        font2.setBold(True)
        font2.setWeight(75)

        self.label.setPixmap(QtGui.QPixmap("../activity_doppler.png"))
        self.label_6.setPixmap(QtGui.QPixmap("../activity_range.png"))
        with open('../output.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for i in reader:
                self.time.append(int(i[0]))
                if (int(i[1]) == 0):
                    self.label_26.setPixmap(QtGui.QPixmap("./static.png"))
                elif (int(i[1]) == 1):
                    self.label_26.setPixmap(QtGui.QPixmap("./moving.png"))

                self.label_22.setText(_translate("Frame", str(i[3][:2])))
                self.label_23.setText(_translate("Frame", str(i[4][:2])))
                self.hr.append(int(i[3][:2]))
                self.br.append(int(i[4][:2]))

                self.label_15.setText(_translate("Frame", zero))
                self.label_16.setText(_translate("Frame", zero))
                self.label_17.setText(_translate("Frame", zero))
                self.label_18.setText(_translate("Frame", zero))
                self.label_19.setText(_translate("Frame", zero))
                if (int(i[2]) == 2):
                    self.label_15.setText(_translate("Frame", "1"))
                    self.label_15.setStyleSheet("color:red")
                    self.label_15.setFont(font)
                    self.label_10.setStyleSheet("color:red")
                    self.label_10.setFont(font)
                    self.label_20.setPixmap(QtGui.QPixmap("./drink.png"))
                elif (int(i[2]) == 4):
                    self.label_16.setText(_translate("Frame", "1"))
                    self.label_16.setStyleSheet("color:red")
                    self.label_16.setFont(font)
                    self.label_11.setStyleSheet("color:red")
                    self.label_11.setFont(font)
                    self.label_20.setPixmap(QtGui.QPixmap("./wave.png"))
                elif (int(i[2]) == 3):
                    self.label_17.setText(_translate("Frame", "1"))
                    self.label_17.setStyleSheet("color:red")
                    self.label_17.setFont(font)
                    self.label_12.setStyleSheet("color:red")
                    self.label_12.setFont(font)
                    self.label_20.setPixmap(QtGui.QPixmap("./fall.png"))
                elif (int(i[2]) == 1):
                    self.label_18.setText(_translate("Frame", "1"))
                    self.label_18.setStyleSheet("color:red")
                    self.label_18.setFont(font)
                    self.label_13.setStyleSheet("color:red")
                    self.label_13.setFont(font)
                    self.label_20.setPixmap(QtGui.QPixmap("./cough.png"))
                elif (int(i[2]) == 0):
                    self.label_19.setText(_translate("Frame", "1"))
                    self.label_19.setStyleSheet("color:red")
                    self.label_19.setFont(font)
                    self.label_14.setStyleSheet("color:red")
                    self.label_14.setFont(font)
                    self.label_20.setPixmap(QtGui.QPixmap("./convulsion.png"))

        self.canvas1.axes.plot(self.time, self.hr, color='blue')
        self.canvas1.draw()
        self.canvas1.update()
        self.canvas2.axes.plot(self.time, self.br, color='orange')
        self.canvas2.draw()
        self.canvas2.update()


if __name__ == "__main__":
    os.system("del ..\\activity_doppler.png")
    os.system("del ..\\activity_range.png")
    app = QtWidgets.QApplication(sys.argv)
    Frame = QtWidgets.QFrame()
    ui = Ui_Frame()
    ui.setupUi(Frame)
    Frame.show()
    if (app.exec_() == 0):
        os.system("del ..\output.csv")
        sys.exit()
