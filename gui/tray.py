import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem

from gui.task import Ui_mainWindow


class TrayWindowMain(object):
    def __init__(self, paddle):
        self.paddle = paddle
        app = QtWidgets.QApplication(sys.argv)
        self.main_window = QtWidgets.QMainWindow()
        self.ui_window = Ui_mainWindow()
        self.ui_window.setupUi(self.main_window)
        self.ui_window.open.triggered.connect(self.open_files)
        self.main_window.show()
        sys.exit(app.exec_())

    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self.main_window, '打开文件', '.', '图像文件(*.jpg *.png)')
        if len(files) > 0:
            for i in range(len(files)):
                file = files[i]
                self.ui_window.tableWidget.insertRow(i)
                self.ui_window.tableWidget.setItem(i, 0, QTableWidgetItem(file))
