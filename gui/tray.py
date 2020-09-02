import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem

from gui.task import Ui_mainWindow


class TrayWindowMain(object):
    def __init__(self, ocr):
        app = QtWidgets.QApplication(sys.argv)
        self.main_window = QtWidgets.QMainWindow()
        self.ui_window = Ui_mainWindow()
        self.ui_window.setupUi(self.main_window)
        self.ui_window.open.triggered.connect(self.open_files)
        self.main_window.show()
        self.ocr = ocr
        # 启动信号槽线程
        # self.queue = result_queue
        # self.file_queue = file_queue
        # self.task_thread = TaskThread(self.queue)
        # self.task_thread.start()
        # self.task_thread.identity.connect(self.change_column)

        sys.exit(app.exec_())

    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self.main_window, '打开文件', '.', '图像文件(*.jpg *.png *.jpeg)')
        if len(files) > 0:
            for i in range(len(files)):
                self.ui_window.tableWidget.insertRow(i)
                self.ui_window.tableWidget.setItem(i, 11, QTableWidgetItem('识别中'))

            for i in range(len(files)):
                self.change_column(i, self.ocr.main(files[i]))

    def change_column(self, i, result):
        self.ui_window.tableWidget.setItem(i, 0, QTableWidgetItem(result.get('name')))
        self.ui_window.tableWidget.setItem(i, 1, QTableWidgetItem('身份证'))
        self.ui_window.tableWidget.setItem(i, 2, QTableWidgetItem(result.get('card')))
        self.ui_window.tableWidget.setItem(i, 3, QTableWidgetItem(result.get('effective_date')))
        self.ui_window.tableWidget.setItem(i, 4, QTableWidgetItem(result.get('expire_date')))
        self.ui_window.tableWidget.setItem(i, 5, QTableWidgetItem(result.get('address')))
        self.ui_window.tableWidget.setItem(i, 6, QTableWidgetItem(result.get('gender')))
        self.ui_window.tableWidget.setItem(i, 7, QTableWidgetItem('中国'))
        self.ui_window.tableWidget.setItem(i, 8, QTableWidgetItem('公司员工'))
        self.ui_window.tableWidget.setItem(i, 9, QTableWidgetItem('0'))
        self.ui_window.tableWidget.setItem(i, 10, QTableWidgetItem(''))
        self.ui_window.tableWidget.setItem(i, 11, QTableWidgetItem('已完成'))

