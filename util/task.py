from PyQt5 import QtCore

from util.ocr import Ocr


class TaskThread(QtCore.QThread):
    identity = QtCore.pyqtSignal(int, dict)

    def __init__(self, files: list):
        super(TaskThread, self).__init__()
        self.files = files
        self.ocr = Ocr()

    def run(self):
        for i in range(len(self.files)):
            self.identity.emit(i, self.ocr.main(self.files[i]))

