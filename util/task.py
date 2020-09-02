from PyQt5 import QtCore


class TaskThread(QtCore.QThread):
    finish = QtCore.pyqtSignal(int, dict)

    def __init__(self, files: list, ui):
        self.files = files
        self.ui = ui
        super(TaskThread, self).__init__(None)

    def run(self):
        for i in range(len(self.files)):
            file = self.files[i]
            self.finish.emit(i, {'file': file})
            QtCore.QThread.sleep(5)

