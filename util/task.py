from PyQt5 import QtCore


class TaskThread(QtCore.QThread):
    identity = QtCore.pyqtSignal(int, dict)

    def __init__(self, queue):
        super(TaskThread, self).__init__()
        self.queue = queue

    def run(self):
        while True:
            if not self.queue.empty():
                a = self.queue.get()
                print(a)

            self.sleep(1)

