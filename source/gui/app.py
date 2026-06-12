# gui/app.py

from PyQt6.QtWidgets import QApplication, QMainWindow
from canvas import NetworkCanvas


class MainWindow(QMainWindow):

    def __init__(self, network):
        super().__init__()

        self.network = network

        self.canvas = NetworkCanvas(network)

        self.setCentralWidget(self.canvas)
        self.setWindowTitle("Neural Network Playground")
        self.resize(1200,800)


app = QApplication([])

# later:
# from network import Network
 net = Network(...)

window = MainWindow(net)
window.show()

app.exec()