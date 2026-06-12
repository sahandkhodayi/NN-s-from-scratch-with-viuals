# gui/canvas.py

from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView
from node_item import NeuronNode


class NetworkCanvas(QGraphicsView):

    def __init__(self, network):

        self.scene = QGraphicsScene()

        super().__init__(self.scene)

        self.network = network

        self.draw_network()


    def draw_network(self):

        y = 0

        for layer in self.network.layers:

            x = 0

            for neuron in layer.main_nodes:

                node = NeuronNode(
                    neuron,
                    x,
                    y
                )

                self.scene.addItem(node)

                x += 100

            y += 150