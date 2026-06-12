# gui/node_item.py

from PyQt6.QtWidgets import QGraphicsEllipseItem
from PyQt6.QtGui import QBrush


class NeuronNode(QGraphicsEllipseItem):

    def __init__(self, neuron, x, y):
        super().__init__(0,0,50,50)

        self.neuron = neuron

        self.setPos(x,y)
        self.setBrush(QBrush())

        self.setFlag(
            QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable
        )