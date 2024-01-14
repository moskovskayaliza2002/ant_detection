import os
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets

class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent)

        self._scene = QtWidgets.QGraphicsScene(self)
        self._gv = QtWidgets.QGraphicsView(self._scene)

        self._videoitem = QtMultimediaWidgets.QGraphicsVideoItem()
        self._scene.addItem(self._videoitem)
        self._ellipse_item = QtWidgets.QGraphicsEllipseItem(QtCore.QRectF(50, 50, 40, 40), self._videoitem)
        self._ellipse_item.setBrush(QtGui.QBrush(QtCore.Qt.green))
        self._ellipse_item.setPen(QtGui.QPen(QtCore.Qt.red))

        self._player = QtMultimedia.QMediaPlayer(self, QtMultimedia.QMediaPlayer.VideoSurface)
        self._player.stateChanged.connect(self.on_stateChanged)
        self._player.setVideoOutput(self._videoitem)

        #file = os.path.join(os.path.dirname(__file__), "P1230175.mp4")
        file = "/home/ubuntu/ant_detection/qt_designer/P1230175.MP4"
        self._player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(file)))
        button = QtWidgets.QPushButton("Play")
        button.clicked.connect(self._player.play)

        self.resize(640, 480)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self._gv)
        lay.addWidget(button)

    @QtCore.pyqtSlot(QtMultimedia.QMediaPlayer.State)
    def on_stateChanged(self, state):
        if state == QtMultimedia.QMediaPlayer.PlayingState:
            self._gv.fitInView(self._videoitem, QtCore.Qt.KeepAspectRatio)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())
