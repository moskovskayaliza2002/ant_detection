import os
import time

from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets
#import tobii_research as tr
import numpy as np


class EyeTracker(QtCore.QObject):
    positionChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, tracker, parent=None):
        super(EyeTracker, self).__init__(parent)
        self._tracker = tracker

    @property
    def tracker(self):
        return self._tracker

    def start(self):
        #self.tracker.subscribe_to(
            #tr.EYETRACKER_GAZE_DATA, self._callback, as_dictionary=True
        #)
        pass

    def _callback(self, gaze_data_):
        self.positionChanged.emit(
            float(np.random.choice(range(0, 300))),
            float(np.random.choice(range(0, 240))),
        )
        print("time.time()::{}".format(time.time()))


class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):

        super(Widget, self).__init__(parent)

        # first window,just have a single button for play the video
        self.resize(256, 256)
        self.btn_play = QtWidgets.QPushButton(self)
        self.btn_play.setGeometry(QtCore.QRect(100, 100, 28, 28))
        self.btn_play.setObjectName("btn_open")
        self.btn_play.setText("Play")
        self.btn_play.clicked.connect(self.Play_video)  # click to play video
        #

        self._scene = QtWidgets.QGraphicsScene(self)
        self._gv = QtWidgets.QGraphicsView(self._scene)
        # construct a videoitem for showing the video
        self._videoitem = QtMultimediaWidgets.QGraphicsVideoItem()
        # add it into the scene
        self._scene.addItem(self._videoitem)

        # assign _ellipse_item is the gaze data, and embed it into videoitem,so it can show above the video.
        self._ellipse_item = QtWidgets.QGraphicsEllipseItem(
            QtCore.QRectF(0, 0, 40, 40), self._videoitem
        )
        self._ellipse_item.setBrush(QtGui.QBrush(QtCore.Qt.black))
        self._ellipse_item.setPen(QtGui.QPen(QtCore.Qt.red))
        # self._scene.addItem(self._ellipse_item)
        self._gv.fitInView(self._videoitem)

        self._player = QtMultimedia.QMediaPlayer(
            self, QtMultimedia.QMediaPlayer.VideoSurface
        )
        self._player.setVideoOutput(self._videoitem)
        #file = os.path.join(
        #    os.path.dirname(__file__), "P1230175.MP4"
        #)  # video.mp4 is under the same dirctory
        file = "/home/ubuntu/ant_detection/qt_designer/P1230175.MP4"
        self._player.setMedia(
            QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(file))
        )
        print(f"self._videoitem::{self._videoitem.size()}")

        # get eye tracker
        #eyetrackers = tr.find_all_eyetrackers()
        self.tracker = EyeTracker([])
        self.tracker.positionChanged.connect(self._ellipse_item.setPos)

    def Play_video(self):
        self.tracker.start()
        # size = QtCore.QSizeF(1920.0, 1080.0)#I hope it can fullscreen the video
        # self._videoitem.setSize(size)
        # self._gv.showFullScreen()
        self._gv.resize(720, 720)
        self._gv.show()
        self._player.play()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())
