from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5 import uic


from main import Ui_MainWindow
import cv2
import sys

def hhmmss(ms):
    # s = 1000
    # m = 60000
    # h = 360000
    h, r = divmod(ms, 36000)
    m, r = divmod(r, 60000)
    s, _ = divmod(r, 1000)
    return ("%d:%02d:%02d" % (h,m,s)) if h else ("%d:%02d" % (m,s))


class MyWidget(QWidget):
    def __init__(self, parent=None):
        super(MyWidget, self).__init__(parent)
        
        self.ui = uic.loadUi('test.ui')
        self.ui._scene = QGraphicsScene(self)
        self.ui._gv = QGraphicsView(self.ui._scene)

        #self._videoitem = QGraphicsVideoItem()
        self.ui.widget = QGraphicsVideoItem()
        self.ui._scene.addItem(self.ui.widget)
        self.ui._ellipse_item = QGraphicsEllipseItem(QRectF(50, 50, 40, 40), self.ui.widget)
        self.ui._ellipse_item.setBrush(QBrush(Qt.green))
        self.ui._ellipse_item.setPen(QPen(Qt.red))
        
        self.fps = None
        self.frame = None

        #self._player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        #self._player.stateChanged.connect(self.on_stateChanged)
        #self._player.setVideoOutput(self._videoitem)
        self._player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        #videoWidget = self.ui.widget
        self._player.stateChanged.connect(self.on_stateChanged)
        self._player.setVideoOutput(self.ui.widget)
        #file = os.path.join(os.path.dirname(__file__), "P1230175.mp4")
        self.ui.open_file_action.triggered.connect(self.open_file)
        
        self.ui.playButton.pressed.connect(self._player.play)
        self.ui.pauseButton.pressed.connect(self._player.pause)
        self.ui.volumeSlider.valueChanged.connect(self._player.setVolume)
        self.ui.timeSlider.valueChanged.connect(self._player.setPosition)
        
        self._player.durationChanged.connect(self.update_duration)
        self._player.positionChanged.connect(self.update_position)
        self._player.positionChanged.connect(self.update_frame_num)
        self.ui.timeSlider.valueChanged.connect(self._player.setPosition)

        #lay = self.ui.centralWidget
        #lay = QVBoxLayout(self)
        self.ui.verticalLayout_2.addWidget(self.ui._gv)
        #lay.addWidget(self.ui._gv)
        self.ui.show()
     
    def open_file(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open file", "", "mp3 Audio (*.mp3);mp4 Video (*.mp4);Movie files (*.mov);All files (*.*)")
        QDir.homePath()
        
        video = cv2.VideoCapture(fileName)
        fps = video.get(cv2.CAP_PROP_FPS)
        self.fps = fps
        print(f"self.fps {self.fps}")
        
        if fileName != " ":
            self._player.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.ui.playButton.setEnabled(True)
 
        else:
            self.ui.playButton.setEnabled(True)
            
    def update_duration(self, duration):
        print("!", duration)
        print("?", self._player.duration())
        
        self.ui.timeSlider.setMaximum(duration)

        if duration >= 0:
            self.ui.totalTimeLabel.setText(hhmmss(duration))

    def update_position(self, position):
        if position >= 0:
            self.ui.currentTimeLabel.setText(hhmmss(position))

        # Disable the events to prevent updating triggering a setPosition event (can cause stuttering).
        self.ui.timeSlider.blockSignals(True)
        self.ui.timeSlider.setValue(position)
        self.ui.timeSlider.blockSignals(False)
        
    def update_frame_num(self, position):
        if position >= 0 and self.fps != None:
            self.frame = position/1000 * (self.fps)
        print('pos', position, 'current frame:', self.frame)
        
    @pyqtSlot(QMediaPlayer.State)
    def on_stateChanged(self, state):
        if state == QMediaPlayer.PlayingState:
            self.ui._gv.fitInView(self.ui.widget, Qt.KeepAspectRatio)
                
'''    
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        self.ui = uic.loadUi('video_player.ui')
        self.player = QMediaPlayer()
        self.player.play()
        self.fps = None
        self.frame = None
        print('current frame:', self.frame)
        videoWidget = self.ui.widget
        self.player.setVideoOutput(videoWidget)

        # Connect control buttons/slides for media player.
        self.ui.playButton.pressed.connect(self.player.play)
        self.ui.pauseButton.pressed.connect(self.player.pause)
        self.ui.volumeSlider.valueChanged.connect(self.player.setVolume)
        self.ui.timeSlider.valueChanged.connect(self.player.setPosition)
        
        self.player.durationChanged.connect(self.update_duration)
        self.player.positionChanged.connect(self.update_position)
        self.ui.timeSlider.valueChanged.connect(self.player.setPosition)
        
        self.ui.open_file_action.triggered.connect(self.open_file)

        self.setAcceptDrops(True)
        self.ui.show()
 
    def open_file(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open file", "", "mp3 Audio (*.mp3);mp4 Video (*.mp4);Movie files (*.mov);All files (*.*)")
        QDir.homePath()
 
        if fileName != " ":
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.ui.playButton.setEnabled(True)
 
        else:
            self.ui.playButton.setEnabled(True)
            
    def update_duration(self, duration):
        print("!", duration)
        print("?", self.player.duration())
        
        self.ui.timeSlider.setMaximum(duration)

        if duration >= 0:
            self.ui.totalTimeLabel.setText(hhmmss(duration))

    def update_position(self, position):
        if position >= 0:
            self.ui.currentTimeLabel.setText(hhmmss(position))

        # Disable the events to prevent updating triggering a setPosition event (can cause stuttering).
        self.ui.timeSlider.blockSignals(True)
        self.ui.timeSlider.setValue(position)
        self.ui.timeSlider.blockSignals(False)
        
'''
if __name__ == '__main__':
    app = QApplication([])
    w = MyWidget()
    app.exec_()
