import sys
import wave
import whisper
import numpy as np
import pyaudio
import torch
from PyQt5.QtCore import pyqtSlot, QDate, QTime, QDateTime, Qt, QThread, QTimer
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QWidget, QVBoxLayout, QPushButton
from PyQt5 import uic
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import torchaudio
from IPython.display import Audio
import librosa
import librosa.display
import noisereduce as nr
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph as pg

class AudioRecorder(QThread):
    def __init__(self):
        super().__init__()
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.is_recording = False
        self.frames = []

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)

        self.is_recording = True
        self.frames = []

        while self.is_recording:
            data = stream.read(self.chunk)
            self.frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop_recording(self):
        self.is_recording = False

    def save_to_wav(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()


class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super(WaveformWidget, self).__init__(parent)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

class SpectoformWidget(QWidget):
    def __init__(self, parent=None):
        super(SpectoformWidget, self).__init__(parent)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

def create_waveplot(data, sr, e, widget):
    widget.ax.clear()
    widget.ax.set_title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr, ax=widget.ax, color="blue")
    widget.canvas.draw()



def create_spectrogram(data, sr, e, widget):
    widget.ax.clear()
    widget.ax.set_title('Spectrogram for audio with {} emotion'.format(e), size=15)

    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    # plt.figure(figsize=(10, 3))
    # plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    widget.canvas.draw()




class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        uic.loadUi('untitled.ui', self)
        self.setWindowTitle('Работа с аудио')
        self.audio_recorder = AudioRecorder()
        self.stop_button.setEnabled(False)
        self.predicted_emotion = ''
        self.setup_ui()



    def setup_ui(self):
        self.start_button.clicked.connect(self.start_recording)

        self.stop_button.clicked.connect(self.stop_recording)

        self.pushButton_3.clicked.connect(self.transcription)

        self.pushButton_4.clicked.connect(self.prediction)

        self.wave_button.clicked.connect(self.wave)

        self.specto_button.clicked.connect(self.specto)

    def wave(self):
        self.wave_widget = WaveformWidget(self)
        self.setCentralWidget(self.wave_widget)

        emo = self.predicted_emotion
        data, sampling_rate = librosa.load('recorded_audio.wav')
        data = nr.reduce_noise(data, sr=sampling_rate)
        xt, index = librosa.effects.trim(data, top_db=33)

        #create_waveplot(xt, sampling_rate, emo, self.wave_widget)

        # Добавляем кнопку для сброса графика
        reset_button = QPushButton('Вернуться', self.wave_widget)
        reset_button.clicked.connect(self.reset_graph)
        self.layout().addWidget(reset_button)

        # Запоминаем оригинальные данные
        self.original_data = data
        self.original_sr = sampling_rate

        create_waveplot(xt, sampling_rate, emo, self.wave_widget)

    def specto(self):
        self.specto_widget = SpectoformWidget(self)
        self.setCentralWidget(self.specto_widget)

        emo = self.predicted_emotion
        data, sampling_rate = librosa.load('recorded_audio.wav')
        data = nr.reduce_noise(data, sr=sampling_rate)
        xt, index = librosa.effects.trim(data, top_db=33)

        #create_waveplot(xt, sampling_rate, emo, self.wave_widget)

        # Добавляем кнопку для сброса графика
        reset_button = QPushButton('Вернуться', self.specto_widget)
        reset_button.clicked.connect(self.reset_graph)
        self.layout().addWidget(reset_button)

        # Запоминаем оригинальные данные
        self.original_data = data
        self.original_sr = sampling_rate


        create_spectrogram(xt, sampling_rate, emo, self.specto_widget)

    def reset_graph(self):
        self.hide()
        self.ex = Main()
        self.ex.setStyleSheet("background-color: #00ffff")
        self.ex.show()

    def start_recording(self):
        self.audio_recorder.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_recording(self):
        self.audio_recorder.stop_recording()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        filename = 'recorded_audio.wav'
        self.audio_recorder.save_to_wav(filename)
        print(f"Recording saved as {filename}")

    def transcription(self):
        model = whisper.load_model('base')
        result = model.transcribe('recorded_audio.wav')
        print(result["text"])
        self.textEdit.setText(result["text"])

    def prediction(self):
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        model = HubertForSequenceClassification.from_pretrained(
            "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
        num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad'}

        filepath = 'recorded_audio.wav'

        waveform, sample_rate = torchaudio.load(filepath, normalize=True)
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = transform(waveform)

        inputs = feature_extractor(
            waveform,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
            max_length=16000 * 10,
            truncation=True
        )

        logits = model(inputs['input_values'][0]).logits
        predictions = torch.argmax(logits, dim=-1)

        self.predicted_emotion = num2emotion[predictions.numpy()[0]]

        self.textEdit_2.setText(self.predicted_emotion)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Main()
    widget.show()
    sys.exit(app.exec_())