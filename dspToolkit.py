import soundfile as sf
import numpy as np
import pandas as pd
from scipy import fftpack
from scipy import signal
import pyaudio

class DspToolkit():
    def __init__(self):
        '''コンストラクタ '''

        # 時間波形関係の変数
        self.time_y = []
        self.time_y_original = []
        self.time_x = []
        self.sampling = 44100
        self.dt = 0
        self.length = 0
        self.record_time = 0

        # フーリエ変換で使う変数
        self.frame_size = 1024
        self.overlap = 0
        self.dbref = 2e-5
        self.averaging = 0
        self.fft_axis = []
        self.fft_mean = []
        self.fft_array = []
        self.acf = 0

        # フィルタ処理で使う変数
        self.lp_freq = 1000
        self.hp_freq = 1000
        self.bp_freq_low = 500
        self.bp_freq_high = 1000
        self.bs_freq_low = 500
        self.bs_freq_high = 1000
        self.attenuation_pass = 3
        self.attenuation_stop = 40

        # ピーク検出で使う変数
        self.num_peaks = 10
        self.find_peak_order = 6
        self.peak_index = []
        self.peaks = []
        self.peak_index_sort = []
        self.peaks_sort = []

    def open_wav(self, path):
        ''' パスを受け取ってwavファイルを読み込む '''

        self.time_y, self.sampling = sf.read(path)
        self.time_y_original = self.time_y.copy()
        self.get_time_information()

    def open_csv(self, path):
        ''' パスを受け取ってcsvファイルを読み込む '''

        df = pd.read_csv(path, encoding='SHIFT-JIS')
        self.time_y = df.T.iloc[1]
        self.time_y_original = self.time_y.copy()
        self.sampling = 1 / df.T.iloc[0, 1]
        self.get_time_information()

    def get_time_information(self):
        ''' Time plotの表示器に表示させる情報の計算と時間軸作成を行う '''

        self.dt = 1 / self.sampling
        self.time_x = np.arange(0, len(self.time_y), 1) * self.dt
        self.length = len(self.time_x) * self.dt
        print('Time waveform information was obtained.')

    def calc_overlap(self):
        ''' 時間波形をオーバーラップ率で切り出してリスト化する '''

        frame_cycle = self.frame_size / self.sampling
        x_ol = self.frame_size * (1 - (self.overlap / 100))
        self.averaging = int((self.length - (frame_cycle * (self.overlap / 100))) / (frame_cycle * (1 - (self.overlap / 100))))
        time_array = []
        final_time = 0
        if self.averaging != 0:
            for i in range(self.averaging):
                ps = int(x_ol * i)
                time_array.append(self.time_y[ps:ps+self.frame_size:1])
                final_time = (ps + self.frame_size) / self.sampling

            print('Frame size=', self.frame_size)
            print('Frame cycle=', frame_cycle)
            print('averaging=', self.averaging)
            return time_array, final_time
        return time_array, final_time

    def hanning(self, time_array):
        ''' ハニング窓をかけ振幅補正係数ACFを計算する '''

        han = signal.hann(self.frame_size)
        self.acf = 1 / (sum(han) / self.frame_size)

        # オーバーラップされた複数時間波形全てに窓関数をかける
        for i in range(self.averaging):
            time_array[i] = time_array[i] * han
        return time_array

    def fft(self, time_array):
        ''' 平均化フーリエ変換をする '''

        fft_array = []
        for i in range(self.averaging):
            # FFTをして配列に追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。
            fft_array.append(self.acf * np.abs(fftpack.fft(np.array(time_array[i])) / (self.frame_size / 2)))

        # 全てのFFT波形のパワー平均を計算してから振幅値とする。
        self.fft_axis = np.linspace(0, self.sampling, self.frame_size)
        self.fft_array = np.array(fft_array)
        self.fft_mean = np.sqrt(np.mean(self.fft_array ** 2, axis=0))

    def db(self, x, dBref):
        ''' dB変換をする '''

        y = 20 * np.log10(x / dBref)

        return y

    def idb(self, x, dBref):
        ''' dB逆変換をする '''

        y = dBref * np.power(10, x / 20)

        return y

    def aweightings(self, f):
        ''' A補正（聴感補正）曲線の計算 '''

        if f[0] == 0:
            f[0] = 1e-6

        ra = (np.power(12194, 2) * np.power(f, 4)) / \
             ((np.power(f, 2) + np.power(20.6, 2)) *
              np.sqrt((np.power(f, 2) + np.power(107.7, 2)) * (np.power(f, 2) + np.power(737.9, 2))) *
              (np.power(f, 2) + np.power(12194, 2)))
        a = 20 * np.log10(ra) + 2.00

        return a

    def filter(self, filter_type):
        ''' フィルタをかける '''

        # ナイキスト周波数fnを設定して通過域端周波数wpと阻止域端周波数wsを正規化
        # 阻止域周波数は通過域周波数の2倍にしている仕様（ここは目的に応じて変えても良い）
        fn = self.sampling / 2

        # Lowpassフィルタ
        if filter_type == 'low':
            wp = self.lp_freq / fn
            ws = (self.lp_freq * 2) / fn

            try:
                # フィルタ次数とバタワース正規化周波数を計算
                N, Wn = signal.buttord(wp, ws, self.attenuation_pass, self.attenuation_stop)

                # フィルタ伝達関数の分子と分母を計算
                b, a = signal.butter(N, Wn, filter_type)
                self.time_y = signal.filtfilt(b, a, self.time_y)
            except ValueError:
                return -1

        # Highpassフィルタ
        if filter_type == 'high':
            wp = self.hp_freq / fn
            ws = (self.hp_freq * 2) / fn

            try:
                # フィルタ次数とバタワース正規化周波数を計算
                N, Wn = signal.buttord(wp, ws, self.attenuation_pass, self.attenuation_stop)

                # フィルタ伝達関数の分子と分母を計算
                b, a = signal.butter(N, Wn, filter_type)
                self.time_y = signal.filtfilt(b, a, self.time_y)
            except ValueError:
                return -1

        # bandpassフィルタ
        if filter_type == 'band':
            fp = np.array([float(self.bp_freq_low), float(self.bp_freq_high)])
            fs = np.array([float(self.bp_freq_low)/2, float(self.bp_freq_high)*2])

            wp = fp / fn
            ws = fs / fn

            try:
                # フィルタ次数とバタワース正規化周波数を計算
                N, Wn = signal.buttord(wp, ws, self.attenuation_pass, self.attenuation_stop)

                # フィルタ伝達関数の分子と分母を計算
                b, a = signal.butter(N, Wn, filter_type)
                self.time_y = signal.filtfilt(b, a, self.time_y)
            except ValueError:
                return -1

        # bandstopフィルタ
        if filter_type == 'bandstop':
            fp = np.array([float(self.bs_freq_low), float(self.bs_freq_high)])
            fs = np.array([float(self.bs_freq_low) / 2, float(self.bs_freq_high) * 2])

            wp = fp / fn
            ws = fs / fn

            try:
                # フィルタ次数とバタワース正規化周波数を計算
                N, Wn = signal.buttord(wp, ws, self.attenuation_pass, self.attenuation_stop)

                # フィルタ伝達関数の分子と分母を計算
                b, a = signal.butter(N, Wn, filter_type)
                self.time_y = signal.filtfilt(b, a, self.time_y)
            except ValueError:
                return -1
        return 0

    def get_mic_index(self):
        ''' マイクチャンネルのindexをリストで取得する '''

        # 最大入力チャンネル数が0でない項目をマイクチャンネルとしてリストに追加
        pa = pyaudio.PyAudio()
        mic_list = []
        for i in range(pa.get_device_count()):
            num_of_input_ch = pa.get_device_info_by_index(i)['maxInputChannels']

            if num_of_input_ch != 0:
                mic_list.append(pa.get_device_info_by_index(i)['index'])

        return mic_list

    def record(self, index, samplerate, fs, time):
        ''' 録音する関数 '''

        pa = pyaudio.PyAudio()

        # ストリームの開始
        data = []
        dt = 1 / samplerate
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=samplerate,
                         input=True, input_device_index=index, frames_per_buffer=fs)

        # フレームサイズ毎に音声を録音していくループ
        for i in range(int(((time / dt) / fs))):
            frame = stream.read(fs)
            data.append(frame)

        # ストリームの終了
        stream.stop_stream()
        stream.close()
        pa.terminate()

        # データをまとめる処理
        data = b"".join(data)

        # データをNumpy配列に変換/時間軸を作成
        data = np.frombuffer(data, dtype="int16") / float((np.power(2, 16) / 2) - 1)
        t = np.arange(0, fs * (i + 1) * (1 / samplerate), 1 / samplerate)

        return data, t

    def findpeaks(self, x, y, n, w):
        ''' 周波数波形の振幅成分からPeakを検出する '''

        # ピーク検出
        index_all = list(signal.argrelmax(y, order=w))
        self.peak_index = []
        self.peaks = []

        # n個分のピーク情報(指標、値）を格納
        for i in range(n):
            # n個のピークに満たない場合は途中でループを抜ける（エラー処理）
            if i >= len(index_all[0]):
                break
            self.peak_index.append(index_all[0][i])
            self.peaks.append(y[index_all[0][i]])

        # 個数の足りない分を0で埋める（エラー処理）
        if len(self.peak_index) != n:
            self.peak_index = self.peak_index + ([0] * (n - len(self.peak_index)))
            self.peaks = self.peaks + ([0] * (n - len(self.peaks)))

        # xの分解能x[1]をかけて指標を物理軸に変換
        self.peak_index = np.array(self.peak_index) * x[1]

        # ピークを大きい順（降順）にソートする
        self.peaks_sort = np.sort(self.peaks)[::-1]
        self.peak_index_sort = self.peak_index[np.argsort(self.peaks)[::-1]]

        return