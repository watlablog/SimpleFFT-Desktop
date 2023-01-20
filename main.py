from kivy.app import App
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen, FallOutTransition
from matplotlib import pyplot as plt
from dspToolkit import DspToolkit
import numpy as np
from kivy.clock import Clock
import os

class Root(Screen):
    ''' Pythonで計算をした結果をmatplotlibでプロットする '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ''' コンストラクタ:プロットの初期化とDspToolkitのインスタンス化を行う '''

        # DspToolkitをインスタンス化する
        self.dsp = DspToolkit()

        # プロットの初期化
        x = []
        y = []
        self.fig = self.plot(x, y)

        # ウィジェットとしてfigをラップする
        self.ids.viewer.add_widget(FigureCanvasKivyAgg(self.fig))

        # 計測時間を時間波形の最大表示時間初期値とする
        self.ids.max_time.text = self.ids.record_time.text

        # 検証用（widgetの位置や状態の確認）
        #print(self.children[0].children[2])
        #print(self.children[0].children[2].children[1])
        #print(self.ids.display_mode.state)


    def on_button(self):
        ''' ボタンがクリックされた時のイベント：波形を更新する '''

        # TextInputからの入力値を引用する
        try:
            self.dsp.record_time = float(self.ids.record_time.text)
            self.dsp.frame_size = int(self.ids.frame_size.text)
        except ValueError:
            print('record time or frame size could not convert to type of number.')
            return

        # 測定時間が指定範囲外であれば測定をしない
        min_time = 0
        max_time = 60
        if self.dsp.record_time > min_time and self.dsp.record_time <= max_time:
            pass
        else:
            print('Error:Record time is out of range!')
            return

        # フレームサイズ範囲外であれば測定をしない
        min_frame_size = 0
        max_frame_size = 10000
        if self.dsp.frame_size > min_frame_size and self.dsp.frame_size <= max_frame_size:
            pass
        else:
            print('Error:Frame size is out of range!')
            return

        # マイクチャンネルを自動で取得（最初のマイクを使用する）
        mic_ch = self.dsp.get_mic_index()[0]

        # 録音する
        print('Record time[s]=', self.dsp.record_time)
        print('Sampling rate[Hz]=', self.dsp.sampling)
        print('Frame size=', self.dsp.frame_size)
        self.dsp.time_y, self.dsp.time_x = self.dsp.record(mic_ch, self.dsp.sampling, self.dsp.frame_size, self.dsp.record_time)

        # フーリエ変換する
        self.dsp.dt = self.dsp.time_x[1]
        self.dsp.length = len(self.dsp.time_x) * self.dsp.dt
        self.fft()

        # 再描画
        self.reflesh_plot()

        return

    def reflesh_plot(self):
        ''' 再描画を行う '''

        # RuntimeWarningを防ぐ：メモリの効率化用。更新時にこれを入れないとメモリWarningがでる。
        plt.close()

        # figの再作成
        if self.ids.display_mode.state == 'down':
            # FFT waveform
            self.fig = self.plot(self.dsp.fft_axis, self.dsp.fft_mean)

        else:
            # Time waveform
            self.fig = self.plot(self.dsp.time_x, self.dsp.time_y)


        # 値の更新（ウィジェットを削除して新しく追加している。ボタンも一緒に消して再度addしている。なんかもっと良い方法が？？）
        # PageLayoutの構成を変更した場合はchildrenの設定を変更する
        temp_button = self.children[0].children[2].children[1]
        self.children[0].children[2].clear_widgets()
        self.ids.viewer.add_widget(temp_button)
        self.ids.viewer.add_widget(FigureCanvasKivyAgg(self.fig))

        return

    def change_record_time(self):
        ''' Record timeが変更された時のイベント:時間軸の範囲をセットしなおす '''

        self.ids.min_time.text = '0'
        self.ids.max_time.text = self.ids.record_time.text

    def change_axis(self):
        ''' 軸設定が変更された時のイベント:無効値のチェックとプロット再描画 '''

        if self.ids.display_mode.state == 'down':
            # Display mode = FFT waveform
            try:
                min_freq = float(self.ids.min_freq.text)
                max_freq = float(self.ids.max_freq.text)
            except ValueError:
                print('There are not float number in axis settings.')
                return

            if max_freq == 0:
                print('max_freq is zero.')
                return

            if min_freq >= max_freq:
                print('min_freq is larger than max_freq.')
                return
        else:
            # Display mode = Time waveform
            try:
                min_time = float(self.ids.min_time.text)
                max_time = float(self.ids.max_time.text)
            except ValueError:
                print('There are not float number in axis settings.')
                return

            if max_time == 0:
                print('max_time is zero.')
                return

            if min_time >= max_time:
                print('min_time is larger than max_time.')
                return

        # 再描画
        self.reflesh_plot()

        return

    def change_display_mode(self):
        ''' Display modeを変更した時のイベント:テキストの切替と再描画 '''

        if self.ids.display_mode.state == 'down':
            self.ids.display_mode.text = 'FFT waveform'
        else:
            self.ids.display_mode.text = 'Time waveform'
        print(self.ids.display_mode.state)

        # 再描画
        self.reflesh_plot()

        return

    def change_peak_info(self):
        ''' Peak infoボタンを変更した時のイベント:テキストの切替と再描画 '''

        if self.ids.peak_info.state == 'down':
            self.ids.peak_info.text = 'ON'
        else:
            self.ids.peak_info.text = 'OFF'
        print(self.ids.peak_info.state)

        # 再描画
        self.reflesh_plot()

        return

    def fft(self):
        ''' フーリエ変換 '''

        # TextInputからの入力値を引用する
        try:
            self.dsp.overlap = float(self.ids.overlap.text)
        except ValueError:
            print('Overlap value could not convert to type of number.')
            return

        # 時間波形をオーバーラップ処理する。
        time_array, final_time = self.dsp.calc_overlap()


        # オーバーラップができない場合は処理を終わらせる。
        if len(time_array[0]) == 0:
            print('Overlap calculation was ignored. Frame size was larger than data length.')
            return False

        # オーバーラップ処理した波形リストに窓間数をかける。
        time_array = self.dsp.hanning(time_array)

        # 平均化フーリエ変化をする。
        self.dsp.fft(time_array)

        # dB変換する（dBrefは2e-5固定）。
        self.dsp.fft_mean = self.dsp.db(self.dsp.fft_mean, self.dsp.dbref)

        # A特性をかける。
        self.dsp.fft_mean = self.dsp.fft_mean + self.dsp.aweightings(self.dsp.fft_axis)

        # ピーク検出をする。
        self.dsp.findpeaks(self.dsp.fft_axis, self.dsp.fft_mean, self.dsp.num_peaks, self.dsp.find_peak_order)

        return True



    def plot(self, x, y):
        ''' プロットする共通のメソッド '''

        # フォントの種類とサイズを設定する。
        plt.rcParams['font.size'] = 20
        plt.rcParams['font.family'] = 'Times New Roman'

        # 目盛を内側にする。
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        # Subplot設定とグラフの上下左右に目盛線を付ける。
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.yaxis.set_ticks_position('both')
        self.ax1.xaxis.set_ticks_position('both')

        # スケールを設定する。TextInputから値を参照。
        try:
            min_freq = float(self.ids.min_freq.text)
            max_freq = float(self.ids.max_freq.text)
            min_time = float(self.ids.min_time.text)
            max_time = float(self.ids.max_time.text)
        except ValueError:
            print('Axis region value could not converted to type of number.')
            print('Process was ignored.')


        if self.ids.display_mode.state == 'down':
            # FFT waveform
            self.ax1.set_xlim(min_freq, max_freq)

            # 軸のラベルを設定する。
            self.ax1.set_xlabel('Frequency[s]')
            self.ax1.set_ylabel('Amplitude[dBA]')

        else:
            # Time waveform
            self.ax1.set_xlim(min_time, max_time)

            # 軸のラベルを設定する。
            self.ax1.set_xlabel('Time[s]')
            self.ax1.set_ylabel('Amplitude')


        # 縦軸設定
        if len(self.dsp.fft_mean) == 0:
            min_amp = 0
            max_amp = 100
        else:
            min_amp = np.mean(self.dsp.fft_mean) - 10
            max_amp = np.max(self.dsp.fft_mean) + 20
        if self.ids.display_mode.state == 'down':
            # FFT waveform
            if len(self.dsp.fft_mean) != 0:
                self.ax1.set_ylim(min_amp, max_amp)
            #self.ax1.set_yscale('log')
        else:
            # Time waveform
            pass

        # 波形のプロット
        self.ax1.plot(x, y, lw=1)

        # Peak infoボタンがONの時だけピークをプロット
        if self.ids.peak_info.state == 'down':

            # 周波数波形の場合はピークのプロット（重ね書き）
            if self.ids.display_mode.state == 'down':
                # FFT waveform
                n = len(self.dsp.peak_index_sort)
                for i in range(n):
                    self.ax1.scatter(self.dsp.peak_index_sort[n - i - 1], self.dsp.peaks_sort[n - i - 1],
                                     color='yellow', edgecolor='black',
                                     s=100)
                    self.ax1.text(min_freq + 100, min_amp + (i + 2) * 4,
                                  '(' + str(np.round(self.dsp.peak_index_sort[n - i - 1], 1)) + ', ' + str(np.round(self.dsp.peaks_sort[n - i - 1], 1)) + ')',
                                  fontsize=14)

        # レイアウト設定
        self.fig.tight_layout()

        return self.fig

class StartingScreen(Screen):
    ''' 起動時に読み込まれるページ '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ids.logo.source = path_icon


class LicenseScreen(Screen):
    ''' ファイルを読み込みLicense文を表示させるページ '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        path_notices = './notices.txt'
        path_notices = os.path.join(os.path.dirname(__file__), path_notices)


        with open(path_notices, 'rt') as f:
            lic = f.read()
        self.ids.lic_document.text = lic
        self.ids.lic_document.font_size = int(self.ids.lic_document.text_size[0] / 5)
        pass


class SimpleFFT(App):
    title = 'SimpleFFT ver.1.0.0'

    global path_icon

    path_icon = './icon.png'
    path_icon = os.path.join(os.path.dirname(__file__), path_icon)

    # デスクトップ環境上でモバイル画面の比率を検証するためにサイズ設定をしている（正式には外す）
    ratio = 8 / 12
    w = 800
    Window.size = (w, w * ratio)
    Window.top = 50

    def build(self):
        global sm
        sm = ScreenManager(transition=FallOutTransition())
        sm.add_widget(Root(name='main'))
        sm.add_widget(LicenseScreen(name='license'))
        sm.add_widget(StartingScreen(name='start'))
        sm.children[0].manager.current = 'start'
        return sm

    def on_start(self):
        Clock.schedule_once(self.login, 8)

    def login(*args):
        sm.current = 'main'


if __name__ == '__main__':
    SimpleFFT().run()