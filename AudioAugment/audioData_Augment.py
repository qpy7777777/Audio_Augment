import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
from AudioAugment import spec_augment_pytorch
from python_speech_features import delta
import pyroomacoustics as pra


# 特征提取：mel谱，一阶差分、二阶差分重构的3-D特征；
def show_melspectrogram(audio,fs):
    melspec = librosa.feature.melspectrogram(y=audio, sr=fs, n_fft=1024, hop_length=512, n_mels=128)
    mfccs = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=40)
    feat_mel_d = delta(melspec, 2)
    feat_mel_dd = delta(feat_mel_d, 2)
    input_feature = torch.stack(
        (torch.from_numpy(melspec), torch.from_numpy(feat_mel_d), torch.from_numpy(feat_mel_dd)), dim=2).permute(2, 0,                                                                                                      1)
    #     print(melspec.shape,feat_mel_d.shape,feat_mel_dd.shape,input_feature.shape)#torch.Size([3, 78, 548])
    # Log-Mel Spectrogram特征是二维数组的形式，(78, 548)
    # 78表示Mel频率的维度（频域），548（时域），Log-Mel Spectrogram特征是音频信号的时频表示特征
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    librosa.display.waveplot(audio, sr=fs)
    #     plt.colorbar(format='%+2.0f dB')
    plt.title("time domain waveform")
    plt.subplot(2, 2, 2)
    melspec = librosa.power_to_db(melspec)
    librosa.display.specshow(melspec, sr=fs, x_axis='time', y_axis='hz')
    plt.title("mel-frequency spectrogram")
    plt.subplot(2, 2, 3)
    librosa.display.specshow(feat_mel_d, sr=fs, x_axis='time', y_axis='mel')
    plt.title("mel-frequency Delta")
    plt.subplot(2, 2, 4)
    librosa.display.specshow(feat_mel_dd, sr=fs, x_axis='time', y_axis='mel')
    #     librosa.display.specshow(mfccs,sr=sr,x_axis='time')
    plt.title("mel-frequency 2-Delta")
    plt.tight_layout()
    plt.show()

# 通过同时随机去除几行(帧)和几列(频率)来实现数据的增强
#对光谱图的修改方式有：沿着时间方向扭曲，遮蔽某一些频率段的信号，以及遮蔽某一些时间段的发音。
def show_time_frequeny_mask(audio,fs):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=fs,
                                                     n_mels=128,
                                                     hop_length=256,
                                                     fmax=8000)
    # reshape spectrogram shape to [batch_size, time, frequency]
    shape = mel_spectrogram.shape
    print(shape)
    mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1]))
    print(mel_spectrogram.shape)
    mel_spectrogram = torch.from_numpy(mel_spectrogram)

    # Show Raw mel-spectrogram
    spec_augment_pytorch.visualization_spectrogram(mel_spectrogram=mel_spectrogram,
                                                   title="Raw Mel Spectrogram")

    # Calculate SpecAugment pytorch
    warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram)

    # Show time warped & masked spectrogram
    spec_augment_pytorch.visualization_spectrogram(mel_spectrogram=warped_masked_spectrogram,
                                                   title="pytorch Warped & Masked Mel Spectrogram")
# 高斯噪声 加性噪声 标准差为1 均值为0
def add_noise(x,w):
    data_nosie = w * np.random.normal(loc=0, scale=1, size=len(x))
    return data_nosie
# 高斯白噪声
def wgn(x, SNR): # wgn是获得原始信号为x,相对于原始信号信噪比是snr dB的高斯噪声
    snr = 10**(SNR/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr # 获取噪声功率
    # 生成标准高斯分布的噪声序列（等于信号长度）,通过转换得到高斯噪声
    return np.random.randn(len(x)) * np.sqrt(npower)
# 返回加噪数据
def Gaussian_main(audio):
    n = wgn(audio, 6) # SNR=6
    audio_Gaussiannoise = audio + n
    audio_noise = audio + add_noise(audio,0.004)
    return audio_Gaussiannoise,audio_noise
# 可视化加噪数据波形图
def show_Addnoise_waveplot(audio, sr):
    audio_Gaussiannoise, audio_noise = Gaussian_main(audio)
    figure = plt.figure(figsize=(6, 6))
    rows,cols = 3,1
    # 可视化原始音频
    audio_list = [audio, audio_Gaussiannoise, audio_noise]
    title = ["original_data", "add_SNRGaussianNoise", "GaussianNoise"]
    for i in range(1,cols*rows+1):
        figure.add_subplot(rows,cols,i)
        librosa.display.waveplot(audio_list[i-1],sr=sr)
        plt.title(title[i-1])
        plt.tight_layout()
    plt.show()
# 频率、相位信息
def get_spectrogram(wav):
    D = librosa.stft(wav,n_fft=480,hop_length=160,win_length=480,window="hann")
    # 获取音频的频率、相位信息
    # spect,phase = librosa.magphase(D)
    spect = np.abs(D)
    phase = np.angle(D)
    return spect,phase
# 可视化频谱、相位谱
def show_spec_log_spectrogram(wav,sr):
    spect,phase = get_spectrogram(wav)
    print('spectrogram shape:', spect.shape,phase.shape)
    audio = [spect,phase]
    title = ["spect of origin audio","phase of origin audio"]
    rows, cols = 1, 2
    figure = plt.figure(figsize=(6, 2))
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        librosa.display.specshow(librosa.amplitude_to_db(audio[i-1]), sr=sr, x_axis='time', y_axis='hz')
        plt.title(title[i - 1])
        plt.tight_layout()
    plt.show()

# 波形拉伸 Time Stretching 参数范围为[0.8,1.2]
#  # rate：拉伸的尺寸，rate > 1 加快速度 rate < 1 放慢速度
# 当参数rate大于1时，Time Stretch执行的是一个在时间维度上压缩的过程，所以视觉上看来是向左偏移了；
def time_stretcheding(audio):
    rate = np.random.uniform(0.8,1.2)
    y_ts = librosa.effects.time_stretch(audio, rate=rate)
    return y_ts
def show_time_stretcheding(audio,sr):
    y_ts = time_stretcheding(audio)
    audio = [audio, y_ts]
    title = ["wavplot of origin audio", "time_stretcheding of origin audio"]
    rows, cols = 1, 2
    figure = plt.figure(figsize=(6, 2))
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        librosa.display.waveplot(audio[i - 1], sr=sr)
        plt.title(title[i - 1])
        plt.tight_layout()
    plt.show()

# 高音修正  增加或减少音频信号的音调，同时保持持续时间不变。参数范围为[-2,2]
# 经过Pitch Shift转换后的波形明显比原始波形的频率更高一些（n_steps>0），也就是音调变大了
def pitch_shift(audio,fs):
    n_steps = np.random.randint(-5,5)
    y_ps = librosa.effects.pitch_shift(audio,fs,n_steps=n_steps)
    return y_ps
def show_pitchShift(audio,fs):
    y_ps = pitch_shift(audio,fs)
    audio = [audio, y_ps]
    title = ["wavplot of origin audio", "pitch_shift of origin audio"]
    rows, cols = 1, 2
    figure = plt.figure(figsize=(6, 2))
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        librosa.display.waveplot(audio[i - 1], sr=fs)
        plt.title(title[i - 1])
        plt.tight_layout()
    plt.show()

# 波形位移 TimeShift
# # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])-># array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
def time_shift(audio,shift):
    # shift:位移的长度
    y_shift = np.roll(audio,int(shift))
    return y_shift
# 可视化波形位移
def show_timeShift(audio,fs):
    y_shift = time_shift(audio, shift=fs//2)
    audio = [audio, y_shift]
    title = ["wavplot of origin audio", "time_shift of origin audio"]
    rows, cols = 1, 2
    figure = plt.figure(figsize=(6, 2))
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        librosa.display.waveplot(audio[i - 1], sr=fs)
        plt.title(title[i - 1])
        plt.tight_layout()
    plt.show()

def add_reverberation(audio,fs):
# 给声音加混响 Image Source Method（镜像源方法）来实现语音加混响
# 1. 创建房间（定义房间大小、所需的混响时间、墙面材料、允许的最大反射次数、）
# 2. 在房间内创建信号源
# 3. 在房间内放置麦克风
# 4. 创建房间冲击响应
# 4、 模拟声音传播
    # 1. 创建房间
    # 所需的混响时间和房间的尺寸
    rt60_tgt = 0.5 # 所需的混响时间，s
    room_dim = [9,7.5,3.5] # 我们定义了一个9m x 7.5m x 3.5m的房间,米

    # 我们可以使用sabine's公式来计算壁面能量吸收和达到预期混响时间所需的ISM的最大阶数（RT60，即RIR衰减60分贝所需的时间）
    e_absorption,max_order = pra.inverse_sabine(rt60_tgt,room_dim) #返回墙壁吸收的能量和允许的反射次数
    # 我们还可以自定义墙壁材料和最大反射次数
    # m = pra.Material(energy_absorption="hard_surface") # 定义墙的材料，我们还可以定义不同墙面的材料
    # max_order = 3
    room = pra.ShoeBox(room_dim,fs=fs,materials=pra.Material(e_absorption),max_order=max_order)
    # 2. 在房间内创建一个位于[2.5,3.73,1.76]的源，从0.3秒开始向仿真中发出wav文件的内容
    room.add_source([2.5,3.73,1.76],signal=audio,delay=0.3)
    # 3. 在房间内创建麦克风
    # 定义麦克风的位置：（ndim,nmics）即每个列包含一个麦克风的坐标
    # 在这里我们创建一个带有两个麦克风的数组，
    # 分别位于[6.3,4.87,1.2]和[6.3,4.93,1.2]
    mic_locs = np.c_[
        [6.3,4.87,1.2], # minc 1
        [6.3,4.93,1.2], # minc 2
    ]
    room.add_microphone_array(mic_locs) # 最后将麦克风阵列放在房间里
    # 4. 创建房间冲击响应(Room Impulse Response)
    room.compute_rir()
    # 5. 模拟声音传播 每个源的信号将与相应的房间脉冲响应进行卷积。卷积的输出将在麦克风上求和
    room.simulate() #room.simulate(reference_mic=0, snr=10)      # 控制信噪比
    # 保存所有的信号到wav文件
    room.mic_array.to_wav("./test.wav", norm=True, bitdepth=np.float32,)
    # 测量混响时间
    rt60 = room.measure_rt60()
    print("The desired RT60 was {}".format(rt60_tgt))
    print("The measured RT60 is {}".format(rt60[1, 0]))


    plt.figure()
    # 绘制其中一个RIR. both can also be plotted using room.plot_rir()
    rir_1_0 = room.rir[1][0]    # 画出 mic 1和 source 0 之间的 RIR
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
    plt.title("The RIR from source 0 to mic 1")
    plt.xlabel("Time [s]")

    # 绘制 microphone 1 处接收到的信号
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(room.mic_array.signals[1, :])) / room.fs, room.mic_array.signals[1, :])
    plt.title("Microphone 1 signal")
    plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.show()

