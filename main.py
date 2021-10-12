# -*- coding: utf-8 -*-
import librosa.display
import os
from AudioAugment import audioData_Augment
import matplotlib.pyplot as plt
# 高音修正
if __name__=="__main__":
    # 读取音频
    path = "data"
    # 数据增强音频文件
    new_path = "data_inverse"
    file = os.listdir(path)
    file_new = os.listdir(path_save)

    for i in range(len(file)):
        file_name = os.path.join(path, file[i])
        audio, fs = librosa.load(file_name, sr=None)
        print(file_name)
        # 可视化mel特征 一阶mel，二阶mel特征谱图
        audioData_Augment.show_melspectrogram(audio,fs)
        # 时间频率掩码 通过同时随机去除几行(帧)和几列(频率)来实现数据的增强
        audioData_Augment.show_time_frequeny_mask(audio,fs)
        # 在原始信号上添加固定信噪比的高斯白噪,及均值为0，标准差为1的高斯白噪声
        audioData_Augment.show_Addnoise_waveplot(audio, fs)
        # 频谱，相位谱特征图
        audioData_Augment.show_spec_log_spectrogram(audio,fs)
        # 时间拉伸 Time Stretching 参数范围为[0.8,1.2]
        audioData_Augment.show_time_stretcheding(audio,fs)
        # 增加或减少音频信号的音调，同时保持持续时间不变。参数范围为[-5,5]
        audioData_Augment.show_pitchShift(audio,fs)
        # 波形位移 TimeShift
        audioData_Augment.show_timeShift(audio,fs)
        # 加混响
        audioData_Augment.add_reverberation(path,new_path,file_name,audio,fs,i)
