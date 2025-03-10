import numpy as np
from numpy import pi
from time import time, sleep
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import uniform_filter1d
import scipy.constants as C
from scipy.signal import lfilter, firwin, find_peaks, filtfilt, butter
from datetime import datetime
import math
import pandas as pd
import glob
import os
from scipy.fft import fft
import argparse
import csv
import time
from model.test import predict

def get_data_from_file(file, sample_number=128, frame_size=512):
    frame = [next(file) for _ in range(frame_size)]
    iq_data = np.array([float(x.decode().strip()) for x in frame])
    complex_data = iq_data[::2] + 1j * iq_data[1::2]

    # Time-Range Map
    fft_data = np.fft.fftshift(np.fft.fft(complex_data))
    fft_data = fft_data[(sample_number):]
    fft_data = np.array(fft_data).reshape(sample_number, 1)

    # Micro-Doppler Map
    fft2_data = np.reshape(complex_data, (sample_number, -1))
    fft2_data = np.fft.fftshift(np.fft.fft2(fft2_data))
    fft2_data = np.abs(fft2_data)
    fft2_data = np.log1p(fft2_data)

    return fft_data[4:], fft_data[8:], fft2_data

def mti_filter(iq_mat):
    order = 7
    cutoff = 0.01
    fs = 2e3

    b, a = butter(order, cutoff, 'high', fs=fs)
    Data_RTI_complex_MTIFilt = np.zeros_like(iq_mat, dtype=np.complex128)
    for k in range(iq_mat.shape[0]):
        Data_RTI_complex_MTIFilt[k, :] = filtfilt(b, a, iq_mat[k, :])

    return Data_RTI_complex_MTIFilt

def save_image(range_bins, doppler_bins, time_bins, fft_data, fft2_data, name):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.imshow(np.abs(fft_data), aspect='auto', extent=[time_bins[0], time_bins[-1], range_bins[-1], range_bins[0]], cmap='viridis')
    ax1.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'./ui/{name}_range.png')
    plt.close()

    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.imshow(fft2_data, aspect='auto', extent=[time_bins[0], time_bins[-1], doppler_bins[-1], doppler_bins[0]], cmap='viridis')
    ax2.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'./ui/{name}_doppler.png')
    plt.close()

# Track human body
def track_body(fft_data, number_of_samples):
    max_distance = 3
    min_range = 0.5
    max_range = 1.5
    range_min = int(number_of_samples*min_range/max_distance)
    range_max = int(number_of_samples*max_range/max_distance)
    max_num = 0
    fft_data_abs = np.abs(fft_data)
    fft_data_last = np.zeros_like(fft_data_abs[:, 0])

    for j in range(range_min, range_max):
        for i in range(fft_data_abs.shape[1]):
            fft_data_last[j] += fft_data_abs[j, i]
        if np.sum(fft_data_last[j]) > np.sum(fft_data_last[max_num]):
            max_num = j
    
    return max_num

def phase_function(fft_data):
    # Extract phase
    real_data = np.real(fft_data)
    imag_data = np.imag(fft_data)
    angle_fft = np.arctan2(imag_data, real_data)

    # Unwrap phase
    for i in range(1, len(angle_fft)):
        diff = angle_fft[i] - angle_fft[i-1]
        if diff > pi:
            angle_fft[i] -= 2*pi
        elif diff < -pi:
            angle_fft[i] += 2*pi

    # Calculate phase difference
    angle_diff = np.zeros_like(angle_fft)
    for i in range(1, len(angle_fft)):
        angle_diff[i] = angle_fft[i] - angle_fft[i-1]

    # Smooth the phase difference
    smoothed_data = uniform_filter1d(angle_diff, size=5)

    return smoothed_data

def bandpass_filter(iq_mat, lowcut_breath, highcut_breath, lowcut_heart, highcut_heart):
    order = 4

    b, a = butter(order, [lowcut_breath, highcut_breath], btype='band', fs=5)
    iq_mat_breath = filtfilt(b, a, iq_mat)
    d, c = butter(order, [lowcut_heart, highcut_heart], btype='band', fs=5)
    iq_mat_heart = filtfilt(d, c, iq_mat)
    
    return iq_mat_breath, iq_mat_heart

def find_signal_peaks(fft_windowed_signal, index_start, index_end, distance, fft_size_vital_signs, vital_signs_sample_rate):
        signal_region = fft_windowed_signal[index_start: index_end]
        peaks, _ = find_peaks(signal_region,
                              distance=int(max(1,distance*fft_size_vital_signs/vital_signs_sample_rate)))
        
        if len(peaks) < 2:
            return 0
    
        peak_times = peaks / vital_signs_sample_rate  # Convert peak indices to time in seconds
        peak_intervals = np.diff(peak_times)  # Calculate time intervals between peaks
        mean_interval = np.mean(peak_intervals)  # Calculate average interval
        bpm = 60 / mean_interval  # Calculate beats per minute
        
        return bpm

def ema_smoothing(heart_hist, breath_hist, alpha):

    heart_ema = [heart_hist[0]]
    breath_ema = [breath_hist[0]]

    for i in range(1, len(heart_hist)):
        heart_ema.append(int(alpha * heart_hist[i] + (1 - alpha) * heart_ema[-1]))
        breath_ema.append(int(alpha * breath_hist[i] + (1 - alpha) * breath_ema[-1]))

    return heart_ema, breath_ema

def vital_sign(cnt, fft_data, select_target_frame, max_num):

    # Radar Parameters
    center_rf_frequency_khz = 6.1044e7
    sampling_frequency_hz = 2e3
    adc_resolution_bits = 12
    number_of_samples = 128
    final_samples = 120
    frame_period_sec = 0.128
    c = 3e8
    bandwidth = 0.41e9
    detect_resolution = 0.025

    # Vital Signs Parameters
    peak_finding_distance = 0.01
    lowcut_breath = 0.1  #  Hz
    highcut_breath = 0.5  
    lowcut_heart = 0.8
    highcut_heart = 2.0
    processing_window_time = 30  # seconds
    vital_signs_sample_rate = 4
    processing_data_size = int(processing_window_time * vital_signs_sample_rate)
    fft_size_vital_signs = processing_data_size*4
    index_start_breathing = int(lowcut_breath/vital_signs_sample_rate * fft_size_vital_signs)
    index_end_breathing = int(highcut_breath/vital_signs_sample_rate * fft_size_vital_signs)
    index_start_heart = int(lowcut_heart/vital_signs_sample_rate * fft_size_vital_signs)
    index_end_heart = int(highcut_heart/vital_signs_sample_rate * fft_size_vital_signs)
    heart_hist = []
    breath_hist = []
   
    # Calculate vital signs per 1 second
    if(cnt % 6 == 0):
        window_fft_data = fft_data[max_num, cnt-select_target_frame:cnt]
        window_fft_data = phase_function(window_fft_data)
        breath_fft, heart_fft = bandpass_filter(window_fft_data, lowcut_breath, highcut_breath, lowcut_heart, highcut_heart)
        rate_index_br = find_signal_peaks(breath_fft, index_start_breathing, index_end_breathing,
                                            peak_finding_distance, fft_size_vital_signs, vital_signs_sample_rate)
        rate_index_hr = find_signal_peaks(heart_fft, index_start_heart, index_end_heart,
                                            peak_finding_distance, fft_size_vital_signs, vital_signs_sample_rate)
                
        return rate_index_hr, rate_index_br
    else:
        return 0, 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_frame_number", type=int, help="Total frame number", default=468)
    args = parser.parse_args()
    sample_number = 128
    frame_size = 512
    frame = []
    static_frame = np.empty((sample_number-4, 0))
    fft_frame = np.empty((sample_number-8, 0))
    fft2_frame = np.empty((sample_number, 0))
    rate_index_hr = -1
    rate_index_br = -1
    heart_hist = [-1]
    breath_hist = [-1]
    heart_hist_ema = [-1]
    breath_hist_ema = [-1]

    select_target_frame = 117
    max_distance = 3
    samples_per_frame = 512
    frame_period_sec = 0.128
    sampling_frequency_hz = 2e3
    frame_number = 78
    range_bins = np.linspace(0, max_distance, sample_number)
    doppler_bins = np.fft.fftfreq(samples_per_frame, d=1/sampling_frequency_hz)
    time_bins = np.array([frame_period_sec * i for i in range(frame_number)])

    file_name = f'C:\Users\HanaL\Desktop\RadarFusionGUI/BGT60LTR11AIP/frame_data.txt'
    move = []
    data = []
    predict_class = -1
    cnt_second = 0

    for i in range (args.total_frame_number):
        with open(file_name, 'rb') as file:
            [next(file) for _ in range((i+1)*(frame_size+1))]
            state_str = next(file)
            fft_data4, fft_data8, fft2_data = get_data_from_file(file, sample_number, frame_size)
            print("Frame: ", i)
            if(i >= 19 and i < 98):
                fft_frame = np.hstack((fft_frame, fft_data8))
                fft2_frame = np.hstack((fft2_frame, fft2_data))
                save_image(range_bins, doppler_bins, time_bins, fft_frame, fft2_frame, "activity")
                if(fft_frame.shape[1] == 78):
                    fft_frame = mti_filter(np.real(fft_frame))
                    predict_class = predict()
                    print("Predict class: ", predict_class)
                    
            elif(i >= 98):
                static_frame = np.hstack((static_frame, fft_data4))
                print("static_frame shape: ", static_frame.shape)
                if(static_frame.shape[1] % 78 == 0):
                        max_num = track_body(static_frame, sample_number)
                if(static_frame.shape[1] > select_target_frame):
                    rate_index_hr, rate_index_br = vital_sign(static_frame.shape[1], static_frame, select_target_frame, max_num)
                    if(rate_index_hr > 0 and rate_index_br > 0):
                        heart_hist.append(int(rate_index_hr))
                        breath_hist.append(int(rate_index_br))
                        heart_hist_ema, breath_hist_ema = ema_smoothing(heart_hist[1:], breath_hist[1:], alpha=0.3)
                        print("Heart rate: ", heart_hist_ema[-1])
                        print("Breath rate: ", breath_hist_ema[-1])
        
        if(str(state_str[:1]) == "b'0'"):
            move.append(0)
        elif(str(state_str[:1]) == "b'1'"):
            move.append(1)
        
        if(i < 39):
            time.sleep(0.14)

        if i % 6 == 0:
            cnt_second += 1
            data = []
            data.append(cnt_second)
            data.append(move[-1])
            data.append(predict_class)
            data.append(heart_hist_ema[-1])
            data.append(breath_hist_ema[-1])
    
            writer = csv.writer(file)
            with open("./ui/output.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            
            