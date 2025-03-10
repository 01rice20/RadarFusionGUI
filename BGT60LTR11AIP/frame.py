import numpy as np
from numpy import pi
from time import time, sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.constants as C
from datetime import datetime


def get_frame_from_file(frame_head_size, frame):
    frame_head = []
    frame_body = []

    timestamp_line = next(line for line in frame if b'Timestamp' in line)
    timestamp = int(timestamp_line.split(b'=')[1])
    datetime_obj = datetime.fromtimestamp(timestamp / 1000.0)
    # time_str = datetime_obj.strftime('%H:%M:%S.%f')
    frame_body = frame[frame_head_size:]

    return datetime_obj, frame_body

def rangefft(final_frame):
    iq_data = [float(x.decode().strip()) for x in final_frame]
    complex_data = np.array(iq_data[::2]) + 1j * np.array(iq_data[1::2])
    fft_data = np.fft.fft(complex_data)
    # fft_data = 20 * np.log10(np.abs(fft_data)/np.max(fft_data))

    return fft_data[6:150]

def showplt(range_bins, time_bins, fft_data):
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(fft_data).reshape(-1, 1), aspect='auto', extent=[time_bins[0], time_bins[-1], range_bins[-1], range_bins[0]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Range (m)')
    plt.title('Range-Time Map')
    plt.show()

    return

# Test if the frame_body length is correct
def test(file):
    rest = []
    for line in file:
        rest.append(line.strip())
    
    print("rest: ", rest)

    return

def main():

    frame = []
    frame_body = []
    fft_data = []
    frame_title_size = 20
    frame_body_size = 519
    frame_head_size = 7
    frame_number = 156  # 20 s for 156 frames ; 59 s for 460 frames

    # Radar Parameters
    center_rf_frequency_khz = 6.1044e7
    sampling_frequency_hz = 2e3
    adc_resolution_bits = 12
    number_of_samples = 256
    samples_per_frame = 512
    frame_period_sec = 0.128
    c = 3e8
    bandwidth = 0.41e9
    max_distance = 7

    index = np.arange(1, number_of_samples+1)
    range_bins = np.linspace(0, max_distance, number_of_samples)
    time_bins = np.array([frame_period_sec * i for i in range(frame_number)])

    # BGT60LTR11AIP_waveForwardBackward_20240502_163953
    # BGT60LTR11AIP_waveLeftRight_20240502_164031
    # BGT60LTR11AIP_normal_20240502_163517
    # BGT60LTR11AIP_withoutBreathing_20240502_163737

    with open('../raw_data/BGT60LTR11AIP_waveForwardBackward_20240502_163953.txt', 'rb') as file:
        frame_title = [next(file) for _ in range(frame_title_size)]
        
        for i in range (frame_number):
            frame = [next(file) for _ in range(frame_body_size)]
            timestamp, frame_body = get_frame_from_file(frame_head_size, frame)
            fft_data_per_frame = rangefft(frame_body)
            fft_data.append(fft_data_per_frame)
        
        # test(file)

    file.close()

    fig, ax = plt.subplots()
    img = ax.imshow(np.abs(fft_data[0]).reshape(-1, 1), aspect='auto', extent=[time_bins[0], time_bins[-1], range_bins[-1], range_bins[0]])
    plt.colorbar(img, ax=ax, label='Magnitude')
    # ax.set_xticks([]) 
    # ax.set_yticks([])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Range (m)')
    ax.set_title('Range-Time Map')

    def update(frame):
        img.set_array(np.abs(fft_data[frame]).reshape(-1, 1))
        if frame == frame_number - 1:
            ani.event_source.stop()

        return img,

    ani = FuncAnimation(fig, update, frames=range(frame_number), interval=128, blit=True)
    plt.show()
    
if __name__ == '__main__':
    main()