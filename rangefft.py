import numpy as np
from numpy import pi
from time import time
import matplotlib.pyplot as plt
import scipy.constants as C
import zarr

def get_frame_from_content(frame):
    rawframe = []
    frame_num = int.from_bytes(frame[103:104], 'little')
    # Frame number
    frame_size = int.from_bytes(frame[105:107], 'little')
    num_iq_per_frame = frame_size * 2
    iq_end = 0
    for i in range(num_iq_per_frame):
        # skip first 107 bytes (metadata and frame header)
        iq_start = 107 + i*2
        iq_end = iq_start + 2
        iq = int.from_bytes(
            frame[iq_start:iq_end], 'little', signed=True)
        rawframe.append(iq)
    frame = frame[iq_end + 4:]
    assert(len(rawframe) == num_iq_per_frame) # 14336
    print("rawframe: ", rawframe)

    return frame_size, rawframe, frame

def draw_range_FFT_one_chirp(rawframe):

    fig = plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    fig.suptitle('Range FFT (chirp 1, second 20, standing, 1 m) ', fontsize=14, fontweight='bold')
    # plt.xlim((0, 15))
    ax.set_xlabel("range (m)")
    
    num_samples = 128 # sample node
    num_chirps = 14
    Fs = 1e7 # 10 MHz = 10,000,000 Hz
    Tc = 2.2e-3 # 2200 µs = 0.002200 s
    Ts = 1/Fs
    band_width = 6.8e9 # 6800 MHz = 6,800,000,000 Hz
    slope = band_width/Tc 
    c = 3e8 # 300000000 m/s
    central_freq = 60.5e9 # 60.5 GHz = 60,500,000,000 Hz
    wave_len = c/central_freq
    range_phase_fac = wave_len/(4*np.pi)*1000 # Unit-mm
    frame_period = 3.5e-3   # * 0.0035
    frame_size = 7168

    sweeps = np.zeros(num_samples*num_chirps, dtype=complex)

    for i in range(num_chirps):
        for x in range(num_samples):
            sweeps[i*num_samples+x] = complex(rawframe[(512*i) + (x*2)], rawframe[(512*i) + (1+x*2)])

    print("sweeps size: ", sweeps.shape)

    # din_fft = np.fft.fft(sweeps[0:128])
    index = np.arange(0, num_samples, 1)
    # range_bin = (index - 1) * (c*Tc*Fs) / (2 * band_width * num_samples)
    # freq_bin = (index - 1) * Fs / num_samples

    range_axis = np.arange(0,128)*(c*Tc*Fs) / (2 * band_width * num_samples)
    freq_axis = np.arange(num_samples)*(Fs/num_samples)

    # doppler = 10*np.log10(np.abs(np.fft.fft(sweeps)))
    # frequency = np.fft.fftfreq(128*14, 1/Fs)
    # rangefft = frequency*c/(2*slope)

    # y1_max=np.argmax(range_axis)

    # show_max='['+str(y1_max)+' '+str(range_axis[y1_max])+']'
    
    # plt.annotate(show_max,xy=(y1_max,range_axis[y1_max]),xytext=(y1_max,range_axis[y1_max]))

    # plt.plot(y1_max,range_axis[y1_max],'ko') 
    # print("y1_max: ", y1_max)
    # print("range_axis[y1_max]: ", range_axis[y1_max])
    plt.plot(range_axis, abs(sweeps[0:128]))
    plt.savefig("pic13.jpg")

   
def main():
    # sweep_path = "./export/1229_023_jump1/sweep.zarr"
    # cnt = 0
    # sweeps = zarr.open(sweep_path, mode='r')
    # print("sweeps size: ", sweeps.shape)

    # for sweep in sweeps:
    #     draw_range_FFT_one_chirp(sweeps)
    #     print("sweep size: ", sweep.shape)
    #     cnt += 1
    #     print("cnt: ", cnt)
    save_data = []
    metadata_plus_header_size = 103 + 4
    cnt = 0
    recieve_buffer = None
    sweep_idx = 0
    with open('0315_01.txt', 'rb') as f:
        content = f.read()
        print("content: ", content)
        while(len(content) > 7168):
            frame = []
            frame_size = 0
            rawframe = []
            frame_size, rawframe, frame = get_frame_from_content(content)
            # if(cnt == 400):
            #     z = draw_range_FFT_one_chirp(rawframe)
            #     break
            content = frame
            cnt += 1


if __name__ == '__main__':
    main()