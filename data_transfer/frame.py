import numpy as np
from numpy import pi
from time import time
import matplotlib.pyplot as plt
import scipy.constants as C
from datetime import datetime

def get_frame_from_file(frame_head_size, frame):
    frame_head = []
    frame_body = []

    timestamp_line = next(line for line in frame if b'Timestamp' in line)
    timestamp = int(timestamp_line.split(b'=')[1])
    frame_body = frame[frame_head_size:]

    return timestamp, frame_body

def main():

    frame = []
    frame_body = []
    final_frame = []
    rest = []
    frame_title_size = 20
    frame_body_size = 519
    frame_head_size = 7
    # 20 s for 156 frames ; 59s for 460 frames
    frame_number = 460

    with open('../raw_data/BGT60LTR11AIP_withoutBreathing_20240502-163737.raw.txt', 'rb') as file:
        frame_title = [next(file) for _ in range(frame_title_size)]
        
        for i in range (frame_number):
            frame = [next(file) for _ in range(frame_body_size)]
            timestamp, frame_body = get_frame_from_file(frame_head_size, frame)
            final_frame.append(frame_body)
            datetime_obj = datetime.fromtimestamp(timestamp / 1000.0)
            # time_str = datetime_obj.strftime('%H:%M:%S.%f')
            print("timestamp: ", datetime_obj)
        
        # Test if the frame_body length is correct
        for line in file:
            rest.append(line.strip())
        print("rest: ", rest)
    
    file.close()
    print("frame number: ", len(final_frame))

if __name__ == '__main__':
    main()