# RadarFusionGUI
A real-time vital sign and activity detection system based on Infineon BGT60LTR11AIP device. The SDK is modified and optimized based on the Infineon Radar Development Kit 3.5.0.

## Overview

[![成果展示](https://img.youtube.com/vi/KkI8NtZr3IY/0.jpg)](https://www.youtube.com/watch?v=KkI8NtZr3IY)


After detecting that the human activity amplitude exceeds a certain threshold, the status will display a yellow exclamation mark and the model will identify the behavior category.
When the human activity amplitude is lower, it will switch to the heart rate and respiratory rate detection mode.
After accumulating static signals for a certain period of time, the corresponding heart rate and respiratory rate values will be output and displayed.

## Directory Structure
```
/.
├── RadarFusionGUI/
│ ├── demo/
│ │ ├── model/
│ │ ├── ui/
│ │ ├── real_time.py
│ │ │ ├── pyqt5/
│ │ │ │ ├── gui.py
│ ├── build/
│ ├── radar_sdk/
│ │ ├── examples/
│ │ │ ├── c/
│ │ │ │ ├── BGT60LTR11AIP/
│ │ │ │ │ ├── advanced_motion_sensing
│ │ │ │ │ │ |__ advanced_motion_sensing.c
```
## Enviroments
- OS: Windows 11
- Ubuntu 20.04 LTS
- GPU: Nvidia Geforce RTX 3060 Laptop
- CUDA: 12.7
- Python 3.10.11
- Pytorch: 1.13.0
## Requirements
The code requires python >= 3.10.11, as well as pytorch >= 1.13.0 and CUDA 12.7
`git clone https://github.com/01rice20/RadarFusionGUI.git`

## Getting Started
### radar_sdk
The radar_sdk is downloaded from the [provided_link](https://softwaretools.infineon.com/tools/com.ifx.tb.tool.ifxradarsdk) and mainly integrates the functionalities of advanced_motion_sensing and raw_data. This enables the BGT60LTR11AIP millimeter-wave radar to perform real-time object motion sensing, while annotating and outputting the radar raw data for both static and dynamic activities of the object to a .txt file.
Please ensure that the radar is properly connected to the computer and that the CMake environment is set up correctly. Then, build and launch advanced_motion_sensing.c. You can refer to the following video for the execution process. However, for the actual code modification, please use the `.\radar_sdk\examples\c\BGT60LTR11AIP\advanced_motion_sensing\advanced_motion_sensing.c` file from the provided webpage as the primary reference.

[![修改展示](https://img.youtube.com/vi/EmWcC_XyXQk/0.jpg)](https://www.youtube.com/watch?v=EmWcC_XyXQk)


### demo
The `.\demo\ui\real_time.py` file is the main detection algorithm. It performs classification and preprocessing on the raw data output from the previous step and saves the results as .jpg files.

If the activity is static, it proceeds to vital sign analysis, including filtering and peak calculations, followed by distinguishing heart rate and respiratory rate, with the predicted results output to a .csv file.
If the activity is dynamic, the .jpg files are passed to the trained PyTorch model. After the model makes predictions, the results are written to a .csv file.
The Python files are executed in the Windows OS with Ubuntu 20.04 LTS.

### GUI display
The `.\demo\pyqt5\gui.py` file provides an interface to display the results of the algorithm. This Python file can be executed directly on the Windows OS.






