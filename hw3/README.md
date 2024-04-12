# homework3-kszuyen
**Usage Demonstration Video:** [Youtube link](https://youtu.be/qtC661V8vtc)
## Create Environment
```
conda create -n <env_name> python=3.9
conda activate <env_name>
pip install -r requirements.txt
```
## Goal: Visual Odometry
### Step 1: Camera Calibration
```
python camera_calibration.py calib_video.avi
```
Enter \<SPACE\> key to add new frame to calibrate.  
Use `python3 camera_calibration.py --help` to check more argument information.  
The program will save "**camera_parameters.npy**" by default.  

### Step 2~4: Perform Visual Odometry
```
python vo.py frames
```
When finished, you can press \<ESC\> to kill each window.
