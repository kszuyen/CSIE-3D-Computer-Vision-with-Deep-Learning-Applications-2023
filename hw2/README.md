[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/cKfvp3Eo)
# Homework2
**Usage Demonstration Video:** [Youtube link](https://youtu.be/faLUxiH6xjo)
## Implemented Methods:
- P3P (with RANSAC): performs not very well
- **DLT (with RANSAC): recommended**  
## Download Dataset
From Link: [Download](https://drive.google.com/u/0/uc?export=download&confirm=qrVw&id=1GrCpYJFc8IZM_Uiisq6e8UxwVMFvr4AJ)

or run `bash download.sh`

## Create Environment
```
conda create -n <env_name> python=3.9
conda activate <env_name>
pip install -r requirements.txt
```
## Problem 1: 2D-3D Matching
### Q1-1, Q1-2
```
python 2d3dmatching.py <method: [p3p]or[dlt]>
```
This creates **camera_position.pkl** file, and prints out the **Translation (pose)** and **Rotation errors**.
### Q1-3
```
python visualization.py
```
This reads the **camera_position.pkl** file and visualize the calculated camera positions.
## Problem 2: Augmented Reality
### Q2-1
```
python transform_cube.py <method: [p3p]or[dlt]>
```
Place a virtual cube in the desired position, and it will make an output **AR_video_[method].mp4** video.