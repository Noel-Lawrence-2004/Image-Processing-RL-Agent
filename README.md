
# Image-Processing-RL-Agent
A RL agent that Learn appropriate Preprocessing factors to be applied to images such that it can improve the performance of Yolov5 predictions of Various objects 

## Continuous Action Space Design
In this project, the action space is a continuous vector representing image transformation parameters.  
Each dimension of the action vector corresponds to a preprocessing parameter:
  - action[0] - Brightness | `[-1, 1]` | Adjusts pixel intensity
  - action[1] - Contrast | `[-1, 1]` | Modifies image contrast
  - action[2] - Sharpness | `[-1, 1]` | Sharpens or blurs the image

