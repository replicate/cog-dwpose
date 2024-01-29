## Cog wrapper for DWPose
Cog wrapper for DWPose, a whole body pose estimation model that detects 2D body, hands and face keypoints of multiple people in images. Refer to the [paper](https://arxiv.org/abs/2307.15880) and original [repo](https://github.com/IDEA-Research/DWPose) for details.


## Using the API
You need to have Cog and Docker installed to run this model locally. You also need to download thee pose model dw-ll_ucoco_384.onnx ([baidu](https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7), [google](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing)) and detection model yolox_l.onnx ([baidu](https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn), [google](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing)), and place them under the annotator/ckpts folder.


To use the DWPose, simply upload an image and set the threshold to filter out low probability detections. The API outputs an *.npz* file with keypoint detections for each person in the image, and a plot of detected keypoints overlaid on the image. 

To build the docker image with cog and run a prediction:
```bash
cog predict -i image=@test_images/running.jpeg -i threshold=0.3
```

To start a server and send requests to your locally or remotely deployed API:
```bash
cog run -p 5000 python -m cog.server.http
```

The output .npz file is organized as follows:
```
{
    "person_0": {
        "body": np.array of shape (18, 2),
        "face": np.array of shape (68, 2),
        "hands": np.array of shape (2, 21, 2),
    },
    "person_1": {
        "body": np.array of shape (18, 2),
        "face": np.array of shape (68, 2),
        "hands": np.array of shape (2, 21, 2),
    },
   ..
}
```
Keypoints are given as relative (x, y) coordinates and independent of image size. DWPose returns 18 body keypoints, 68 face keypoints and 21 hand keypoints per hand.

```
import numpy as np

# load keypoints
data = np.load("result.npz", allow_pickle=True)

# number of detected people
num_people = len(data.files)

# body, face, hands keypoints of person_0
person_0 = data["person_0"].item()
body_kpts = person_0["body"] 
face_kpts = person_0["face"] 
hands_kpts = person_0["hands"] 
```

## References
```
@inproceedings{yang2023effective,
  title={Effective whole-body pose estimation with two-stages distillation},
  author={Yang, Zhendong and Zeng, Ailing and Yuan, Chun and Li, Yu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4210--4220},
  year={2023}
} 
```