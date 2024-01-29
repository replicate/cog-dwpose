import json
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from annotator.dwpose import DWposeDetector
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = DWposeDetector()

    def postprocess(self, pose, scores, threshold=0.3):
        num_people = pose["bodies"]["subset"].shape[0]
        bodies = pose["bodies"]["candidate"].reshape((num_people, 18, 2))
        subset = np.swapaxes(pose["bodies"]["subset"], 0, 1)

        all_poses = {}
        for i in range(num_people):
            scores_i = scores[i]
            hands_filter = scores_i[92:][scores_i[92:] < threshold]

            body = bodies[i]
            face = pose["faces"][i]
            hands = pose["hands"][i * 2 : (i + 1) * 2]

            body[scores_i[:18] < threshold] = -1
            face[scores_i[24:92] < threshold] = -1
            hands[0][scores_i[92:113] < threshold] = -1
            hands[1][scores_i[113:] < threshold] = -1

            pose_i = {"body": body, "face": face, "hands": hands}
            all_poses[f"person_{i}"] = pose_i

        return all_poses

    def predict(
        self,
        image: Path = Input(description="Input image for pose detection"),
        threshold: float = Input(
            description="Probability threshold to filter detected keypoints",
            ge=0,
            le=1,
            default=0.3,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        img = cv2.imread(str(image))
        out_img, pose, scores = self.model(img)

        # plot output on input image
        out_img_path = "/tmp/result.png"
        plt.imsave(out_img_path, out_img)

        # postprocess and filter detections
        out_path = "/tmp/result.npz"
        all_poses = self.postprocess(pose, scores, threshold)
        np.savez(out_path, **all_poses)

        return [Path(out_path), Path(out_img_path)]
