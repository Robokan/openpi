"""OpenArm bimanual robot policy transforms.

OpenArm is a 7-DOF bimanual robot with grippers:
- State/Action format: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)] = 16 DOF
- Cameras: ego (overhead), left_wrist, right_wrist

Unlike ALOHA (6-DOF), OpenArm has an extra joint per arm, allowing human-like elbow movement.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_openarm_example() -> dict:
    """Creates a random input example for the OpenArm policy."""
    return {
        "state": np.ones((16,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class OpenArmInputs(transforms.DataTransformFn):
    """Inputs for the OpenArm policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [16] - [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
    - actions: [action_horizon, 16]
    """

    # The expected camera names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            return einops.rearrange(img, "c h w -> h w c")

        base_image = convert_image(in_images["cam_high"])

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = convert_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OpenArmOutputs(transforms.DataTransformFn):
    """Outputs for the OpenArm policy.
    
    Slices model output (32 dims) back to OpenArm's 16 DOF.
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :16])
        return {"actions": actions}
