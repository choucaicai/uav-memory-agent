"""Convert raw RGBD steps into policy inputs and action-token labels."""
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from model.action_tokenizer import ActionTokenizer
from model.vision_backbone import DinoSigLIPImageTransform
from model.prompt_llama2 import LLaMa2ChatPromptBuilder

IGNORE_INDEX = -100


@dataclass
class SingleDepthRLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: DinoSigLIPImageTransform
    predict_stop_token: bool = True
    camera_names = ['camera_0']
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

        # for k,v in rlds_batch["observation"].items():
        #     if isinstance(v,np.ndarray):
        #         print(f"{k},{type(v)}",v.shape)

        img_cur = Image.fromarray(rlds_batch["observation"]["image_0"][0])
        # img_cur = Image.fromarray(rlds_batch["observation"]["image_1"][0])
        # img_past1 = Image.fromarray(rlds_batch["observation"]["image_secondary"][0])
        # img_past2 = Image.fromarray(rlds_batch["observation"]["image_wrist"][0])

        camera_0_fov = rlds_batch["observation"]["camera_0_fov"][0]
        camera_0_image = rlds_batch["observation"]["camera_0_image"][0]
        camera_0_quat_wb = rlds_batch["observation"]["camera_0_quat_wb"][0]
        camera_0_camera_pos = rlds_batch["observation"]["camera_0_camera_pos"][0]
        camera_0_depth = rlds_batch["observation"]["camera_0_depth"][0]


        target_pose = rlds_batch["observation"]["target_pose"][0]

        # print('camera_0_image:',camera_0_image.shape)
        # print('camera_0_fov:',camera_0_fov.shape)
        # print('camera_0_quat_wb:',camera_0_quat_wb.shape)
        # print('camera_0_camera_pos:',camera_0_camera_pos.shape)
        # print('camera_0_depth:',camera_0_depth.shape)
        camera_params = np.concatenate([camera_0_quat_wb,camera_0_fov[:, None]],axis=1)
        camera_params = torch.tensor(camera_params)
        camera_poses = torch.tensor(camera_0_camera_pos)
        camera_depths = torch.tensor(camera_0_depth)

        target_pose = torch.tensor(target_pose)


        # Image.fromarray(camera_0_image[0]).save('temp.png')

        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # print('RLDS action:',rlds_batch.keys(),action)
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = LLaMa2ChatPromptBuilder("prismatic")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        # print(self.action_tokenizer(action),action)
        # print('action ids:',self.base_tokenizer(self.action_tokenizer(action), add_special_tokens=True).input_ids)
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # print('prompt:',prompt_builder.get_prompt())

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        # tr_img_cur, tr_pst1, tr_pst2 = self.image_transform(img_cur), self.image_transform(img_past1), self.image_transform(img_past2)# , self.image_transform(img_past3),
        pixel_values = self.image_transform(img_cur)

        # camera_images = torch.stack([self.image_transform(Image.fromarray(cur_img)) for cur_img in camera_0_image])
        camera_images = [self.image_transform(Image.fromarray(cur_img)) for cur_img in camera_0_image]

        if isinstance(camera_images[0],dict):
            camera_images = {
                k: torch.cat(
                      [img[k] for img in camera_images], dim=0  # , tr_pst3[k]
                )
                for k in camera_images[0].keys()
            }
        else:
            camera_images = torch.stack(camera_images)
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # if isinstance(camera_images,dict):
        #     print('camera_params,camera_pos:',camera_params.shape,camera_poses.shape,len(camera_images),{ k:v.shape for k,v in camera_images.items()},camera_depths.shape)
        # else:
            # print('camera_params,camera_pos:',camera_params.shape,camera_poses.shape,len(camera_images),camera_images.shape,camera_depths.shape)
        # print('input_ids:',input_ids)
        # print('labels:',labels)
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        # print('input_ids:',input_ids.shape,labels.shape)
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, camera_images=camera_images, camera_params=camera_params, camera_poses=camera_poses, camera_depths=camera_depths,target_poses=target_pose)
