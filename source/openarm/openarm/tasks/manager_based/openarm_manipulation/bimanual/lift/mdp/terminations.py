# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def both_objects_goal_reached(
    env: ManagerBasedRLEnv,
    threshold: float,
    minimal_height: float,
    left_command_name: str,
    right_command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_object_cfg: SceneEntityCfg = SceneEntityCfg("object_left"),
    right_object_cfg: SceneEntityCfg = SceneEntityCfg("object_right"),
) -> torch.Tensor:
    """Terminate episode when both objects are close to their goals and lifted."""
    robot: RigidObject = env.scene[robot_cfg.name]
    left_object: RigidObject = env.scene[left_object_cfg.name]
    right_object: RigidObject = env.scene[right_object_cfg.name]

    left_command = env.command_manager.get_command(left_command_name)
    right_command = env.command_manager.get_command(right_command_name)

    left_des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        left_command[:, :3],
    )
    right_des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        right_command[:, :3],
    )

    left_distance = torch.norm(left_des_pos_w - left_object.data.root_pos_w, dim=1)
    right_distance = torch.norm(right_des_pos_w - right_object.data.root_pos_w, dim=1)

    left_reached = (left_distance < threshold) & (left_object.data.root_pos_w[:, 2] > minimal_height)
    right_reached = (right_distance < threshold) & (right_object.data.root_pos_w[:, 2] > minimal_height)

    return left_reached & right_reached
