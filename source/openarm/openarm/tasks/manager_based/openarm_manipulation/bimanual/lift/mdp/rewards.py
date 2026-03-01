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


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_left"),
) -> torch.Tensor:
    """Reward if the object is lifted above the target height."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    return torch.where(object_asset.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_hand_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the hand getting closer to the target object using tanh-kernel."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[hand_cfg.name]

    cube_pos_w = object_asset.data.root_pos_w
    hand_pos_w = robot.data.body_pos_w[:, hand_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(cube_pos_w - hand_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_left"),
) -> torch.Tensor:
    """Reward object tracking its commanded goal position in robot frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        des_pos_b,
    )
    distance = torch.norm(des_pos_w - object_asset.data.root_pos_w, dim=1)
    return (object_asset.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
