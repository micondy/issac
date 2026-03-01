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

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the bimanual lift scene with robot, table and two cubes."""

    robot: ArticulationCfg = MISSING
    object_left: RigidObjectCfg = MISSING
    object_right: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    mini_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MiniTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.40, 0.0, 0.18], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(0.38, 0.38, 0.33),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    left_object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.22, 0.32),
            pos_y=(0.10, 0.18),
            pos_z=(0.12, 0.22),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(0.0, 0.0),
        ),
    )

    right_object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.22, 0.32),
            pos_y=(-0.18, -0.10),
            pos_z=(0.12, 0.22),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: mdp.JointPositionActionCfg = MISSING
    right_arm_action: mdp.JointPositionActionCfg = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_left_joint.*",
                        "openarm_right_joint.*",
                        "openarm_left_finger_joint.*",
                        "openarm_right_finger_joint.*",
                    ],
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_left_joint.*",
                        "openarm_right_joint.*",
                        "openarm_left_finger_joint.*",
                        "openarm_right_finger_joint.*",
                    ],
                )
            },
        )
        left_object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_left")},
        )
        right_object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_right")},
        )
        left_target_object_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_object_pose"},
        )
        right_target_object_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_object_pose"},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_left_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (0.13, 0.19), "z": (0.10, 0.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_left", body_names="ObjectLeft"),
        },
    )

    reset_right_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.19, -0.13), "z": (0.10, 0.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_right", body_names="ObjectRight"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    left_reaching_object = RewTerm(
        func=mdp.object_hand_distance,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("object_left"),
            "hand_cfg": SceneEntityCfg("robot", body_names=["openarm_left_hand"]),
        },
        weight=1.6,
    )

    right_reaching_object = RewTerm(
        func=mdp.object_hand_distance,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("object_right"),
            "hand_cfg": SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
        },
        weight=1.6,
    )

    left_lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("object_left")},
        weight=5.0,
    )

    right_lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("object_right")},
        weight=5.0,
    )

    both_lifting_object = RewTerm(
        func=mdp.both_objects_lifted,
        params={
            "minimal_height": 0.04,
            "left_object_cfg": SceneEntityCfg("object_left"),
            "right_object_cfg": SceneEntityCfg("object_right"),
        },
        weight=3.0,
    )

    left_object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "left_object_pose",
            "object_cfg": SceneEntityCfg("object_left"),
        },
        weight=40.0,
    )

    right_object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "right_object_pose",
            "object_cfg": SceneEntityCfg("object_right"),
        },
        weight=40.0,
    )

    left_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.10,
            "minimal_height": 0.04,
            "command_name": "left_object_pose",
            "object_cfg": SceneEntityCfg("object_left"),
        },
        weight=15.0,
    )

    right_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.10,
            "minimal_height": 0.04,
            "command_name": "right_object_pose",
            "object_cfg": SceneEntityCfg("object_right"),
        },
        weight=15.0,
    )

    both_object_goal_tracking = RewTerm(
        func=mdp.both_objects_goal_distance,
        params={
            "std": 0.25,
            "minimal_height": 0.04,
            "left_command_name": "left_object_pose",
            "right_command_name": "right_object_pose",
            "left_object_cfg": SceneEntityCfg("object_left"),
            "right_object_cfg": SceneEntityCfg("object_right"),
        },
        weight=40.0,
    )

    success_bonus = RewTerm(
        func=mdp.both_objects_goal_reached_bonus,
        params={
            "threshold": 0.22,
            "minimal_height": 0.04,
            "left_command_name": "left_object_pose",
            "right_command_name": "right_object_pose",
            "left_object_cfg": SceneEntityCfg("object_left"),
            "right_object_cfg": SceneEntityCfg("object_right"),
        },
        weight=50.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "openarm_left_joint.*",
                    "openarm_right_joint.*",
                    "openarm_left_finger_joint.*",
                    "openarm_right_finger_joint.*",
                ],
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    left_object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object_left")},
    )

    right_object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object_right")},
    )

    success = DoneTerm(
        func=mdp.both_objects_goal_reached,
        params={
            "threshold": 0.22,
            "minimal_height": 0.04,
            "left_command_name": "left_object_pose",
            "right_command_name": "right_object_pose",
            "left_object_cfg": SceneEntityCfg("object_left"),
            "right_object_cfg": SceneEntityCfg("object_right"),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -2e-4, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -2e-4, "num_steps": 10000},
    )


@configclass
class LiftBiEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bimanual lifting environment with two cubes."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0

        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
