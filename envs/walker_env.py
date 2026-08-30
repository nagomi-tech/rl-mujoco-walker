"""
BlockyWalkerEnv: MuJoCo 3D ブロック型人型ロボットのカリキュラム学習環境

Level 0 (FLAT)     : ゴールに向かってできるだけ速く走る
                     転倒 / 手をつくとペナルティ
Level 1 (WALL_GAP) : 壁の隙間を通り抜けながらゴールへ
Level 2 (DODGE)    : ジグザグ壁 + 飛んでくるキューブを回避しながらゴールへ
"""

import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
FRAME_SKIP    = 4       # mj_step を何回繰り返すか（実効 dt = 0.005*4 = 0.02 s）
MIN_Z         = 0.60    # これより低いと転倒（Level 0: 厳しめ）
MAX_Z         = 1.9
MAX_STEPS     = 1000    # 1エピソードの最大ステップ数
N_CUBES       = 2

STEP_BONUS      = 3.0   # 左右交互着地ボーナス
MIN_SWING_STEPS = 5     # 足が最低この steps 浮いていないとボーナスなし（0.1秒）

GOAL_INIT_X   = 10.0   # 最初のゴール位置 (m)
GOAL_STEP_X   = 10.0   # ゴール到達後に次のゴールまで追加する距離

# 階段環境定数
N_STAIRS      = 8
STAIR_HEIGHT  = 0.05   # 1段の高さ (m)
STAIR_DEPTH   = 0.40   # 1段の奥行き (m)  ※足長26cmに対して14cm余裕
STAIR_START_X = 2.0
STAIR_END_X   = STAIR_START_X + N_STAIRS * STAIR_DEPTH  # 5.2m
STAIR_TOP_Z   = N_STAIRS * STAIR_HEIGHT                  # 0.40m
STAIRS_GOAL_X = STAIR_END_X + 1.0                        # 6.2m

OBS_DIM = 35   # qpos(15) + qvel(16) + goal/障害物情報(4)
N_ACTS  = 10

# 地形視覚観測
N_HEIGHT_SAMPLES = 10
HEIGHT_SAMPLE_OFFSETS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]  # 前方サンプル点(m)


# ---------------------------------------------------------------------------
# XML ビルダー
# ---------------------------------------------------------------------------

def _robot_bodies_xml() -> str:
    return """
    <body name="torso" pos="0 0 1.00">
      <freejoint name="root"/>
      <geom name="torso_box" type="box" size="0.18 0.12 0.13" material="mat_robot"/>

      <body name="head" pos="0 0 0.25">
        <geom name="head_box" type="box" size="0.12 0.12 0.12" material="mat_robot"/>
      </body>

      <!-- 左腕 -->
      <body name="left_upper_arm" pos="0 0.245 0.06">
        <joint name="left_shoulder" type="hinge" axis="0 1 0" range="-2.0 1.0"/>
        <geom name="left_upper_arm_geom" type="box" size="0.065 0.065 0.14"
              pos="0 0 -0.14" material="mat_robot"/>
        <body name="left_forearm" pos="0 0 -0.28">
          <joint name="left_elbow" type="hinge" axis="0 1 0" range="-1.8 0.1"/>
          <geom name="left_forearm_geom" type="box" size="0.055 0.055 0.115"
                pos="0 0 -0.115" material="mat_robot"/>
        </body>
      </body>

      <!-- 右腕 -->
      <body name="right_upper_arm" pos="0 -0.245 0.06">
        <joint name="right_shoulder" type="hinge" axis="0 1 0" range="-2.0 1.0"/>
        <geom name="right_upper_arm_geom" type="box" size="0.065 0.065 0.14"
              pos="0 0 -0.14" material="mat_robot"/>
        <body name="right_forearm" pos="0 0 -0.28">
          <joint name="right_elbow" type="hinge" axis="0 1 0" range="-1.8 0.1"/>
          <geom name="right_forearm_geom" type="box" size="0.055 0.055 0.115"
                pos="0 0 -0.115" material="mat_robot"/>
        </body>
      </body>

      <!-- 左脚 -->
      <body name="left_thigh" pos="0 0.10 -0.26">
        <joint name="left_hip" type="hinge" axis="0 1 0" range="-1.5 0.3"/>
        <geom type="box" size="0.085 0.085 0.18" pos="0 0 -0.18" material="mat_robot"/>
        <body name="left_shin" pos="0 0 -0.36">
          <joint name="left_knee" type="hinge" axis="0 1 0" range="-0.1 1.8"/>
          <geom name="left_shin_geom" type="box" size="0.07 0.07 0.15" pos="0 0 -0.15" material="mat_robot"/>
          <body name="left_foot" pos="0 0 -0.30">
            <joint name="left_ankle" type="hinge" axis="0 1 0" range="-0.6 0.6"/>
            <geom name="left_foot_geom" type="box" size="0.13 0.08 0.04"
                  pos="0.04 0 -0.04" material="mat_robot"/>
          </body>
        </body>
      </body>

      <!-- 右脚 -->
      <body name="right_thigh" pos="0 -0.10 -0.26">
        <joint name="right_hip" type="hinge" axis="0 1 0" range="-1.5 0.3"/>
        <geom type="box" size="0.085 0.085 0.18" pos="0 0 -0.18" material="mat_robot"/>
        <body name="right_shin" pos="0 0 -0.36">
          <joint name="right_knee" type="hinge" axis="0 1 0" range="-0.1 1.8"/>
          <geom name="right_shin_geom" type="box" size="0.07 0.07 0.15" pos="0 0 -0.15" material="mat_robot"/>
          <body name="right_foot" pos="0 0 -0.30">
            <joint name="right_ankle" type="hinge" axis="0 1 0" range="-0.6 0.6"/>
            <geom name="right_foot_geom" type="box" size="0.13 0.08 0.04"
                  pos="0.04 0 -0.04" material="mat_robot"/>
          </body>
        </body>
      </body>
    </body>
"""


def _actuators_xml() -> str:
    return """
  <actuator>
    <motor name="left_shoulder_act"  joint="left_shoulder"  gear="60"  ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="left_elbow_act"     joint="left_elbow"     gear="40"  ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="right_shoulder_act" joint="right_shoulder" gear="60"  ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="right_elbow_act"    joint="right_elbow"    gear="40"  ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="left_hip_act"       joint="left_hip"       gear="120" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="left_knee_act"      joint="left_knee"      gear="90"  ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="left_ankle_act"     joint="left_ankle"     gear="60"  ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="right_hip_act"      joint="right_hip"      gear="120" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="right_knee_act"     joint="right_knee"     gear="90"  ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="right_ankle_act"    joint="right_ankle"    gear="60"  ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
"""


def _header_xml() -> str:
    return """<mujoco model="blocky_walker">
  <compiler angle="radian"/>
  <option timestep="0.005" integrator="RK4" iterations="50"/>

  <default>
    <joint limited="true" damping="0.5" armature="0.01"/>
    <geom contype="1" conaffinity="1" condim="3" friction="1.0 0.1 0.01" rgba="0.7 0.7 0.7 1"/>
  </default>

  <asset>
    <texture type="2d" name="tex_floor" builtin="checker"
             rgb1="0.45 0.45 0.45" rgb2="0.62 0.62 0.62" width="100" height="100"/>
    <material name="mat_floor"  texture="tex_floor" texrepeat="15 15"/>
    <material name="mat_robot"  rgba="0.95 0.55 0.10 1"/>
    <material name="mat_wall"   rgba="0.92 0.92 0.92 1"/>
    <material name="mat_cube"   rgba="0.85 0.15 0.15 1"/>
    <material name="mat_goal"   rgba="0.10 0.90 0.20 0.45"/>
  </asset>
"""


def _lights_xml(slope_deg: float = 0.0) -> str:
    if slope_deg == 0.0:
        floor_geom = ('    <geom name="floor" type="plane" size="15 5 0.1" material="mat_floor"\n'
                      '          pos="5 0 0" contype="1" conaffinity="1"/>')
    else:
        slope_rad = math.radians(slope_deg)
        floor_geom = (f'    <geom name="floor" type="plane" size="15 5 0.1" material="mat_floor"\n'
                      f'          pos="0 0 0" euler="0 {-slope_rad:.6f} 0" contype="1" conaffinity="1"/>')
    return f"""
    <light pos="2 -2 6"  dir="-0.3 0.5 -1" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
    <light pos="-2 2 6"  dir="0.3 -0.5 -1" diffuse="0.5 0.5 0.5"/>
{floor_geom}
    <camera name="track" pos="5 -12 5" xyaxes="1 0 0 0 0.32 0.95" fovy="55"/>
"""


def _goal_marker_xml(init_x: float = GOAL_INIT_X, init_z: float = 0.0) -> str:
    """モカップボディ：Python から位置を動的に更新できるゴールマーカー。"""
    return f"""
    <body name="goal_marker" mocap="true" pos="{init_x} 0 {init_z}">
      <!-- 地面のゴールゾーン（緑の円盤） -->
      <geom name="goal_disc" type="cylinder" size="1.0 0.02" material="mat_goal"
            contype="0" conaffinity="0" pos="0 0 0.02"/>
      <!-- 細いゴールポスト -->
      <geom name="goal_post" type="cylinder" size="0.06 1.5" material="mat_goal"
            contype="0" conaffinity="0" pos="0 0 1.5"/>
    </body>
"""


def _stairs_xml() -> str:
    """8段の階段 + 上部プラットフォームを生成する。"""
    lines = []
    for i in range(N_STAIRS):
        cx = STAIR_START_X + i * STAIR_DEPTH + STAIR_DEPTH / 2
        hz = (i + 1) * STAIR_HEIGHT / 2
        cz = hz
        lines.append(
            f'    <geom name="stair_{i}" type="box" '
            f'size="{STAIR_DEPTH/2:.3f} 2.5 {hz:.4f}" '
            f'pos="{cx:.3f} 0 {cz:.4f}" '
            f'rgba="0.60 0.45 0.30 1" contype="1" conaffinity="1"/>')
    # 上部プラットフォーム
    plat_cx = STAIR_END_X + 1.0
    plat_hz = STAIR_TOP_Z / 2
    lines.append(
        f'    <geom name="stair_platform" type="box" '
        f'size="1.0 2.5 {plat_hz:.4f}" '
        f'pos="{plat_cx:.3f} 0 {plat_hz:.4f}" '
        f'rgba="0.60 0.45 0.30 1" contype="1" conaffinity="1"/>')
    return "\n".join(lines) + "\n"


def _wall_segments_xml(wall_x: float, gap_center_y: float, gap_width: float,
                        name_prefix: str, room_half_w: float = 2.2) -> str:
    h, d = 0.85, 0.15
    left_end    = gap_center_y - gap_width / 2
    right_start = gap_center_y + gap_width / 2
    left_hw  = (left_end  - (-room_half_w)) / 2
    left_cy  = (-room_half_w + left_end) / 2
    right_hw = (room_half_w - right_start) / 2
    right_cy = (right_start + room_half_w) / 2
    parts = []
    if left_hw > 0.01:
        parts.append(
            f'    <geom name="{name_prefix}_L" type="box" '
            f'size="{d} {left_hw:.3f} {h}" '
            f'pos="{wall_x} {left_cy:.3f} {h}" '
            f'material="mat_wall" contype="1" conaffinity="1"/>')
    if right_hw > 0.01:
        parts.append(
            f'    <geom name="{name_prefix}_R" type="box" '
            f'size="{d} {right_hw:.3f} {h}" '
            f'pos="{wall_x} {right_cy:.3f} {h}" '
            f'material="mat_wall" contype="1" conaffinity="1"/>')
    return "\n".join(parts) + "\n"


def _hurdles_xml(height: float = 0.08, spacing: float = 0.5,
                 start_x: float = 1.5, end_x: float = 9.5,
                 half_width_x: float = 0.075) -> str:
    """平らな上面を持つ低いハードル（段差）を生成する。"""
    lines = []
    positions = np.arange(start_x, end_x, spacing)
    for i, rx in enumerate(positions):
        lines.append(
            f'    <geom name="hurdle_{i}" type="box" '
            f'size="{half_width_x} 2.5 {height/2:.4f}" '
            f'pos="{rx:.2f} 0 {height/2:.4f}" '
            f'rgba="0.60 0.40 0.20 1" contype="1" conaffinity="1"/>')
    return "\n".join(lines) + "\n"


def build_level0_xml(slope_deg: float = 0.0) -> str:
    return (_header_xml()
            + "\n  <worldbody>"
            + _lights_xml(slope_deg)
            + _goal_marker_xml(GOAL_INIT_X)
            + _robot_bodies_xml()
            + "\n  </worldbody>\n"
            + _actuators_xml()
            + "\n</mujoco>")


def build_stairs_xml() -> str:
    return (_header_xml()
            + "\n  <worldbody>"
            + _lights_xml(0.0)
            + _goal_marker_xml(STAIRS_GOAL_X, STAIR_TOP_Z)
            + _stairs_xml()
            + _robot_bodies_xml()
            + "\n  </worldbody>\n"
            + _actuators_xml()
            + "\n</mujoco>")


def build_level1_xml() -> str:
    wall = _wall_segments_xml(wall_x=4.0, gap_center_y=0.0, gap_width=0.75, name_prefix="wall1")
    return (_header_xml()
            + "\n  <worldbody>"
            + _lights_xml()
            + _goal_marker_xml(GOAL_INIT_X)
            + wall
            + _robot_bodies_xml()
            + "\n  </worldbody>\n"
            + _actuators_xml()
            + "\n</mujoco>")


def build_level2_xml() -> str:
    wall1 = _wall_segments_xml(wall_x=4.0, gap_center_y=1.4, gap_width=0.8, name_prefix="wall1")
    wall2 = _wall_segments_xml(wall_x=8.0, gap_center_y=-1.4, gap_width=0.8, name_prefix="wall2")
    cubes = ""
    for i in range(N_CUBES):
        cubes += f"""
    <body name="cube_{i}" pos="{20 + i*5} 0 1.0">
      <freejoint name="cube_{i}_joint"/>
      <geom name="cube_{i}_geom" type="box" size="0.22 0.22 0.22" material="mat_cube"
            contype="2" conaffinity="2" mass="3"/>
    </body>"""
    return (_header_xml()
            + "\n  <worldbody>"
            + _lights_xml()
            + _goal_marker_xml(GOAL_INIT_X)
            + wall1
            + wall2
            + cubes
            + _robot_bodies_xml()
            + "\n  </worldbody>\n"
            + _actuators_xml()
            + "\n</mujoco>")


def build_xml(level: int, slope_deg: float = 0.0, stairs: bool = False) -> str:
    if stairs:
        return build_stairs_xml()
    if level == 0:
        return build_level0_xml(slope_deg)
    return {1: build_level1_xml, 2: build_level2_xml}[level]()


# ---------------------------------------------------------------------------
# Gymnasium 環境
# ---------------------------------------------------------------------------

class BlockyWalkerEnv(gym.Env):
    """
    ブロック型3Dヒューマノイドのカリキュラム学習環境。

    共通ルール（全レベル）:
      - ゴール（緑パネル）に向かって走る。到達したら次のゴールが出現。
      - 転倒（torso z < MIN_Z）→ 大ペナルティ＋エピソード終了
      - 手（前腕）が床に触れる → ステップごとにペナルティ
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    LEVEL_NAMES = ["FLAT", "WALL_GAP", "DODGE"]

    def __init__(self, level: int = 0, render_mode: str | None = None,
                 slope_deg: float = 0.0, stairs: bool = False,
                 terrain_vision: bool = False):
        super().__init__()
        assert 0 <= level <= 2
        self.level = level
        self.render_mode = render_mode
        self._slope_deg = slope_deg
        self._slope_tan = math.tan(math.radians(slope_deg))
        self._stairs = stairs
        self._terrain_vision = terrain_vision

        xml = build_xml(level, slope_deg, stairs)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data  = mujoco.MjData(self.model)

        # ---- ボディ / ジオム ID ----
        def _body(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        def _geom(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

        self._torso_id      = _body("torso")
        self._torso_geom_id = _geom("torso_box")
        self._floor_geom_id = _geom("floor")

        # 足ジオム（交互歩行報酬用）
        self._left_foot_geom_id  = _geom("left_foot_geom")
        self._right_foot_geom_id = _geom("right_foot_geom")

        # 股関節 qpos アドレス（腿上げ報酬用）
        def _joint(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        self._left_hip_qpos  = self.model.jnt_qposadr[_joint("left_hip")]
        self._right_hip_qpos = self.model.jnt_qposadr[_joint("right_hip")]

        # 腕ジオム（床接触ペナルティ用）
        self._arm_geom_ids = {
            _geom("left_upper_arm_geom"),
            _geom("left_forearm_geom"),
            _geom("right_upper_arm_geom"),
            _geom("right_forearm_geom"),
        }
        # 脛ジオム（膝つきペナルティ用）
        self._shin_geom_ids = {
            _geom("left_shin_geom"),
            _geom("right_shin_geom"),
        }

        # ゴールマーカー（モカップ）
        goal_body_id = _body("goal_marker")
        self._goal_mocap_idx = self.model.body_mocapid[goal_body_id]

        # Level 2: キューブ
        self._cube_body_ids  = []
        self._cube_geom_ids  = set()
        if level == 2:
            for i in range(N_CUBES):
                bid = _body(f"cube_{i}")
                gid = _geom(f"cube_{i}_geom")
                self._cube_body_ids.append(bid)
                self._cube_geom_ids.add(gid)

        # ---- 観測・行動空間 ----
        obs_dim = OBS_DIM + (N_HEIGHT_SAMPLES if terrain_vision else 0)
        obs_limit = np.full(obs_dim, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_limit, obs_limit, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(N_ACTS,), dtype=np.float32)

        # ---- 内部状態 ----
        self._goal_x        = GOAL_INIT_X
        self._prev_dist     = GOAL_INIT_X
        self._step_count    = 0
        self._wall1_passed  = False
        self._wall2_passed  = False
        self._renderer      = None
        self._last_raised_hip = None  # 最後に上げた腿: 'left' / 'right' / None

    # ------------------------------------------------------------------
    # 観測
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        qpos_obs = self.data.qpos[2:17].astype(np.float32)   # (15,)
        qvel_obs = self.data.qvel[:16].astype(np.float32)     # (16,)

        torso_pos = self.data.xpos[self._torso_id]
        dist_to_goal = max(0.0, self._goal_x - torso_pos[0])

        # 腕接触フラグ
        hand_flag = float(self._arm_on_ground())

        if self.level == 0:
            obs4 = np.array([dist_to_goal / GOAL_INIT_X, hand_flag, 0.0, 0.0], dtype=np.float32)
        elif self.level == 1:
            dx_wall = np.clip((4.0 - torso_pos[0]) / 8.0, -1.0, 1.0)
            obs4 = np.array([dist_to_goal / GOAL_INIT_X, hand_flag, dx_wall, 0.0], dtype=np.float32)
        else:
            dx_w1 = np.clip((4.0 - torso_pos[0]) / 8.0, -1.0, 1.0)
            dx_w2 = np.clip((8.0 - torso_pos[0]) / 8.0, -1.0, 1.0)
            obs4 = np.array([dist_to_goal / GOAL_INIT_X, hand_flag, dx_w1, dx_w2], dtype=np.float32)

        base_obs = np.concatenate([qpos_obs, qvel_obs, obs4])

        if self._terrain_vision:
            torso_x = self.data.xpos[self._torso_id][0]
            heights = np.array([
                self._terrain_z(torso_x + offset)
                for offset in HEIGHT_SAMPLE_OFFSETS
            ], dtype=np.float32)
            base_obs = np.concatenate([base_obs, heights])

        return np.clip(base_obs, -10.0, 10.0)

    def _terrain_z(self, torso_x: float) -> float:
        """現在のx位置での地形の高さを返す。"""
        if self._stairs:
            if torso_x < STAIR_START_X:
                return 0.0
            elif torso_x < STAIR_END_X:
                step_num = min(int((torso_x - STAIR_START_X) / STAIR_DEPTH), N_STAIRS - 1)
                return (step_num + 1) * STAIR_HEIGHT
            else:
                return STAIR_TOP_Z
        return torso_x * self._slope_tan

    def _is_healthy(self) -> bool:
        torso_x = max(0.0, self.data.xpos[self._torso_id][0])
        floor_z = self._terrain_z(torso_x)
        height_above_floor = self.data.qpos[2] - floor_z
        return MIN_Z < height_above_floor < MAX_Z

    def _arm_on_ground(self) -> bool:
        """前腕 or 上腕が床に接触しているか。"""
        for c in range(self.data.ncon):
            ct = self.data.contact[c]
            g1, g2 = ct.geom1, ct.geom2
            if (g1 == self._floor_geom_id and g2 in self._arm_geom_ids) or \
               (g2 == self._floor_geom_id and g1 in self._arm_geom_ids):
                return True
        return False

    def _foot_contacts(self) -> tuple[bool, bool]:
        """左右の足が床に接触しているか (left, right)。"""
        left, right = False, False
        for c in range(self.data.ncon):
            ct = self.data.contact[c]
            g1, g2 = ct.geom1, ct.geom2
            if g1 == self._floor_geom_id or g2 == self._floor_geom_id:
                other = g2 if g1 == self._floor_geom_id else g1
                if other == self._left_foot_geom_id:
                    left = True
                if other == self._right_foot_geom_id:
                    right = True
        return left, right

    def _shin_on_ground(self) -> bool:
        """脛（膝下）が床に接触しているか。"""
        for c in range(self.data.ncon):
            ct = self.data.contact[c]
            g1, g2 = ct.geom1, ct.geom2
            if (g1 == self._floor_geom_id and g2 in self._shin_geom_ids) or \
               (g2 == self._floor_geom_id and g1 in self._shin_geom_ids):
                return True
        return False

    # ------------------------------------------------------------------
    # Level 2: キューブ管理
    # ------------------------------------------------------------------
    def _spawn_cube(self, cube_idx: int, ahead_dist: float):
        body_id = self._cube_body_ids[cube_idx]
        jnt_id  = self.model.body_jntadr[body_id]
        qp = self.model.jnt_qposadr[jnt_id]
        qv = self.model.jnt_dofadr[jnt_id]
        torso_x = self.data.xpos[self._torso_id][0]
        sx = torso_x + ahead_dist
        sy = self.np_random.uniform(-1.2, 1.2)
        sz = self.np_random.uniform(0.5, 1.3)
        self.data.qpos[qp:qp+7] = [sx, sy, sz, 1.0, 0.0, 0.0, 0.0]
        speed = self.np_random.uniform(2.0, 3.5)
        self.data.qvel[qv:qv+6] = [-speed, self.np_random.uniform(-0.4, 0.4), -0.1, 0, 0, 0]

    def _manage_cubes(self) -> float:
        penalty = 0.0
        torso_x = self.data.xpos[self._torso_id][0]
        for c in range(self.data.ncon):
            ct = self.data.contact[c]
            g1, g2 = ct.geom1, ct.geom2
            if ((g1 in self._cube_geom_ids and g2 == self._torso_geom_id) or
                    (g2 in self._cube_geom_ids and g1 == self._torso_geom_id)):
                penalty -= 3.0
        for i, bid in enumerate(self._cube_body_ids):
            if self.data.xpos[bid][0] < torso_x - 2.0:
                self._spawn_cube(i, ahead_dist=self.np_random.uniform(8.0, 14.0))
        return penalty

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 胴体：固定位置・直立姿勢（少し高めにして足のめり込みを防ぐ）
        self.data.qpos[0:3] = [0.0, 0.0, 1.05]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # 関節角度をランダム初期化（各関節の有効範囲の一部でサンプリング）
        # 順序: left_shoulder, left_elbow, right_shoulder, right_elbow,
        #       left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle
        joint_ranges = [
            (-0.3,  0.3),   # left_shoulder  (range: -2.0 ~ 1.0)
            (-0.4,  0.0),   # left_elbow     (range: -1.8 ~ 0.1)
            (-0.3,  0.3),   # right_shoulder (range: -2.0 ~ 1.0)
            (-0.4,  0.0),   # right_elbow    (range: -1.8 ~ 0.1)
            (-0.3,  0.1),   # left_hip       (range: -1.5 ~ 0.3)
            ( 0.0,  0.4),   # left_knee      (range: -0.1 ~ 1.8)
            (-0.15, 0.15),  # left_ankle     (range: -0.6 ~ 0.6)
            (-0.3,  0.1),   # right_hip      (range: -1.5 ~ 0.3)
            ( 0.0,  0.4),   # right_knee     (range: -0.1 ~ 1.8)
            (-0.15, 0.15),  # right_ankle    (range: -0.6 ~ 0.6)
        ]
        for i, (lo, hi) in enumerate(joint_ranges):
            self.data.qpos[7 + i] = self.np_random.uniform(lo, hi)

        # 速度もランダム化（小さめ）
        self.data.qvel[:] = self.np_random.uniform(-0.05, 0.05, self.model.nv)

        # ゴールをリセット
        self._goal_x    = STAIRS_GOAL_X if self._stairs else GOAL_INIT_X
        self._prev_dist = self._goal_x
        goal_z = STAIR_TOP_Z if self._stairs else self._goal_x * self._slope_tan
        self.data.mocap_pos[self._goal_mocap_idx] = [self._goal_x, 0.0, goal_z]

        mujoco.mj_forward(self.model, self.data)

        self._step_count   = 0
        self._wall1_passed = False
        self._wall2_passed = False
        self._left_contact  = False
        self._right_contact = False
        self._left_air_steps  = 0
        self._right_air_steps = 0
        self._last_raised_hip = None

        if self.level == 2:
            for i in range(N_CUBES):
                self._spawn_cube(i, ahead_dist=6.0 + i * 5.0)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

        torso_x   = self.data.xpos[self._torso_id][0]
        torso_z   = self.data.qpos[2]
        healthy   = self._is_healthy()

        # ---- ゴール進捗（距離更新）----
        curr_dist     = max(0.0, self._goal_x - torso_x)
        self._prev_dist = curr_dist

        # ---- 前進速度報酬 ----
        vx = float(self.data.qvel[0])   # 根元ジョイントのx方向速度 (m/s)
        reward = vx * 2.0

        # ---- 直立維持報酬（前傾を抑制、±10度の遊びを許容）----
        # torso_mat[2, 2] = cos(傾き角)  1.0=直立, cos(10°)≈0.985 を閾値とする
        torso_mat = self.data.xmat[self._torso_id].reshape(3, 3)
        upright = float(torso_mat[2, 2])
        _upright_threshold = 0.9848  # cos(10°)
        if upright < _upright_threshold:
            # 10度を超えた傾きにのみペナルティ（最大 -0.5 程度）
            reward += (upright - _upright_threshold) * 0.5

        # ---- ゴール到達ボーナス ----
        if torso_x >= self._goal_x:
            reward += 200.0
            self._goal_x += GOAL_STEP_X
            goal_z = STAIR_TOP_Z if self._stairs else self._goal_x * self._slope_tan
            self.data.mocap_pos[self._goal_mocap_idx] = [self._goal_x, 0.0, goal_z]
            self._prev_dist = max(0.0, self._goal_x - torso_x)

        # ---- 手・膝・脛ペナルティ: なし ----

        # ---- 壁通過ボーナス (Level 1+) ----
        if self.level >= 1 and not self._wall1_passed and torso_x > 4.6:
            reward += 20.0
            self._wall1_passed = True

        # ---- 交互腿上げ報酬 (階段モード) ----
        # 左右を交互に腿を上げた瞬間にボーナス（両足同時はボーナスなし）
        if self._stairs:
            left_raised  = self.data.qpos[self._left_hip_qpos]  < -0.3
            right_raised = self.data.qpos[self._right_hip_qpos] < -0.3
            if left_raised and not right_raised:
                if self._last_raised_hip != 'left':
                    self._last_raised_hip = 'left'
                    reward += 2.0
            elif right_raised and not left_raised:
                if self._last_raised_hip != 'right':
                    self._last_raised_hip = 'right'
                    reward += 2.0

        # ---- Level 2: キューブ管理 ----
        if self.level == 2:
            if not self._wall2_passed and torso_x > 8.6:
                reward += 20.0
                self._wall2_passed = True
            reward += self._manage_cubes()

        # ---- 転倒ペナルティ: なし ----

        terminated = not healthy
        truncated  = self._step_count >= MAX_STEPS
        obs        = self._get_obs()
        info = {
            "torso_x": float(torso_x),
            "torso_z": float(torso_z),
            "goal_x":  float(self._goal_x),
            "level":   self.level,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="track")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
