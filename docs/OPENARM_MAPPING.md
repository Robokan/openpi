# OpenArm ↔ OpenPI Observation/Action Mapping

This document explains how observations and actions are mapped between the OpenArm robot and the π₀ (Pi-Zero) VLA model.

## Overview

OpenArm is a **bimanual 7-DOF robot** (7 joints per arm + gripper), while the base π₀ model was trained on ALOHA (6-DOF per arm). The OpenPI framework uses **policy transforms** to convert between robot-specific formats and the model's internal representation.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  OpenArm Robot  │────▶│  OpenArmInputs   │────▶│    π₀ Model     │
│  (16 DOF state) │     │  (transform)     │     │  (internal)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│  OpenArm Robot  │◀────│  OpenArmOutputs  │◀─────────────┘
│  (16 DOF action)│     │  (transform)     │
└─────────────────┘     └──────────────────┘
```

## State/Action Format

### OpenArm (16 DOF)
```
Index:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
        └───────── Left Arm ─────────┘   └───────── Right Arm ────────┘
        │◀──── 7 joints ────▶│ grip │   │◀──── 7 joints ────▶│ grip │
```

| Index | Joint | Description |
|-------|-------|-------------|
| 0-6   | Left arm joints 1-7 | 7-DOF arm (includes elbow, unlike ALOHA's 6-DOF) |
| 7     | Left gripper | Gripper position (0 = closed, 0.044 = open in sim) |
| 8-14  | Right arm joints 1-7 | 7-DOF arm |
| 15    | Right gripper | Gripper position |

### Comparison with ALOHA (14 DOF)
```
ALOHA:   [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] = 14 DOF
OpenArm: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)] = 16 DOF
```

OpenArm has **one extra joint per arm** (joint 7), enabling human-like elbow movement.

## Camera Mapping

### Client → Server (Observation)
The client sends images with these keys:

| Client Key | Description | Required |
|------------|-------------|----------|
| `cam_high` | Overhead/base camera | Yes |
| `cam_left_wrist` | Left wrist-mounted camera | Optional |
| `cam_right_wrist` | Right wrist-mounted camera | Optional |

### Internal Model Mapping
The `OpenArmInputs` transform converts camera names:

```python
"cam_high"        → "base_0_rgb"
"cam_left_wrist"  → "left_wrist_0_rgb"
"cam_right_wrist" → "right_wrist_0_rgb"
```

### Image Format
- **Client sends**: `[C, H, W]` format (channels first), `uint8` (0-255)
- **Transform converts to**: `[H, W, C]` format (channels last) for the model
- **Resolution**: 224×224 pixels

## Observation Dictionary

### What the Client Sends
```python
observation = {
    "state": np.array([...]),  # Shape: (16,) - current joint positions
    "images": {
        "cam_high": np.array([...]),        # Shape: (3, 224, 224), uint8
        "cam_left_wrist": np.array([...]),  # Shape: (3, 224, 224), uint8
        "cam_right_wrist": np.array([...]), # Shape: (3, 224, 224), uint8
    },
    "prompt": "pick up the red cube",  # Natural language task description
}
```

### After OpenArmInputs Transform
```python
model_input = {
    "state": np.array([...]),  # Shape: (16,) - unchanged
    "image": {
        "base_0_rgb": np.array([...]),        # Shape: (224, 224, 3)
        "left_wrist_0_rgb": np.array([...]),  # Shape: (224, 224, 3)
        "right_wrist_0_rgb": np.array([...]), # Shape: (224, 224, 3)
    },
    "image_mask": {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": True,
    },
    "prompt": "pick up the red cube",
}
```

## Action Output

### Model Output
The π₀ model outputs actions with shape `(action_horizon, 32)` where 32 is the padded action dimension.

### After OpenArmOutputs Transform
```python
# Slices to OpenArm's 16 DOF
actions = model_output["actions"][:, :16]  # Shape: (action_horizon, 16)
```

### What the Client Receives
```python
response = {
    "actions": np.array([...]),  # Shape: (action_horizon, 16)
    "server_timing": {...},      # Performance metrics
}
```

The client typically uses `action_horizon=10` and executes actions one at a time.

## Delta Actions

During **training**, delta actions are used for arm joints (grippers stay absolute):

```python
# Delta action mask: True = delta, False = absolute
delta_action_mask = [True]*7 + [False] + [True]*7 + [False]
#                   └─left arm─┘  grip   └─right arm┘  grip
```

This means:
- **Arm joints (0-6, 8-14)**: Trained as deltas (action = change in position)
- **Grippers (7, 15)**: Trained as absolute positions

During **inference**, the `AbsoluteActions` transform converts deltas back to absolute positions.

## Normalization

State and action values are normalized using statistics computed from the training data:

```
normalized = (value - mean) / std
```

The normalization stats are stored in:
```
assets/pi05_openarm_ngc_lora/openarm/norm_stats.json
```

## Isaac Lab Simulation Specifics

When running in Isaac Lab simulation:

### Action Space: VLA (16 DOF) vs Sim (18 DOF)

**VLA outputs 16 DOF**:
```
[left_arm(7), left_grip(1), right_arm(7), right_grip(1)]
```

**Sim expects 18 DOF** (due to 2 finger joints per gripper):
```
[left_arm(7), left_grip(2), right_arm(7), right_grip(2)]
```

The OpenArm gripper has 2 finger joints that move together (mimic). The environment config uses a wildcard pattern `openarm_left_finger_joint.*` which matches both finger joints:

```python
# joint_pos_env_cfg.py - TeleopActionsCfg
self.actions.left_gripper_action = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["openarm_left_finger_joint.*"],  # Matches 2 joints!
    scale=1.0,
    use_default_offset=False,
)
```

### Client-Side Expansion (16 → 18 DOF)

The `openpi_client_bimanual.py` expands VLA output to match the environment:

```python
# VLA: [left_arm(7), left_grip(1), right_arm(7), right_grip(1)] = 16 DOF
# Env: [left_arm(7), left_grip(2), right_arm(7), right_grip(2)] = 18 DOF
left_arm = vla_actions[0:7]
left_grip = vla_actions[7]   # Single value
right_arm = vla_actions[8:15]
right_grip = vla_actions[15]  # Single value

expanded_actions = np.concatenate([
    left_arm,
    [left_grip, left_grip],   # Duplicate for both finger joints
    right_arm,
    [right_grip, right_grip], # Duplicate for both finger joints
])  # Shape: (18,)
```

### Why Teleop Uses Direct Robot Control

The `teleop_bimanual.py` script **bypasses `env.step()` entirely** and directly controls the robot:

```python
# Direct articulation control
robot.set_joint_position_target(left_targets, joint_ids=left_gripper_ids)
robot.set_joint_position_target(right_targets, joint_ids=right_gripper_ids)
robot.set_joint_position_target(left_joint_des, joint_ids=left_arm_joint_ids)
robot.set_joint_position_target(right_joint_des, joint_ids=right_arm_joint_ids)
```

This approach was used for:
1. **IK Control**: Teleop uses inverse kinematics which needs direct articulation access
2. **Bypassing Command Manager**: `env.step()` triggers command resampling (unwanted during teleop)
3. **Flexibility**: Can target any joint configuration without env config constraints

### Gripper Value Scaling

When grippers are properly connected:
```python
# VLA outputs normalized [0, 1] where:
#   0 = closed
#   1 = open

# Isaac Lab finger joints use [0, 0.044] where:
#   0 = closed
#   0.044 = open

# Conversion:
gripper_sim = gripper_vla * 0.044
```

### Alternative: Bypass env.step() Like Teleop

If you can't modify the environment, bypass `env.step()` and control the robot directly:

```python
# Get robot articulation
robot = env.unwrapped.scene["robot"]

# Get joint IDs
left_arm_ids = robot.find_joints("openarm_left_joint.*")[0]
right_arm_ids = robot.find_joints("openarm_right_joint.*")[0]
left_gripper_ids = robot.find_joints("openarm_left_finger_joint.*")[0]
right_gripper_ids = robot.find_joints("openarm_right_finger_joint.*")[0]

# Apply VLA actions directly
robot.set_joint_position_target(vla_actions[0:7], joint_ids=left_arm_ids)
robot.set_joint_position_target(vla_actions[7:8] * 0.044, joint_ids=left_gripper_ids)
robot.set_joint_position_target(vla_actions[8:15], joint_ids=right_arm_ids)
robot.set_joint_position_target(vla_actions[15:16] * 0.044, joint_ids=right_gripper_ids)

# Step physics directly
robot.write_data_to_sim()
env.unwrapped.sim.step()
robot.update(env.unwrapped.sim.get_physics_dt())
```

This matches how `teleop_bimanual.py` works (see lines 3650-3830).

## Complete Data Flow Example

```
1. Client captures state and images from robot/sim
   ↓
2. Client sends observation dict with prompt to server
   ↓
3. Server receives via WebSocket (msgpack encoded)
   ↓
4. OpenArmInputs transform:
   - Converts image format [C,H,W] → [H,W,C]
   - Maps camera names to internal names
   - Passes state through unchanged
   ↓
5. Normalization (using norm_stats.json)
   ↓
6. Model transforms (tokenization, etc.)
   ↓
7. π₀ model inference
   ↓
8. Output transforms (de-tokenization)
   ↓
9. Unnormalization
   ↓
10. OpenArmOutputs transform:
    - Slices actions to 16 DOF
    - Converts deltas to absolute (if applicable)
    ↓
11. Server sends actions back via WebSocket
    ↓
12. Client applies actions to robot/sim
```

## File References

| File | Purpose |
|------|---------|
| `src/openpi/policies/openarm_policy.py` | Input/output transforms |
| `src/openpi/training/config.py` | Training configuration (LeRobotOpenArmDataConfig) |
| `assets/pi05_openarm*/openarm/norm_stats.json` | Normalization statistics |
| `packages/openpi-client/` | Client library for connecting to server |

## Troubleshooting

### "Invalid action shape, expected: 18, received: 16"
The environment expects 18 DOF (grippers have 2 finger joints each) but VLA outputs 16 DOF.
**Fix**: The client expands 16 → 18 DOF by duplicating gripper values. Ensure you're using the updated `openpi_client_bimanual.py`.

### "Invalid action shape, expected: 14, received: 16"
The environment's `ActionsCfg` only defines 14 DOF (arms only, no grippers).
**Fix**: Use `Isaac-Reach-OpenArm-Bi-Teleop-v0` which includes `TeleopActionsCfg` with gripper actions.

### "Normalization stats not found"
Copy `norm_stats.json` to the correct assets directory for your config.

### Camera images not working
Ensure images are:
- Shape: `(3, 224, 224)` - channels first
- Dtype: `uint8` (0-255)
- Keys: `cam_high`, `cam_left_wrist`, `cam_right_wrist`

### Grippers not moving
Check that:
1. Using `TeleopActionsCfg` environment (includes gripper actions)
2. Client is expanding 16 → 18 DOF correctly
3. Gripper values are in sim range [0, 0.044]

### Lift task has binary grippers
The Lift task uses `BinaryJointPositionActionCfg` (open/close only). For VLA which outputs continuous values, use `JointPositionActionCfg` instead.

## Key Files in OpenArm Isaac Lab Trainer

| File | Purpose |
|------|---------|
| `source/openarm/.../bimanual/reach/reach_env_cfg.py` | Base environment config (ActionsCfg defined here) |
| `source/openarm/.../bimanual/reach/config/joint_pos_env_cfg.py` | Teleop environment config (actions configured here) |
| `source/openarm/.../bimanual/lift/lift_env_cfg.py` | Lift task with gripper actions (binary) |
| `scripts/teleoperation/teleop_bimanual.py` | Teleop script that bypasses env.step() |
| `scripts/teleoperation/openpi_client_bimanual.py` | VLA client (currently uses env.step()) |
