# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd supermimic
```

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

## Resume training from network weights

Use `--runner.resume` to load `world_model.pt`, `actor.pt`, and `critic.pt` from
one checkpoint directory and continue training:

```bash
CUDA_VISIBLE_DEVICES=0 uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-G1 \
  --motion-file 'motions_npz/*.npz' --num-envs 1024 \
  --runner.resume 'logs/SuperMimic-GenCoord-G1/YYYY-MM-DD_HH-MM-SS' \
  --runner.run-name o1_resume
```

Replace the directory with your saved training directory. To load the latest
model artifact from a W&B run, replace that argument with:

```bash
--runner.resume 'wandb://ENTITY/PROJECT/RUN_ID'
```

All network parameters and their saved normalization statistics are restored.
Actor-critic training requires all three files; BPTT training only needs the
world model and actor. Loading finishes before collecting data or updating any
network. Use the same task, actor observation mode, history length, task settings,
and network sizes as the source checkpoint; resume does not replace your current
training configuration. Weight/context mismatches raise an error.

This is a **weight resume**: optimizers, replay buffer, reset sampler, and step
counters start fresh, and a new log directory and W&B run are created. You can
set a new learning rate or `--runner.max-iterations` for this additional training.
The previous `--runner.wandb-run-path` option still loads only the world model;
do not combine it with `--runner.resume`.

## Domain randomization

Training and play use the same settings in
[`src/supermimic/tasks/domain_randomization.py`](src/supermimic/tasks/domain_randomization.py).
Edit the `DomainRandomizationCfg` defaults used by `DR_CONFIG` to adjust ranges:

| Setting | Default | Scope |
| --- | --- | --- |
| `kp_scale`, `kd_scale` | `(0.75, 1.25)` | Each of the 29 joint PD gains |
| `armature_scale` | `(0.75, 1.25)` | Each of the 29 joint armatures |
| `mass_scale` | `(0.75, 1.25)` | Each of the 30 robot links; inertia scales with mass |
| `com_x`, `com_y`, `com_z` | `(-0.03, 0.03)` m | Each link's local COM offset |
| `foot_friction` | `(0.3, 1.2)` | Shared across one robot's foot collision geoms |
| `encoder_bias` | `(-0.01, 0.01)` rad | Each joint |

Every environment independently resamples these uniform distributions on its
first reset and **every subsequent reset**, including automatic resets and play
resets. Joint/link samples are independent. Changes always use the original model
as their baseline, so repeated resets do not accumulate offsets or multipliers.
The all-link COM event replaces the old torso-only `base_com` event. Set a scale
range to `(1.0, 1.0)` or an offset range to `(0.0, 0.0)` to remove that perturbation.
Restart training/play after editing the file.

The world model now receives **237 context channels**: foot friction (1), encoder
bias (29), kp/kd scales (58), armature scales (29), link mass scales (30), and local
COM offsets (90). Context is read from the actual simulator parameters before
each collected step, normalized, and stored with that transition. Replay windows
stay within an episode; imagined rollouts use their sampled starting state's
context. The critic receives the same context; the actor's input dimensions are unchanged.
Mass and inertia share one multiplier, so no separate inertia context is needed.

New checkpoints save the context layout and training normalization. Play can
use different sampling ranges while retaining that saved normalization. Older
checkpoints with the previous context are unsupported; start a new training run.
Training and play commands are otherwise unchanged.

## Playback observation noise

Play enables observation noise at the real-simulation policy input. The
world-model viewer's policy observations remain noise-free. Noise is applied once
per new real observation, before adding it to the history window. The simulator's
separate actor corruption is disabled to avoid applying noise twice. Clean
dynamics states and motion goals are preserved, and checkpoint input dimensions
stay the same.

The ranges come from `PROPRIO_NOISE_HALF_WIDTHS` in
[`representations.py`](src/supermimic/representations.py): uniform gravity noise
of ±0.05, base angular velocity ±0.2 rad/s, joint position ±0.01 rad, and joint
velocity ±0.5 rad/s. Actions have no added noise. All four apply to `proprio`
policies and to GenCoord `rep_state` policies. GenCoord height and linear velocity
have no configured noise. MaxCoord `rep_state` uses the gravity and root angular
velocity noise; joint noise is not directly mapped onto its body pose channels.

Startup reports `Observation noise: real-sim policy ON (...), WM viewer policy OFF`.
No extra command-line flag is needed. This playback change preserves the existing
training noise behavior (`rep_state` clean, `proprio` noisy).

## Convert Westlake O1 motions

Convert the 50 Hz `westlake.o1_29dof` Joblib `.pkl` exports to G1 reference
motions locally:

```bash
uv run --script src/supermimic/scripts/convert_motions.py \
  --input motions --output motions_npz
```

The script declares its own NumPy, SciPy, and Joblib dependencies, so conversion
can run without installing the simulator. An environment with those packages can
also run it with `python` directly. Only load trusted pickle files.

Each output NPZ contains `root_pos_w`, `root_quat_w` (wxyz), `root_lin_vel_w`,
`root_ang_vel_w`, `joint_pos`, `joint_vel`, `joint_names`, and `fps`. The root
comes from body zero of `reset_body_*`; the joints come from `reset_joint_pos`
and `reset_joint_vel`. The exporter uses MuJoCo depth-first joint order, which
matches G1's joint order and axis signs. The O1 knee/elbow names are mapped to
G1 names. Optional input `joint_names` are used to reorder columns explicitly.
The `body_*` and `omni_body_*` skeletons are not used.

Input and output frame rates default to **50 Hz**, matching SuperMimic's default
control frequency. Existing velocities are preserved, and root quaternions are
normalized with continuous signs. `--input-fps` and `--output-fps` allow other
rates; changing the rate resamples positions/joints linearly and rotations with
SLERP, then recomputes world-frame velocities. `--recompute-velocities` can also
request this at the original rate. Match the output rate to the training
environment; the training loader does not resample based on the `fps` field.

`--input` accepts a file, a directory (including subdirectories), or a quoted
glob. Directory inputs preserve relative paths. Existing outputs are refused
unless `--overwrite` is supplied. `conversion_report.json` records the files,
frame counts, frame rates, and velocity handling. Original PKL files are kept.
This is a format conversion: root height, joint angles, and robot proportions
are not retargeted. Training computes body states using the G1 model.

Train on the converted clips with either SuperMimic task:

```bash
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --motion-file 'motions_npz/*.npz' --runner.run_name o1_motions
```

The O1 root correction is enabled directly in code: reset/root targets get
`+0.1 m` in world Z, while other body reference positions and joint states stay
unchanged. This is implemented directly in `MotionLoader` in
`../mjlab_research/src/mjlab/tasks/tracking/mdp/commands.py`: one `+= 0.1` in
`__init__` and one in `body_state`. Comment out both additions to disable it.
The original root is retained for FK so other bodies do not move. There is no
extra module, flag, or configuration. NPZ files are not rewritten. The training
viewer shows reference frames because a rigid ghost cannot depict this correction.

For nested inputs, use a matching glob or pack all clips into one directory with
`mjlab.scripts.pack_motions` before training.

## Preview converted motions

Open the converted NPZ clips in MuJoCo, using the same G1 model as training:

```bash
uv run --script src/supermimic/scripts/motion_vizz.py
```

The default input is this repository's `motions_npz/` directory. The standalone
script needs only MuJoCo and NumPy; an existing environment with both packages
can run it with `python` directly. GUI playback requires a desktop display.
It reads root poses and joint angles, maps columns by `joint_names`, and calls
MuJoCo forward kinematics to display the reference trajectory.

```bash
# Inspect one clip at half speed, initially paused.
python src/supermimic/scripts/motion_vizz.py \
  --motion-file motions_npz/walkslow1_36.npz --speed 0.5 --paused

# Validate every frame of every converted clip without opening a window.
python src/supermimic/scripts/motion_vizz.py --headless
```

Controls: **Space** pauses/resumes, **R** restarts, **Left/Right** steps and pauses,
**N/P** selects the next clip, **O** selects the previous clip, **+/-** changes
speed, **B** toggles the G1 skeleton, and **F** toggles camera following. Each
selected clip loops until switched. Mouse controls rotate/pan/zoom the camera;
turn following off to pan freely. The overlay shows the filename, frame, and
playback speed.

`--motion-file` also accepts a directory or quoted glob. `--fps` overrides the
file rate (50 Hz if absent); `--xml` overrides the automatically located G1 XML.
`--show-bodies` enables the G1 skeleton at startup, and `--no-follow` starts with
a free camera. Skeleton markers come from the G1 model's forward kinematics.
This standalone preview displays the original NPZ; training-only reference root
height corrections are displayed by policy playback using the same code above.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
