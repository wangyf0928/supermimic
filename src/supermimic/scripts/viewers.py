"""World model visualization viewers for SuperMimic.

Provides mixin and concrete viewer classes that overlay world model
predictions as ghost entities on top of the ground-truth simulation.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import torch
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from supermimic.models import Dynamics
from supermimic.representations import (
  MaxCoordRepresentation,
  StateRepresentation,
  extract_actor_goal,
)
from supermimic.sim_context import SimContext
from supermimic.tasks.task_spec import (
  ContinuousGoalSpec,
  DiscreteGoalSpec,
  TaskSpec,
  build_actor_obs_td,
)


def create_wm_vis_model(base_model: mujoco.MjModel) -> mujoco.MjModel:
  model = copy.deepcopy(base_model)
  model.geom_rgba[:, 3] = 0.8
  model.geom_rgba[:, 0:3] = [0.5, 0.5, 0.8]
  return model


class SuperMimicViewerMixin:
  def setup_supermimic(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    num_steps_rollout: int,
    sim_context: SimContext,
    task_specs: list[TaskSpec],
    initial_task: str = "motion",
  ) -> None:
    self.dynamics = dynamics
    self.representation = representation
    self.num_steps_rollout = num_steps_rollout
    self.sim_context = sim_context
    self.sim_ctx = sim_context.read(self.env.unwrapped)

    self.task_specs: dict[str, TaskSpec] = {ts.name: ts for ts in task_specs}
    self.set_task(initial_task)
    device = next(self.policy.parameters()).device
    self._raw_goal_state: dict[str, torch.Tensor] = {
      name: torch.zeros(1, ts.raw_goal_dim, device=device)
      for name, ts in self.task_specs.items()
      if ts.raw_goal_dim > 0
    }
    motion = self.env.unwrapped.command_manager.get_term("motion").motion
    self._replay_goal_tables: dict[str, torch.Tensor] = {
      name: ts.precompute_replay_goal(motion).to(device)
      for name, ts in self.task_specs.items()
      if ts.precompute_replay_goal is not None
    }
    self._replay_mode: dict[str, bool] = dict.fromkeys(self._replay_goal_tables, False)

    mj_model = self.env.unwrapped.sim.mj_model
    joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
    self.joint_q_adr = np.array(
      [mj_model.joint(n).qposadr[0] for n in joint_names if mj_model.joint(n).type != 0]
    )
    # Find free joint
    free_joint_id = next(
      (i for i in range(mj_model.njnt) if mj_model.joint(i).type == 0), -1
    )
    self.free_joint_q_adr = None
    if free_joint_id >= 0:
      adr = mj_model.joint(free_joint_id).qposadr[0]
      self.free_joint_q_adr = np.arange(adr, adr + 7)

    self.reset_env_supermimic()

    # For per-body tracking: map entity-local body indexes to MuJoCo body IDs
    if isinstance(self.representation, MaxCoordRepresentation):
      motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
      entity_body_indexes = motion_cmd.body_indexes
      mj_body_ids = motion_cmd.robot.indexing.body_ids[entity_body_indexes]
      self.mj_body_ids = mj_body_ids.cpu().numpy()

  def reset_env_supermimic(self) -> None:
    self.step_in_rollout = 0
    self._episode_starts = None
    self.curr_pred_rep_state = None
    self.curr_wm_aux = None
    self.last_actions = None
    self._gt_history = None
    self._clean_gt_history = None
    self._wm_history = None

  def set_task(self, name: str) -> None:
    """Switch the active task. Reroutes the actor through that task's encoder."""
    if name not in self.task_specs:
      raise ValueError(f"Unknown task {name!r}; have {list(self.task_specs)}")
    self.current_task = name
    self.policy.set_slot_mix({name: 1.0})

  def set_raw_goal(self, task_name: str, idx: int, value: float) -> None:
    """Update one dimension of a task's raw goal."""
    if task_name in self._raw_goal_state:
      self._raw_goal_state[task_name][0, idx] = value

  def _advance_history(
    self, history: torch.Tensor | None, sensor: torch.Tensor
  ) -> torch.Tensor:
    """Roll the policy's obs-history window forward by one frame and return it.
    A fresh window (``history is None``) repeats the current frame, matching the
    training reset padding; ``history_length == 1`` is just that single frame."""
    history_length = self.policy.history_length
    if history_length == 1:
      return sensor.unsqueeze(1)
    if history is None:
      return sensor.unsqueeze(1).repeat(1, history_length, 1)
    advanced = torch.cat([history[:, 1:], sensor.unsqueeze(1)], dim=1)
    return torch.where(self._episode_starts[:, None, None], sensor[:, None], advanced)

  def _play_actor_sensor(self, rep_state: torch.Tensor, **kwargs) -> torch.Tensor:
    """Apply noise once to the real-sim policy observation, before history."""
    sensor = self.representation.compute_actor_sensor(
      rep_state, self.sim_ctx, noisy=True, **kwargs
    )
    if self.representation.actor_obs_mode == "rep_state":
      sensor = self.representation.add_rep_state_noise(sensor)
    return sensor

  def _build_actor_goal(
    self, curr_goals: torch.Tensor, aux_state: torch.Tensor
  ) -> torch.Tensor:
    """Compute goal_obs for the current task.

    Motion uses the env's running motion goal; other tasks use slider state
    projected via the task's ``build_inference_goal_obs``. If replay mode
    is enabled for the active task, the slider is overridden by the
    precomputed motion-derived command at the current motion time.
    """
    if self.current_task == "motion":
      return extract_actor_goal(curr_goals)
    ts = self.task_specs[self.current_task]
    if self._replay_mode.get(self.current_task, False):
      motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
      table = self._replay_goal_tables[self.current_task]
      t = motion_cmd.time_steps.long().clamp(max=table.shape[0] - 1)
      raw = table[t]
    else:
      raw = self._raw_goal_state[self.current_task].expand(aux_state.shape[0], -1)
    return ts.build_inference_goal_obs(raw)

  def setup_wm_visualizer(
    self, wm_vis_model: mujoco.MjModel, *, update_every_n_frames: int | None = None
  ) -> None:
    self.wm_vis_model = wm_vis_model
    self.wm_vis_qpos = None
    self.wm_vis_mocap_pos = None
    self.wm_vis_mocap_quat = None
    self.wm_vis_xpos = None
    self.wm_vis_xquat = None

    # For per-body tracking: hide geoms belonging to non-tracked bodies (needed
    # for native viewer's mjv_addGeoms which renders all visible geoms).
    if isinstance(self.representation, MaxCoordRepresentation):
      tracked_set = set(self.mj_body_ids.tolist())
      for geom_id in range(wm_vis_model.ngeom):
        if wm_vis_model.geom_bodyid[geom_id] not in tracked_set:
          wm_vis_model.geom_rgba[geom_id, 3] = 0

    base_env = self.env.unwrapped
    original_vis_callback = base_env.update_visualizers

    def combined_vis_callback(viz):
      original_vis_callback(viz)
      if (
        update_every_n_frames is not None and self._counter % update_every_n_frames != 0
      ):
        return
      if self.wm_vis_xpos is None and self.wm_vis_qpos is None:
        return
      origins = base_env.scene.env_origins.cpu().numpy()
      episode_starts = self._episode_starts.cpu().numpy()
      for i in viz.get_env_indices(len(origins)):
        if episode_starts[i]:
          continue
        label = f"wm_prediction_{i}"
        if self.wm_vis_xpos is not None:
          viz.add_ghost_bodies(
            model=self.wm_vis_model,
            body_ids=self.mj_body_ids,
            body_xpos=self.wm_vis_xpos[i] + origins[i],
            body_xquat=self.wm_vis_xquat[i],
            label=label,
          )
        else:
          qpos = self.wm_vis_qpos[i].copy()
          mocap_pos = mocap_quat = None
          if self.free_joint_q_adr is not None:
            qpos[self.free_joint_q_adr[:3]] += origins[i]
          else:
            mocap_pos = self.wm_vis_mocap_pos[i] + origins[i]
            mocap_quat = self.wm_vis_mocap_quat[i]
          viz.add_ghost_entity(
            qpos,
            model=self.wm_vis_model,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            label=label,
          )

    base_env.update_visualizers = combined_vis_callback

  def reset_wm_visualizer(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is not None:
      if self.curr_wm_aux is not None:
        self.step_in_rollout[env_ids] = 0
        self._episode_starts[env_ids] = True
      return
    self.reset_env_supermimic()
    self.wm_vis_qpos = None
    self.wm_vis_mocap_pos = None
    self.wm_vis_mocap_quat = None
    self.wm_vis_xpos = None
    self.wm_vis_xquat = None

  def step_wm_visualizer(self, *, force: bool = False) -> bool:
    return self.step_and_predict(force=force)

  def _execute_step(self) -> bool:
    # BaseViewer uses this for both running and paused single-step.
    return self.step_wm_visualizer(force=self._is_paused)

  def step_and_predict(self, *, force: bool = False) -> bool:
    if self._is_paused and not force:
      return False

    try:
      with torch.no_grad():
        obs = self.env.get_observations()
        curr_gt_states, curr_goals = obs["full_state"], obs["goal"]

        gt_aux = self.representation.extract_aux_state(curr_gt_states)
        gt_rep_state = self.representation.represent(curr_gt_states)
        gt_actions = obs["actor"][:, -self.dynamics.num_actions :]
        if self.curr_wm_aux is None:
          self.step_in_rollout = gt_aux.new_zeros(gt_aux.shape[0], dtype=torch.long)
          self._episode_starts = torch.ones_like(self.step_in_rollout, dtype=torch.bool)
          self.curr_wm_aux = gt_aux
          self.curr_pred_rep_state = gt_rep_state
          self.sim_ctx = self.sim_context.read(self.env.unwrapped)
          self.last_actions = gt_actions.clone()
        else:
          restart = self.step_in_rollout == 0
          if restart.any():
            self.curr_wm_aux = torch.where(restart[:, None], gt_aux, self.curr_wm_aux)
            self.curr_pred_rep_state = torch.where(
              restart[:, None], gt_rep_state, self.curr_pred_rep_state
            )
            self.last_actions = torch.where(
              restart[:, None], gt_actions, self.last_actions
            )
            self.sim_ctx = torch.where(
              restart[:, None], self.sim_context.read(self.env.unwrapped), self.sim_ctx
            )
            if self._clean_gt_history is not None:
              self._wm_history = torch.where(
                restart[:, None, None], self._clean_gt_history, self._wm_history
              )

        # Actor step on ground-truth obs.
        actor_goal = self._build_actor_goal(curr_goals, gt_aux)
        proprio = self.representation.actor_obs_mode == "proprio"
        sensor = self._play_actor_sensor(
          gt_rep_state,
          sim_actor_obs=obs["actor"] if proprio else None,
        )
        self._gt_history = self._advance_history(self._gt_history, sensor)
        if self.policy.history_length > 1:
          clean_sensor = obs["actor"] if proprio else gt_rep_state
          self._clean_gt_history = self._advance_history(
            self._clean_gt_history, clean_sensor
          )
        actor_obs = build_actor_obs_td(
          {self.current_task: actor_goal}, self._gt_history
        )
        actions = self.policy(actor_obs)

        # Predict Next State (WM)
        motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
        wm_goal, _ = self.representation.compute_wm_goal(
          motion_cmd.motion,
          motion_cmd.time_steps,
          self.curr_wm_aux,
        )

        # Closed-loop actor step on the WM-predicted state.
        wm_actor_goal = self._build_actor_goal(wm_goal, self.curr_wm_aux)
        wm_sensor = self.representation.compute_actor_sensor(
          self.curr_pred_rep_state,
          self.sim_ctx,
          aux_state=self.curr_wm_aux,
          joint_state=(
            self.representation.joint_state(self.curr_pred_rep_state)
            if proprio
            else None
          ),
          last_action=self.last_actions,
          noisy=False,
        )
        self._wm_history = self._advance_history(self._wm_history, wm_sensor)
        wm_actor_obs = build_actor_obs_td(
          {self.current_task: wm_actor_goal}, self._wm_history
        )
        wm_actions = self.policy(wm_actor_obs)

        # Both actors consume this frame's goal before the real step advances it.
        _, _, dones, _ = self.env.step(actions)
        self._step_count += 1
        self._stats_steps += 1

        next_aux, next_rep_state = self.dynamics(
          self.curr_pred_rep_state, wm_actions, self.curr_wm_aux, self.sim_ctx
        )
        self.last_actions = wm_actions.clone().detach()

        self.curr_pred_rep_state = next_rep_state
        self.curr_wm_aux = next_aux

        # Visualization: reconstruct full state for ghost rendering
        full_state = self.representation.reconstruct_full_state(
          self.curr_wm_aux, self.curr_pred_rep_state
        )
        state = full_state.cpu().numpy()
        num_envs = state.shape[0]

        if isinstance(self.representation, MaxCoordRepresentation):
          B = self.representation.num_bodies
          self.wm_vis_xpos = state[:, : B * 3].reshape(num_envs, B, 3)
          self.wm_vis_xquat = state[:, B * 3 : B * 7].reshape(num_envs, B, 4)
        else:
          nq = self.env.unwrapped.sim.mj_model.nq
          qpos = np.zeros((num_envs, nq))
          root_pos = state[:, :3]
          root_quat = state[:, 3:7]
          if self.free_joint_q_adr is not None:
            qpos[:, self.free_joint_q_adr[:3]] = root_pos
            qpos[:, self.free_joint_q_adr[3:7]] = root_quat
            self.wm_vis_mocap_pos = None
            self.wm_vis_mocap_quat = None
            joint_state_start = 7
          else:
            # Fixed-base models may still need mocap pose for ghost visualization.
            self.wm_vis_mocap_pos = root_pos
            self.wm_vis_mocap_quat = root_quat
            joint_state_start = 0
          joint_state = state[
            :, joint_state_start : joint_state_start + len(self.joint_q_adr)
          ]
          qpos[:, self.joint_q_adr] = joint_state
          self.wm_vis_qpos = qpos

        self.step_in_rollout += 1
        self._episode_starts = dones.reshape(-1).bool().clone()
        self.step_in_rollout.masked_fill_(
          self._episode_starts | (self.step_in_rollout >= self.num_steps_rollout), 0
        )
      return True
    except Exception as exc:
      self._last_error = f"{type(exc).__name__}: {exc}"
      self.pause()
      return False


class SuperMimicNativeViewer(SuperMimicViewerMixin, NativeMujocoViewer):
  def __init__(
    self,
    env,
    actor,
    dynamics,
    representation,
    wm_vis_model,
    num_steps_rollout,
    sim_context: SimContext,
    task_specs: list[TaskSpec],
    initial_task: str = "motion",
    **kwargs,
  ):
    super().__init__(env, actor, **kwargs)
    self.setup_supermimic(
      dynamics,
      representation,
      num_steps_rollout,
      sim_context,
      task_specs,
      initial_task,
    )
    self.setup_wm_visualizer(wm_vis_model)

  def reset_environment(self) -> None:
    super().reset_environment()
    self.reset_wm_visualizer()


class SuperMimicViserViewer(SuperMimicViewerMixin, ViserPlayViewer):
  _clip_indices: list[int] | None = None

  def __init__(
    self,
    env,
    actor,
    dynamics,
    representation,
    wm_vis_model,
    num_steps_rollout,
    motion_names,
    sim_context: SimContext,
    task_specs: list[TaskSpec],
    initial_task: str = "motion",
    **kwargs,
  ):
    super().__init__(env, actor, **kwargs)
    self.setup_supermimic(
      dynamics,
      representation,
      num_steps_rollout,
      sim_context,
      task_specs,
      initial_task,
    )
    self.setup_wm_visualizer(wm_vis_model, update_every_n_frames=2)
    self.motion_names = motion_names
    self._selected_clip_idx = 0
    self._clip_dropdown = None
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    self._clip_indices = motion_cmd.clip_indices

  def setup(self) -> None:
    super().setup()
    self._scene.camera_tracking_enabled = self.env.num_envs == 1
    self._setup_task_widgets()
    self._setup_motion_clip_widget()

  def _setup_task_widgets(self) -> None:
    """Task dropdown + one folder of widgets per task with raw goals."""
    task_names = tuple(self.task_specs.keys())
    if len(task_names) > 1:
      task_dropdown = self._server.gui.add_dropdown(
        "Task",
        options=task_names,
        initial_value=self.current_task,
        order=1,
      )

      @task_dropdown.on_update
      def _(event) -> None:
        self.set_task(event.target.value)

    for task_name, ts in self.task_specs.items():
      if not ts.goal_specs:
        continue
      with self._server.gui.add_folder(f"{task_name.capitalize()} Command"):
        if task_name in self._replay_goal_tables:
          replay_box = self._server.gui.add_checkbox(
            "Replay (motion-derived)",
            initial_value=self._replay_mode[task_name],
          )

          @replay_box.on_update
          def _(_ev, _name=task_name, _ctrl=replay_box) -> None:
            self._replay_mode[_name] = bool(_ctrl.value)

        for i, spec in enumerate(ts.goal_specs):
          if isinstance(spec, DiscreteGoalSpec):
            options = tuple(f"{c:g}" for c in spec.choices)
            widget = self._server.gui.add_dropdown(
              spec.name, options, initial_value=f"{spec.default:g}"
            )
          else:
            assert isinstance(spec, ContinuousGoalSpec)
            widget = self._server.gui.add_slider(
              spec.name,
              min=spec.min,
              max=spec.max,
              step=spec.step,
              initial_value=spec.default,
            )

          @widget.on_update
          def _(_ev, _name=task_name, _idx=i, _ctrl=widget) -> None:
            self.set_raw_goal(_name, _idx, float(_ctrl.value))

          self.set_raw_goal(task_name, i, float(spec.default))

  def _setup_motion_clip_widget(self) -> None:
    if not self._supports_clip_selection():
      return
    options = self.motion_names
    if len(set(options)) != len(options):
      options = [f"{i + 1} · {name}" for i, name in enumerate(options)]
    multi = len(self._clip_indices or ()) > 1
    label = "Motion Clip"
    idx = 0
    if self._clip_indices:
      env_idx = self._scene.env_idx
      if multi:
        label = f"Motion Clip (env {env_idx})"
      idx = self._clip_indices[env_idx]
    self._clip_dropdown = self._server.gui.add_dropdown(
      label,
      options=options,
      initial_value=options[idx],
      hint="Choose a robot using Scene → Environment → Select." if multi else None,
      order=0,
    )

    @self._clip_dropdown.on_update
    def _(event) -> None:
      if self._clip_indices:
        # A server-side update reflects the selected robot's existing clip.
        if event.client_id is None:
          return
        self.request_action(
          "CUSTOM",
          {
            "type": "select_motion_clip",
            "env_idx": self._scene.env_idx,
            "clip_idx": options.index(event.target.value),
          },
        )
      else:
        self._selected_clip_idx = options.index(event.target.value)
        self.request_reset()

    shuffle_button = self._server.gui.add_button(
      "Randomize All Motions",
      hint="Start each robot on a randomly chosen different motion.",
      disabled=len(options) < 2,
      order=0.5,
    )

    @shuffle_button.on_click
    def _(_) -> None:
      self.request_action("CUSTOM", {"type": "randomize_motion_clips"})

  def _sync_motion_clip_widget(self) -> None:
    if self._clip_indices and self._clip_dropdown is not None:
      env_idx = self._scene.env_idx
      idx = self._clip_indices[env_idx]
      if len(self._clip_indices) > 1:
        self._clip_dropdown.label = f"Motion Clip (env {env_idx})"
      self._clip_dropdown.value = self._clip_dropdown.options[idx]

  def _update_env_dependent_plots(self) -> None:
    if self._scene.env_idx != self._prev_env_idx:
      self._sync_motion_clip_widget()
    super()._update_env_dependent_plots()

  def _handle_custom_action(self, action, payload) -> bool:
    if isinstance(payload, dict) and payload.get("type") == "randomize_motion_clips":
      motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
      if motion_cmd.randomize_clips():
        self.reset_environment()
        self._sync_motion_clip_widget()
      return True
    if isinstance(payload, dict) and payload.get("type") == "select_motion_clip":
      assert self._clip_indices is not None
      env_idx = payload["env_idx"]
      env_ids = torch.tensor([env_idx], device=self.env.device)
      with self._sim_lock:
        self._clip_indices[env_idx] = payload["clip_idx"]
        self.env.unwrapped.reset(env_ids=env_ids)
        self.reset_wm_visualizer(env_ids)
      self._sync_motion_clip_widget()
      return True
    return super()._handle_custom_action(action, payload)

  def reset_environment(self) -> None:
    super().reset_environment()
    self.reset_wm_visualizer()
    if not self._clip_indices and self._supports_clip_selection():
      with self._sim_lock:
        self._apply_selected_clip()

  def _handle_gui_reset(self, all_envs: bool) -> None:
    super()._handle_gui_reset(all_envs)
    if self._clip_indices and not all_envs:
      env_ids = torch.tensor([self._scene.env_idx], device=self.env.device)
      self.reset_wm_visualizer(env_ids)
    else:
      self.reset_wm_visualizer()
    self._sync_motion_clip_widget()

  def _supports_clip_selection(self) -> bool:
    return bool(self.motion_names)

  def _apply_selected_clip(self) -> None:
    """Override robot state to a random frame within the selected clip."""
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    motion = motion_cmd.motion

    idx = self._selected_clip_idx
    clip_start = 0 if idx == 0 else motion.motion_ends[idx - 1].item()
    clip_end = motion.motion_ends[idx].item()
    frame = torch.randint(clip_start, clip_end, (1,)).item()
    motion_cmd.reset_to_frame(torch.tensor([0], device=self.env.device), frame)

    base_env = self.env.unwrapped
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()
    base_env.sim.sense()
    base_env.observation_manager.compute(update_history=True)
