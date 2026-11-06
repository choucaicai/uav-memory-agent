# HiRaMem depth collision-termination evaluation

This optional evaluator uses the same depth checkpoint and YAML paths as
`eval.sh`, but terminates an episode on the first collision and marks it failed.
The recorded 23.2% result uses the standard evaluator, not this protocol.

`eval_hiramem_collision.sh` runs `eval_uavon_oct_collision.py` with
`collision_motion.py`. Motion is split into small position/yaw increments;
collision metadata, actual poses, and termination reasons are recorded.
Initial placement resets the collision baseline. Motion timeouts are errors.

```bash
CHECK_ONLY=1 bash eval_hiramem_collision.sh
bash eval_hiramem_collision.sh
```

Paths come from `configs/runtime.yaml` or `UAV_MEMORY_CONFIG`. The configured
checkpoint is evaluated with a 100-action limit and 20 m success threshold.
Results go under `<output_root>/collision/<run-name>--<checkpoint-name>/`.
`MODEL_PATH`, `EVAL_DATASET`, and `OUTPUT_ROOT` override these paths.

AirSim ports start at 30400; the torchrun master port is 29332 in the launcher.
The Python entry point also accepts `--collision_substep_m`,
`--collision_yaw_step_degrees`, `--collision_poll_seconds`,
`--motion_timeout_seconds`, and `--settle_seconds`.
