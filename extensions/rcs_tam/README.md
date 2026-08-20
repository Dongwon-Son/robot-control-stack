# RCS TAM extension (`rcs_tam`) — single-process deployment

[TAM](https://github.com/Dongwon-Son/TAM) (Torque Adaptation Module) on
robot-control-stack, everything on **one machine in one Python process**:

```
one machine, one process
┌──────────────────────────────────────────────────────────────┐      ┌─────────────┐
│ Python: TamDeployment (JAX history encoder, ~5 Hz embeddings) │      │ Franka      │
│   robot.tam_get_history() ──► encoder ──► robot.tam_set_embedding()  │ FCI         │
│ Python: policy / example loop (20 Hz joint or Cartesian targets)│◄──▶│ 1 kHz torque│
│ C++ (RCS control thread): hw.Franka osc()/joint_controller()  │ FCI  │             │
│   + TamHook: tau_d += residual(local window, embedding)       │      │             │
└──────────────────────────────────────────────────────────────┘      └─────────────┘
```

No realtime kernel is required: the example opens the robot with
`FrankaConfig.ignore_realtime=True` (libfranka `kIgnore`), so a stock kernel
works — the 1 kHz callback then runs without RT scheduling, keep the machine
lightly loaded and prefer a machine with a GPU for the encoder (the fused
history encoder is ~20× slower than real time on CPU; embeddings then update
slower but the 1 kHz residual keeps using the latest one). Importing
`rcs_tam` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` (unless already set), so
the encoder allocates GPU memory on demand and coexists with a policy running
in the same process (e.g. torch); export the variable yourself to override.

## Install (after building `rcs-core` and `rcs_panda`/`rcs_fr3` from this branch)

```shell
pip install -e extensions/rcs_tam        # numpy, jax, flax, einops, tqdm, mujoco, mujoco-mjx
```

## Run

```shell
python extensions/rcs_tam/examples/franka_tam_direct.py \
    --robot panda --ip 192.168.0.52 --ckpt /path/to/tam_checkpoint \
    --motion sine --duration-s 30 --residual-clip 2
```

The script connects, exports the checkpoint's adaptor into the controller-side
`TamHook` (`robot.tam_load_adaptor`), starts the RCS joint controller and the
encoder thread, streams a sinusoidal joint reference, and enables TAM after the
first embedding. `--no-tam` runs the identical motion as a baseline.

Supported checkpoints (mode auto-resolved from the checkpoint):

| checkpoint | history_torque_mode |
| --- | --- |
| Panda-specific TAM | `applied` (one applied-torque history stream) |
| DAgger-finetuned | `applied` |
| DAgger + fused input | `base_tam_fusion` (applied/base/residual streams + linear fusion) |

The ideal-model MJCF (`panda_pandagripper.xml` + meshes, the exact model the
checkpoints were trained on) is packaged under `src/rcs_tam/assets/` and used
by default; `--xml` overrides it. The ideal model always includes gravity: the
hook feeds `tau + gravity` to the adaptor and the vendored inference code
rejects checkpoints trained otherwise.
Raise `FrankaConfig.torque_limit` (RCS default 5 Nm clips the residual; the
example uses 87/87/87/87/12/12/12) and keep the TAM residual clip
(`robot.tam_set_torque_limits`, default 10/10/10/10/4/4/4) small on first runs.

## Layout

| path | role |
| --- | --- |
| `src/rcs_tam/runtime.py` | `TamDeployment`: checkpoint load, mode resolution, bin export, encoder thread, controller-restart handling |
| `src/rcs_tam/simadaptor/` | vendored TAM inference code (checkpoint restore, streaming AR history encoder, models, ideal-model physics) — no external TAM dependency |
| `examples/franka_tam_direct.py` | the single-script example |
| `tests/test_runtime.py` | offline unit tests (fake robot/runtimes) |
| `tests/checkpoint_smoke.py` | robot-free end-to-end check of a real checkpoint through the real C++ `TamHook` |

The C++ side (`TamHook`, `hw.Franka.tam_*`) lives in `extensions/rcs_fr3/src/hw`
and is shared by `rcs_panda`.
