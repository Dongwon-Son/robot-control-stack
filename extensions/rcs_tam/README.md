# RCS TAM extension (`rcs_tam`)

NUC-side bridge for [TAM](https://github.com/Dongwon-Son/TAM) (Torque Adaptation
Module) on top of robot-control-stack. It serves the TAM history/embedding ZMQ
protocol — PUB history windows (`:5555`), async commands (`:5556`), reliable
commands (`:5557`) — over the `TamHook` that the `tam` branch adds to
`hw::Franka` (`rcs_panda` / `rcs_fr3`). The workstation side (history encoder,
`mapping_server.py`, clients, launchers) is unchanged.

```
workstation (GPU)  ──ZMQ──▶  NUC: python -m rcs_tam  ──pybind──▶  hw.Franka + TamHook  ──FCI──▶  Panda/FR3
 history encoder → z          rcs_tam.bridge / rcs_backend          osc()/joint_controller() + residual
```

## Install (NUC, after building `rcs-core` and `rcs_panda`/`rcs_fr3` from the `tam` branch)

```shell
pip install -e extensions/rcs_tam          # numpy + pyzmq only
python -m rcs_tam --robot panda --print-config
python -m rcs_tam --robot panda            # serves the protocol; holds the current pose with the RCS joint controller
```

Configuration: `--config rcs_tam_config.json` (schema in
`rcs_tam_config.example.json` / `rcs_tam/config.py`) or flags.
Important defaults: `FrankaConfig.torque_limit` is set to `87,87,87,87,12,12,12`
(RCS' own 5 Nm default clips the TAM residual); TAM residual clip
`10,10,10,10,8,8,8`; the adaptor is enabled by the workstation after the first
embedding.

## Modules

| module | role |
| --- | --- |
| `protocol.py` | wire format: history sample dict, async dedup/merge, `{"cmd": ...}` normalization, reset publish |
| `backend.py` | `BridgeBackend` interface (+ `history_rows_dict_to_samples`) |
| `bridge.py` | `TamBridge`: PUB/PULL/REP loop, resets, stale-stream neutralization, bin upload |
| `rcs_backend.py` | `RcsBackend`: `target_q → controller_set_joint_position`, Cartesian targets → `osc_set_cartesian_position`, TAM commands → `robot.tam_*`, reflex/force safety resets |
| `env_wrapper.py` | `TamBridgeWrapper(env, endpoints)`: run the bridge next to an RCS hardware gym env (policy via gym, TAM over ZMQ) |
| `config.py` | JSON + env configuration |

Protocol features without an RCS equivalent (SysID torque maps, external-torque
prediction, soft-block, test disturbances, feedforward torque, live gain
changes) answer `ok=False, unsupported=True`; disabling them is a no-op success.

## Tests

```shell
python -m pytest extensions/rcs_tam/tests -q     # fake backend, localhost ZMQ; no robot
```

A MuJoCo backend for hardware-free system tests (bridge + RCS torque-law
replica + `hw.TamHook`) lives in the TAM repository
(`simadaptor.deploy.rcs_mujoco_backend`, `scripts/deploy/rcs_tam_bridge_sim_smoke.py`).
