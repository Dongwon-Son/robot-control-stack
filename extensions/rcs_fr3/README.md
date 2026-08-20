# RCS FR3 Extension

Support for the Franka Research 3 (FR3) robot in RCS.

This extension depends on [`rcs-core`](https://pypi.org/project/rcs-core/).
Documentation: <https://robotcontrolstack.org/extensions/rcs_fr3>

## Installation

Install the Debian dependency first:

```shell
sudo apt install $(cat debian_deps.txt)
```

Install from PyPI:

```shell
pip install rcs-fr3
```

Warning: plain `pip install rcs-fr3` will install the published `rcs-core` dependency from PyPI.

Install from a local checkout for development:

```shell
pip install -ve . --no-build-isolation
```

If you want this extension to use your local RCS checkout instead of the published `rcs-core` package, first install the main package from the repository root:

```shell
pip install -ve . --no-build-isolation
pip install -ve extensions/rcs_fr3 --no-build-isolation
```

For `libfranka` version details, see <https://robotcontrolstack.org/extensions/libfranka_versions>.

Add your FR3 Desk credentials to a `.env` file:

```env
DESK_USERNAME=...
DESK_PASSWORD=...
```

## Usage

```python
from rcs.envs.base import ControlMode, RelativeTo
from rcs_fr3.configs import DefaultFR3HardwareEnv

env_creator = DefaultFR3HardwareEnv()
env_creator.ip = "192.168.101.1"

cfg = env_creator.config()
cfg.control_mode = ControlMode.CARTESIAN_TQuat
cfg.camera_cfgs = None
cfg.max_relative_movement = 0.5
cfg.relative_to = RelativeTo.LAST_STEP

env = env_creator.create_env(cfg)
obs, info = env.reset()
print(env.get_wrapper_attr("robot").get_cartesian_position())
```

For a maintained end-to-end example, see [examples/fr3/fr3_env_cartesian_control.py](../../examples/fr3/fr3_env_cartesian_control.py).

## CLI

```shell
python -m rcs_fr3 --help
```

## TAM (Torque Adaptation Module) hook — `tam` branch

The `tam` branch adds a learned residual-torque hook to the async torque
controllers of `hw::Franka` (`osc()` and `joint_controller()`), used by the
[TAM](https://github.com/Dongwon-Son/TAM) sim-to-real stack:

```
tau_base = <RCS law>                     # gravity-free; libfranka adds gravity
tau_d    = tau_base + tam.apply(...)     # residual from a SimAdaptor MLP (Eigen, ~0.1 ms)
tau_d    = limitRate(...); clamp(torque_limit)   # unchanged RCS tail
tam.finalize_row(tau_d)                  # 1 kHz history row published to the workstation
```

* `src/hw/TamHook.{h,cpp}` — ring-buffer history, embedding hand-off, adaptor
  gating/ramp/clip; `src/hw/simadaptor.h` — the SimAdaptor network + `.bin`
  weight loader; `src/hw/tam_hook_test.cpp` — standalone unit test
  (`-DRCS_FR3_BUILD_TAM_HOOK_TEST=ON`, Eigen only).
* Python: `hw.Franka.tam_load_adaptor / tam_set_embedding / tam_enable /
  tam_set_ideal_model_has_gravity / tam_set_torque_limits / tam_get_history /
  tam_status / tam_reset`, plus a standalone `hw.TamHook` for sim backends and
  parity tests.
* The TAM history encoder runs in the same Python process
  (`extensions/rcs_tam`, `rcs_tam.TamDeployment`): it reads
  `robot.tam_get_history()`, computes the latent embedding with the vendored
  TAM inference code, and writes it back via `robot.tam_set_embedding()`.
  Single machine, no separate workstation; RCS core gains no new dependencies.
* Raise `FrankaConfig.torque_limit` (default 5 Nm on the whole gravity-free
  command) when using the hook, e.g. `[87,87,87,87,12,12,12]`.

`rcs_panda` materializes the same sources, so both extensions carry the hook.
