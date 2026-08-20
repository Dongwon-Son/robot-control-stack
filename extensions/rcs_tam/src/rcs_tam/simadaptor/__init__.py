"""Vendored TAM inference code (pruned public TAM release, imports rewritten).

Only the surfaces needed for deployment on robot-control-stack are included:
checkpoint loading and adaptor export (``deploy.inf_util``), the streaming
history encoder (``deploy.history_runtime``), and their model/physics
dependencies. Training, data generation, and transport code are not part of
this package.
"""

# Checkpoint pickles (save_dict.pkl) reference classes by their original module
# path ("simadaptor.config.train.TrainConfig", ...). Install a lazy import
# alias so any "simadaptor[.sub]" import resolves to this vendored package —
# no eager submodule imports, so package initialization order is unaffected.
# A real simadaptor installation, if present first on sys.path, wins.


def _install_alias_finder() -> None:
    import importlib
    import importlib.abc
    import importlib.machinery
    import sys

    vendored = __name__  # "rcs_tam.simadaptor"

    class _AliasLoader(importlib.abc.Loader):
        def __init__(self, real_name: str) -> None:
            self._real_name = real_name

        def create_module(self, spec):
            return importlib.import_module(self._real_name)

        def exec_module(self, module) -> None:
            pass

    class _VendoredSimadaptorAliasFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "simadaptor" and not fullname.startswith("simadaptor."):
                return None
            real_name = vendored + fullname[len("simadaptor"):]
            return importlib.machinery.ModuleSpec(fullname, _AliasLoader(real_name))

    if not any(type(f).__name__ == "_VendoredSimadaptorAliasFinder" for f in sys.meta_path):
        sys.meta_path.append(_VendoredSimadaptorAliasFinder())


_install_alias_finder()
