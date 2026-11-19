"""Run a script with container packages ahead of optional user dependencies.

Invoke with ``python3 -s scripts/container_python.py SCRIPT [ARGS...]`` or
``python3 -s scripts/container_python.py -m MODULE [ARGS...]``. Unlike
site's default ordering, appending user packages cannot replace the container's
matched PyTorch/Triton installation. Spawned replay workers inherit this path.
User typing_extensions and protobuf supply compatibility for current W&B when
the container's older versions do not satisfy its dependencies.
"""
import importlib
import importlib.util
import runpy
import site
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: container_python.py SCRIPT [ARGS...] | -m MODULE [ARGS...]')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    user_site = Path(site.getusersitepackages())
    sys.path.append(str(user_site))
    import typing_extensions
    user_typing = user_site / 'typing_extensions.py'
    if not hasattr(typing_extensions, 'Sentinel') and user_typing.is_file():
        # A pure-Python compatibility backport; retain the container's priority
        # for compiled libraries, especially its matched PyTorch and Triton.
        spec = importlib.util.spec_from_file_location('typing_extensions', user_typing)
        module = importlib.util.module_from_spec(spec)
        sys.modules['typing_extensions'] = module
        spec.loader.exec_module(module)
    user_google = user_site / 'google'
    if (user_google / 'protobuf' / '__init__.py').is_file():
        import google
        import google.protobuf
        if int(google.protobuf.__version__.split('.')[0]) not in (4, 5, 6, 7):
            # W&B 0.26 ships generated messages for protobuf 4--7. Keep its
            # protobuf package and matching google._upb extension together.
            google.__path__ = [str(user_google)] + [path for path in google.__path__
                                                   if path != str(user_google)]
            sys.modules.pop('google.protobuf')
            importlib.import_module('google.protobuf')
    if sys.argv[1] == '-m':
        if len(sys.argv) < 3:
            raise SystemExit('-m requires a module name')
        module = sys.argv[2]
        sys.argv = sys.argv[2:]
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    else:
        script = sys.argv[1]
        sys.argv = sys.argv[1:]
        runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
