# import os
# import importlib
#
#
# ENVS = {}
#
#
# def register_env(name):
#     """Registers a env by name for instantiation in rlkit."""
#
#     def register_env_fn(fn):
#         if name in ENVS:
#             raise ValueError("Cannot register duplicate env {}".format(name))
#         if not callable(fn):
#             raise TypeError("env {} must be callable".format(name))
#         ENVS[name] = fn
#         return fn
#
#     return register_env_fn
#
#
# # automatically import any envs in the envs/ directory
# for file in os.listdir(os.path.join(os.path.dirname(__file__))):
#     try:
#         if file.endswith('.py') and not file.startswith('_'):
#             module = file[:file.find('.py')]
#             importlib.import_module('gym_sog.' + module)
#             print(file)
#     except Exception as e:
#         if 'mjpro131' in str(e):
#             print(f'{file} needs mujoco131')


from .half_cheetah_dir import HalfCheetahDirEnv
from .ant_dir import AntDirEnv
