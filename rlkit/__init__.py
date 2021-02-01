from gym.envs.registration import register


register(id='HalfCheetahDir-v0',
         entry_point='rlkit.envs:HalfCheetahDirEnv',
         )

register(id='AntDir-v0',
         entry_point='rlkit.envs:AntDirEnv',
         )
