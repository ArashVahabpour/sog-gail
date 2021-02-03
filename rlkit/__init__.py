from gym.envs.registration import register


register(id='HalfCheetahDir-v0',
         entry_point='rlkit.envs:HalfCheetahDirEnv',
         )

register(id='HalfCheetahVel-v0',
         entry_point='rlkit.envs:HalfCheetahVelEnv',
         )

register(id='AntDir-v0',
         entry_point='rlkit.envs:AntDirEnv',
         )

register(id='AntGoal-v0',
         entry_point='rlkit.envs:AntGoalEnv',
         )

