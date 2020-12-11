from gym.envs.registration import register


register(id='Circles-v0',
         entry_point='gym_sog.envs:CirclesEnv',
         )
