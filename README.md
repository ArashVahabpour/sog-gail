# SOG-GAIL
Commands for training different experiments 

```shell script
python train.py --name circles --env-name Circles-v0 --sog-gail --sog-gail-coef 0.1 --latent-optimizer ohs --latent-dim 3 --result-interval 10 --save-interval 5 --gpu-id 2 --adjust-scale --shared --seed 0
python train.py --name ant-fwd-back --env-name AntDir-v0 --mujoco --sog-gail --sog-gail-coef 1 --latent-optimizer ohs --latent-dim 2 --result-interval 10 --save-interval 5 --gpu-id 4 --shared --seed 0
python train.py --name ant-dir-6 --env-name AntDir-v0 --mujoco --sog-gail --sog-gail-coef 1 --latent-optimizer ohs --latent-dim 6 --result-interval 10 --save-interval 5 --expert-filename trajs_antdir6.pt --gpu-id 0 --shared --seed 0
python train.py --name halfcheetahdir --env-name HalfCheetahDir-v0 --mujoco --sog-gail --sog-gail-coef 1 --latent-optimizer ohs --latent-dim 2 --result-interval 10 --save-interval 5 --gpu-id 2 --shared --seed 0
python train.py --name halfcheetahvel --env-name HalfCheetahVel-v0 --mujoco --sog-gail --sog-gail-coef 1 --latent-optimizer bcs --block-size 1 --latent-dim 1 --result-interval 10 --save-interval 5 --gpu-id 3 --shared --seed 0
```

---

## Requirements

* Python 3
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```
