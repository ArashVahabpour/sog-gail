# Arash notes
To train the Circles experiment, run 

```shell script
python train.py --name circles --env-name Circles-v0 --sog-gail --sog-gail-coef 0.1 --latent-optimizer ohs --save-dir /mnt/SSD3/arash/sog-gail/trained_models/ --results-root /mnt/SSD3/arash/sog-gail/results/ --latent-dim 3 --result-interval 10 --save-interval 5 --gpu-id 2 --adjust-scale --shared --seed 0
python train.py --name antdir --env-name AntDir-v0 --mujoco --sog-gail --sog-gail-coef 0.1 --latent-optimizer ohs --save-dir /mnt/SSD3/arash/sog-gail/trained_models/ --results-root /mnt/SSD3/arash/sog-gail/results/ --latent-dim 2 --result-interval 10 --save-interval 5 --gpu-id 2 --shared --seed 0
python train.py --name ad.s.shared.0.1x --env-name AntDir-v0 --mujoco --sog-gail --sog-gail-coef 0.01 --latent-optimizer ohs --save-dir /mnt/SSD3/arash/sog-gail/trained_models/ --results-root /mnt/SSD3/arash/sog-gail/results/ --latent-dim 2 --result-interval 10 --save-interval 5 --gpu-id 3 --shared --seed 0
python train.py --name ad.s.shared.10x --env-name AntDir-v0 --mujoco --sog-gail --sog-gail-coef 1 --latent-optimizer ohs --save-dir /mnt/SSD3/arash/sog-gail/trained_models/ --results-root /mnt/SSD3/arash/sog-gail/results/ --latent-dim 2 --result-interval 10 --save-interval 5 --gpu-id 4 --shared --seed 0
python train.py --name ad6.s.shared.10x.seed1 --env-name AntDir-v0 --mujoco --sog-gail --sog-gail-coef 1 --latent-optimizer ohs --save-dir /mnt/SSD3/arash/sog-gail/trained_models/ --results-root /mnt/SSD3/arash/sog-gail/results/ --latent-dim 6 --result-interval 10 --save-interval 5 --expert-filename trajs_antdir6.pt --gpu-id 0 --shared --seed 0
python train.py --name ad6.s.shared.5x --env-name AntDir-v0 --mujoco --sog-gail --sog-gail-coef 0.5 --latent-optimizer ohs --save-dir /mnt/SSD3/arash/sog-gail/trained_models/ --results-root /mnt/SSD3/arash/sog-gail/results/ --latent-dim 6 --result-interval 10 --save-interval 5 --expert-filename trajs_antdir6.pt --gpu-id 0 --shared --seed 0
python train.py --name ad6.s.shared.1x --env-name AntDir-v0 --mujoco --sog-gail --sog-gail-coef 0.1 --latent-optimizer ohs --save-dir /mnt/SSD3/arash/sog-gail/trained_models/ --results-root /mnt/SSD3/arash/sog-gail/results/ --latent-dim 6 --result-interval 10 --save-interval 5 --expert-filename trajs_antdir6.pt --gpu-id 0 --shared --seed 0
python train.py --name hcd.s.shared.10x --env-name HalfCheetahDir-v0 --mujoco --sog-gail --sog-gail-coef 1 --latent-optimizer ohs --save-dir /mnt/SSD3/arash/sog-gail/trained_models/ --results-root /mnt/SSD3/arash/sog-gail/results/ --latent-dim 2 --result-interval 10 --save-interval 5 --gpu-id 2 --shared --seed 0
python train.py --name hcv.s.shared.10x --env-name HalfCheetahVel-v0 --mujoco --sog-gail --sog-gail-coef 1 --latent-optimizer bcs --block-size 1 --save-dir /mnt/SSD3/arash/sog-gail/trained_models/ --results-root /mnt/SSD3/arash/sog-gail/results/ --latent-dim 1 --result-interval 10 --save-interval 5 --gpu-id 3 --shared --seed 0
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
