# Arash notes
This repository is blah blah blah... This is a generic code for gym environments with flat (1-dimensional) observation and action spaces.
Continuous latent spaces not implemented

```shell script
python train.py --name sog-pretrain-coef-0.1 --env-name Circles-v0 --sog-gail --sog-gail-coef 0.1 --latent-optimizer ohs --latent-dim 3 --gpu-id 1 --adjust-scale
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

TODO: review the installation requirements above

---
### Parallel jobs
1. Modify `jobs.xlsx`
2. Generate jobs as desired, e.g. ```python generate_tmux_yaml.py --num-seeds 4 --job-ids 0,1 --task 'benchmark'```
3. Run the jobs: ```tmuxp load run_all.yaml```

### Datasets
You can download a set of expert datasets from [this]() Google Drive directory and put them in `gail_experts` sub-directory.
Each file consists of torch tensors for states and actions of trajectories, as well as modes for each trajectory. 

---
### TODO
- the extra `DiagGaussian` + the `CirclesMLP` module to be moved out of the main code base
- take care of all TODO tags!
