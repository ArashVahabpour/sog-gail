# Arash notes
This repository is blah blah blah... This is a generic code for gym environments with flat (1-dimensional) observation and action spaces.
Continuous latent spaces not implemented

```python train.py --name 3.adjust_scale.gradient_clip --env-name "Circles-v0" --use-gae --log-interval 1 --num-steps 2048 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 10000000 --use-linear-lr-decay --use-proper-time-limits --infogail --infogail-coef 0.1 --adjust-scale --gpu-id 3```

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
