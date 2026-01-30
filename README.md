# RECAP Policy for LeRobot

A LeRobot policy plugin implementing RECAP for PI0.5.

Based on Pi0.6 paper: https://www.pi.website/download/pistar06.pdf

## Architecture

Using SigLip from the PI0.5 model as a feature extractor, projecting  and appending the image features into the Gemma3-270M model. The final trainable <VAL> token is added at the end, and the output of the model is used to generate the value distribution.

## Installation

```bash
cd recap-lerobot
pip install -e .
```

# Training recap

## Libero-Pro
Evaluate on libero-pro.
```bash
pip install "libero @ git+https://github.com/stepanfeduniak/lerobot-libero-pro.git"
```
### Per Stage Run:
Collect Dataset:
```bash
python lerobot_policy_recap/reinforcement_loop/builds/data_collection/record_interaction_dataset.py \
  --dataset.repo_id=${HF_USER}/<dataset> \
  --output_dir=./outputs/[RUN_NAME] \
  --policy.repo_id=${HF_USER}/recap_pi \
  --policy.path=lerobot/pi05_libero_finetuned_quantiles \
  --env.type=libero \
  --env.task=libero_10 \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --eval.n_episodes=500 \
  --eval.batch_size=10 \
```    
One Training Iteration:

```bash
python lerobot_policy_recap/reinforcement_loop/builds/train/train_recap.py \
  --dataset.repo_id=${HF_USER}/<dataset> \
  --output_dir=./outputs/[RUN_NAME] \
  --policy.repo_id=${HF_USER}/recap_pi \
  --policy.path=lerobot/pi05_libero_finetuned_quantiles \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --critic_steps=5000 \
  --actor_steps=5000 \
  --batch_size=32
```
