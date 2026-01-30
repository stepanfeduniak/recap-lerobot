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

## Usage

```bash
# Train with RECAP
python -m lerobot_policy_recap.reinforcement_loop.builds.train.train_recap \
    --policy.type recap_pi \
    --policy.diffusion_repo_id lerobot/pi05_libero_finetuned
```

## License

Apache 2.0
