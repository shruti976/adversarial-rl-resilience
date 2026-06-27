# Adversarial Resilience in RL Systems

**Diagnosing which observations make a reinforcement-learning agent brittle — and how it fails when those signals are corrupted.**

RL agents often look robust on average yet collapse when a few observation channels are noised or adversarially perturbed. This project uses entropy-based feature analysis to (1) rank the state dimensions a trained policy depends on most and (2) measure how behavior degrades when those dimensions are attacked, across the **LunarLander** and **BipedalWalker** control tasks.

## Demo

![Clean, adversarial, and BipedalWalker rollouts](assets/adv-demo.gif)

*Left → right: a clean LunarLander run, the same policy under adversarial ("imposter") observations, and a BipedalWalker rollout.*

## Results

| Clean policy | Adversarial / perturbed observations |
| :---: | :---: |
| ![Clean LunarLander](assets/clean-lunar-lander-run.gif) | ![Perturbed LunarLander](assets/lunar_lander_imposter4.gif) |

- Entropy and joint-entropy rankings identify the observation channels a policy is most sensitive to.
- KL-divergence between clean and perturbed action distributions quantifies behavioral drift under attack.
- Robustness is treated as a measurable, diagnosable property — not just a single average-return number.

## Approach

- **Environments:** OpenAI Gym / Gymnasium control tasks (LunarLander, BipedalWalker).
- **Feature analysis:** per-dimension entropy / joint-entropy to locate brittle state channels.
- **Perturbation model:** targeted noise and "imposter" corruption of selected observation dimensions.
- **Evaluation:** clean vs. noisy vs. adversarial rollouts, return curves, and action-distribution divergence.

## Code

```text
code/lunarlander/
  Lunar_Lander_V2.ipynb       baseline agent training
  Render_Lunar_Lander.ipynb   record policy rollouts
  custom_envs/                LunarLander variants with one observation feature removed (LL1-LL8)
  entropy_methods/            entropy, joint-entropy, and KL feature-ranking experiments
code/bipedalwalker/           entropy / KL feature analysis + imposter-perturbation experiments
code/requirements.txt         dependencies
```

## Tech stack

Python · Gym / Gymnasium (Box2D) · Stable-Baselines3 · NumPy / Pandas / SciPy · SHAP · Matplotlib · Jupyter
