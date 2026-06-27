# Adversarial Resilience in RL Systems

**Diagnosing which observations make a reinforcement-learning agent brittle — and how it fails when those signals are corrupted.**

RL agents often look robust on average yet collapse when a few observation channels are noised or adversarially perturbed. This project uses entropy-based feature analysis to (1) rank the state dimensions a trained policy depends on most and (2) measure how behavior degrades when those dimensions are attacked, across the **LunarLander** and **BipedalWalker** control tasks.

## Demo

![Clean, adversarial, and BipedalWalker rollouts](assets/adv-demo.gif)

▶️ [Watch the demo as MP4](assets/adversarial-rl-resilience-demo.mp4)

*Left → right: a clean LunarLander run, the same policy under adversarial ("imposter") observations, and a BipedalWalker rollout.*

## Results

| Clean policy | Adversarial / perturbed observations |
| :---: | :---: |
| ![Clean LunarLander](assets/clean-lunar-lander-run.gif) | ![Perturbed LunarLander](assets/lunar_lander_imposter4.gif) |

![BipedalWalker rollout](assets/bipedal_walker.gif)

- Entropy and joint-entropy rankings identify the observation channels a policy is most sensitive to.
- KL-divergence between clean and perturbed action distributions quantifies behavioral drift under attack.
- Robustness is treated as a measurable, diagnosable property — not just a single average-return number.

## Approach

- **Environments:** OpenAI Gym control tasks (LunarLander, BipedalWalker).
- **Feature analysis:** per-dimension entropy / joint-entropy to locate brittle state channels.
- **Perturbation model:** targeted noise and "imposter" corruption of selected observation dimensions.
- **Evaluation:** clean vs. noisy vs. adversarial rollouts, return curves, and action-distribution divergence.

## Tech stack

Python · OpenAI Gym · NumPy / Pandas · Matplotlib · notebook-based experiments

---

*Curated results showcase for a reinforcement-learning robustness study. Full training and evaluation code available on request.*
