# Qualitative examples — vision_r2p1d_cv

0/47 test clips have every one of the 10 labels exactly right (rare by construction — 10 independent binary decisions per clip). Mean per-clip F1 = 0.40. Below: the clips the model got most right (by per-clip F1) and most wrong.

## Correct / strong examples
- **2TpC6CB0iXw_549_668** (clip F1 0.89) — true: Absence or Avoidance of Eye Contact, Aggressive Behavior, Hyper- or Hyporeactivity to Sensory Input, Non-Responsiveness to Verbal Interaction, Self-Hitting or Self-Injurious Behavior | predicted: Aggressive Behavior, Hyper- or Hyporeactivity to Sensory Input, Non-Responsiveness to Verbal Interaction, Self-Hitting or Self-Injurious Behavior
- **7lgAK1z-Scs_0_26** (clip F1 0.67) — true: Upper Limb Stereotypies | predicted: Hyper- or Hyporeactivity to Sensory Input, Upper Limb Stereotypies
- **3BQvkkTN3Sk_184_206** (clip F1 0.67) — true: Upper Limb Stereotypies | predicted: Hyper- or Hyporeactivity to Sensory Input, Upper Limb Stereotypies
- **3BQvkkTN3Sk_76_113** (clip F1 0.67) — true: Hyper- or Hyporeactivity to Sensory Input, Non-Typical Language, Self-Spinning or Spinning Objects | predicted: Absence or Avoidance of Eye Contact, Hyper- or Hyporeactivity to Sensory Input, Non-Responsiveness to Verbal Interaction, Non-Typical Language, Object Lining-Up, Self-Spinning or Spinning Objects

## Failed examples
- **BUAFdqrw4fA_20_44** — true: Self-Hitting or Self-Injurious Behavior | missed: Self-Hitting or Self-Injurious Behavior; false alarm: Hyper- or Hyporeactivity to Sensory Input, Non-Responsiveness to Verbal Interaction, Non-Typical Language, Object Lining-Up, Self-Spinning or Spinning Objects
- **5GJfaMIIUk8_240_264** — true: Background | missed: Background; false alarm: Hyper- or Hyporeactivity to Sensory Input, Non-Responsiveness to Verbal Interaction, Object Lining-Up, Self-Spinning or Spinning Objects, Upper Limb Stereotypies
- **02mvgUiO-eU_16_22** — true: Upper Limb Stereotypies | missed: Upper Limb Stereotypies; false alarm: Aggressive Behavior, Hyper- or Hyporeactivity to Sensory Input, Non-Responsiveness to Verbal Interaction, Self-Hitting or Self-Injurious Behavior
- **4p9XrwKJkxc_0_15** — true: Absence or Avoidance of Eye Contact, Upper Limb Stereotypies | missed: Absence or Avoidance of Eye Contact, Upper Limb Stereotypies; false alarm: Background, Hyper- or Hyporeactivity to Sensory Input, Object Lining-Up