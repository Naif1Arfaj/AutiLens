# Qualitative examples — vision_r2p1d_cv

0/92 test clips have every one of the 10 labels exactly right (rare by construction — 10 independent binary decisions per clip). Mean per-clip F1 = 0.41. Below: the clips the model got most right (by per-clip F1) and most wrong.

## Correct / strong examples
- **R_gZqQy_Ae4_0_56** (clip F1 0.80) — true: Non-Typical Language, Upper Limb Stereotypies | predicted: Hyper- or Hyporeactivity to Sensory Input, Non-Typical Language, Upper Limb Stereotypies
- **FrsDDZycjfY_129_141** (clip F1 0.80) — true: Absence or Avoidance of Eye Contact, Non-Typical Language, Upper Limb Stereotypies | predicted: Non-Typical Language, Upper Limb Stereotypies
- **2TpC6CB0iXw_549_668** (clip F1 0.80) — true: Absence or Avoidance of Eye Contact, Aggressive Behavior, Hyper- or Hyporeactivity to Sensory Input, Non-Responsiveness to Verbal Interaction, Self-Hitting or Self-Injurious Behavior | predicted: Absence or Avoidance of Eye Contact, Aggressive Behavior, Hyper- or Hyporeactivity to Sensory Input, Non-Responsiveness to Verbal Interaction, Non-Typical Language
- **E-XgK_LaFKI_28_38** (clip F1 0.80) — true: Non-Typical Language, Upper Limb Stereotypies | predicted: Hyper- or Hyporeactivity to Sensory Input, Non-Typical Language, Upper Limb Stereotypies

## Failed examples
- **JCHtVLqJWpg_73_80** — true: Aggressive Behavior, Self-Hitting or Self-Injurious Behavior | missed: Aggressive Behavior, Self-Hitting or Self-Injurious Behavior; false alarm: Absence or Avoidance of Eye Contact, Hyper- or Hyporeactivity to Sensory Input, Non-Responsiveness to Verbal Interaction, Non-Typical Language, Upper Limb Stereotypies
- **-9aliPAniig_66_71** — true: Non-Responsiveness to Verbal Interaction, Self-Spinning or Spinning Objects | missed: Non-Responsiveness to Verbal Interaction, Self-Spinning or Spinning Objects; false alarm: Absence or Avoidance of Eye Contact, Hyper- or Hyporeactivity to Sensory Input, Non-Typical Language, Upper Limb Stereotypies
- **02mvgUiO-eU_16_22** — true: Upper Limb Stereotypies | missed: Upper Limb Stereotypies; false alarm: Absence or Avoidance of Eye Contact, Aggressive Behavior, Hyper- or Hyporeactivity to Sensory Input, Non-Typical Language, Self-Hitting or Self-Injurious Behavior
- **JCHtVLqJWpg_54_62** — true: Aggressive Behavior, Self-Hitting or Self-Injurious Behavior | missed: Aggressive Behavior, Self-Hitting or Self-Injurious Behavior; false alarm: Hyper- or Hyporeactivity to Sensory Input, Non-Typical Language, Self-Spinning or Spinning Objects, Upper Limb Stereotypies