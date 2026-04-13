# ADR 0002 — Cox partial likelihood loss for the baseline

- **Date:** 2026-04-13
- **Status:** Accepted

## Context

Deep survival models commonly use one of three loss families:

1. **Cox partial likelihood** (Faraggi & Simon 1995; DeepSurv, Katzman 2018):
   the model outputs a scalar risk score `r_i = f(x_i)`, and training
   maximizes the partial likelihood of a Cox proportional-hazards model with
   `r_i` as the linear predictor.
2. **Discretized categorical survival** (PathoGems, DeepHit / Nnet-survival):
   time is binned into `K` intervals and the model outputs a discrete
   hazard or survival distribution; training uses cross-entropy or a
   likelihood over bins.
3. **Parametric / accelerated-failure-time losses** (Weibull, log-normal).

## Decision

For the baseline, use the **negative Cox partial log-likelihood** (Breslow
ties handling). We will implement it ourselves in ~20 lines of PyTorch so
the loss is fully visible and testable.

## Rationale

- **Hyperparameter-free.** The categorical loss requires choosing a bin
  count `K` and a discretization strategy (equal-time vs equal-event). Every
  downstream experiment would have to pin those choices. Cox PH has none of
  these knobs.
- **C-index alignment.** Harrell's concordance index is the standard survival
  metric and it is directly a measure of how well a scalar risk score orders
  patients — which is exactly what a Cox model outputs. The loss and the
  metric are defined on the same quantity, so training signal and evaluation
  signal cannot diverge the way they can with a discretized loss evaluated
  by C-index.
- **Minimal output head.** One scalar per patient. The categorical loss needs
  `K` outputs and a softmax, which is more code to get right and more places
  for off-by-one errors around right-censoring.
- **Literature baseline.** Cox partial likelihood with a deep feature
  extractor (DeepSurv) is the canonical starting point cited by nearly every
  multimodal survival paper, including PathoGems. Starting elsewhere would
  make comparison harder.

## Consequences

### Positive

- Simplest possible training code, and the loss can be unit-tested against
  a closed-form reference (partial likelihood is analytic for small risk
  sets).
- When we later swap in the categorical loss as a controlled experiment
  (per the brief's "change one thing at a time" plan), the delta is directly
  attributable to the loss change because nothing else moved.

### Negative

- Cox PH assumes proportional hazards. For real BRCA data the assumption is
  only approximately true — this is a known limitation shared by the entire
  Cox-based survival literature and we accept it for the baseline. It does
  not invalidate C-index comparisons, only calibration claims (which we do
  not make yet).
- Numerical stability requires the standard log-sum-exp trick (subtracting
  the max before exponentiating the risk set). Our implementation includes
  this and a pytest for numerical stability at large `|r|`.

### Follow-ups

- ADR to be written when the categorical-survival variant is introduced,
  documenting the exact bin strategy and tying back to this ADR.
