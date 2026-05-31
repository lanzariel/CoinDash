# CoinDash — Model Documentation

## What the app computes

The dashboard answers one question: **given a betting opportunity with known edge, what fraction of your wealth should you bet?**

It does this by computing the full probability distribution of final wealth over a structured sequence of bets, then evaluating two utility functions over that distribution. The computation is **exact** (not Monte Carlo): it enumerates all possible outcomes and their probabilities using the binomial PMF, accumulating a wealth→probability dictionary via dynamic programming.

---

## Bet structure

Bets are arranged in two dimensions:

### Contemporaneous bets (`ncs`)
At each point in time you place `ncs` bets simultaneously. All `ncs` bets resolve at the same instant. The number of wins `X` is drawn from:

```
X ~ Binomial(n=ncs, p=p)
```

Your wealth then evolves as:

```
W_next = W * (1 + s * (2X - ncs))
```

where `s` is the bet size (fraction of wealth wagered per bet). Each win contributes `+s·W` and each loss contributes `−s·W`, so the net change is `(wins − losses)·s·W = (2X − ncs)·s·W`.

### Sequential rounds (`nsl`)
After the contemporaneous bets resolve, you repeat the whole process for `nsl` independent rounds. The outcome of round `t+1` does not depend on which specific bets were won or lost in round `t` — only on the resulting wealth level carried forward.

Final wealth is the product of `nsl` multiplicative factors, one per round:

```
W_final = W_0 · ∏_{t=1}^{nsl} (1 + s·(2X_t − ncs)),   X_t ~ Binomial(ncs, p) i.i.d.
```

---

## Correlation assumptions

### Between contemporaneous bets: **zero**
The binomial distribution models `ncs` coin tosses as **independent and identically distributed**. There is no correlation between simultaneous bets. This means the variance of the number of wins is:

```
Var(X) = ncs · p · (1−p)
```

If instead each pair of bets had correlation ρ, the correct variance would be:

```
Var(X) = ncs · p · (1−p) · (1 + (ncs−1)·ρ)
```

Positive correlation (ρ > 0) inflates variance relative to the binomial, which matters for Kelly sizing: correlated bets are riskier than independent ones, and the optimal fraction is lower. The current model does not capture this — it implicitly prices all contemporaneous bets as uncorrelated.

### Between sequential rounds: **zero**
Each round's binomial draw is independent of all previous rounds. There is no autocorrelation or path-dependency across time steps (beyond the wealth level itself).

---

## Absorbing state (ruin barrier)

If the absorbing state is enabled, a wealth level `W` is frozen (no further betting) when:

```
W · (1 − s·ncs) < absorbing_threshold
```

The condition checks whether a total loss on all `ncs` contemporaneous bets in the next round would push wealth below the threshold. Once absorbed, the wealth level stays fixed for all remaining sequential rounds.

---

## Computation method

The distribution is built iteratively using **exact dynamic programming**, not Monte Carlo:

1. Start with the distribution `{1.0: 1.0}` (wealth = 1 with probability 1).
2. For each sequential round, for each current wealth level and each possible number of wins `x ∈ {0, …, ncs}`:
   - Look up the exact probability `Binomial(ncs, p).pmf(x)`.
   - Compute the new wealth level.
   - Add the joint probability to the next-round distribution.
3. Wealth levels that coincide across different paths are merged (their probabilities summed). This keeps the state space tractable for small-to-moderate `ncs` and `nsl`.

The result is an exact discrete probability distribution over all reachable final wealth levels.

A 10 ms wall-clock timeout is applied; if the parameter space is too large the computation returns an empty distribution rather than blocking the UI.

---

## Utility comparison

Two utility functions are evaluated over the final wealth distribution `P(W)`:

| Name | Function | Notes |
|---|---|---|
| **Kelly / log** | `U(W) = log(W)` | Maximising E[log W] yields the Kelly fraction. Undefined for W ≤ 0 (returns −∞). |
| **Exponential** | `U(W) = 1 − exp(−(W−1)/0.1)` | Risk-averse with constant absolute risk aversion (CARA). The denominator 0.1 sets the risk-tolerance scale. |

Expected utility is:

```
E[U] = Σ_W  P(W) · U(W)
```

The utility chart sweeps bet size `s` across 60 evenly-spaced values in `[x_min, x_max]` and plots `E[U(s)]` for both functions. The peak of the Kelly/log curve is the Kelly-optimal fraction for the given bet structure.

---

## Key limitations

- **Independence assumed throughout.** Real markets exhibit both cross-sectional correlation (contemporaneous bets) and serial correlation (across rounds). Ignoring positive correlation understates risk and overstates the optimal bet size.
- **Fixed p.** The win probability is assumed known and constant. There is no parameter uncertainty or Bayesian updating.
- **Symmetric bet sizes.** Each of the `ncs` bets uses the same fraction `s`. Non-uniform sizing across bets is not modelled.
- **Discrete approximation of a continuous problem.** Because `ncs` is finite, the binomial distribution is a discrete approximation of any underlying continuous return distribution.
