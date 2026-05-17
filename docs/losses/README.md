# MiraFrag Losses

MiraFrag predicts a sparse probability distribution over generated fragment
candidates plus one out-of-support, or OOS, bucket. Candidate probabilities are
aggregated into m/z bins before comparing against the measured spectrum. The OOS
bucket is not emitted as a predicted peak; it is used during training for target
intensity that cannot be matched to generated fragments.

## Notation

For one spectrum, let:

- `S` be the set of m/z bins reachable by at least one generated fragment.
- `p_i` be the model probability assigned to reachable bin `i` after summing
  all fragment-candidate probabilities in that bin.
- `p_oos` be the model probability assigned to the OOS bucket.
- `y_i` be the measured target intensity in bin `i`.
- `T` be the set of target bins with nonzero intensity.

The usual binned cosine objective is:

```text
cos(p, y) = (sum_i p_i y_i) / (||p||_2 ||y||_2)
```

The model cannot assign probability to bins outside `S`, so target peaks in
`T \ S` cannot be learned from directly. They lower the maximum achievable
cosine, but they do not identify a reachable fragment bin that should receive
more probability.

## Cosine Optimum On Reachable Bins

Restrict the target to reachable bins:

```text
r_i = y_i        if i in S
      0          otherwise
```

For nonnegative predictions over reachable bins, cosine is maximized when the
prediction points in the same direction as the reachable target:

```text
p* = r / sum_j r_j
```

Equivalently, define the projected target distribution:

```text
q_i = y_i / sum_{j in S} y_j      for i in S and y_i > 0
```

Then the cosine optimum over the model's reachable support is `p = q`.

## Projected KL With OOS

`projected_kl` uses the projected target distribution for reachable bins and
assigns unreachable target intensity to OOS:

```text
q_i = y_i       for i in S and y_i > 0
q_oos = sum_{i not in S} y_i

projected_kl = KL(q_aug || p_aug)
```

The generated-fragment part has the same reachable-support minimizer as cosine,
but it gives the model cross-entropy-like gradients:

```text
CE(q, p) = -sum_i q_i log p_i
```

The OOS term gives explicit supervision for target intensity that the current
fragment generator cannot explain.

If no target bin is reachable for a spectrum, `projected_kl` now trains the OOS
bucket for that target intensity instead of contributing zero.

## Difference From `kl`

`kl` compares the target distribution against all measured target bins. With OOS
enabled, target bins that are unreachable by generated fragments are assigned to
the OOS bucket instead of an epsilon fallback probability.

`projected_kl` assigns unreachable target bins to OOS. It therefore trains
generated fragments on the part of the measured spectrum that the current
fragment generator can explain, while still giving explicit supervision for
unexplained target intensity.

## Soft Projected KL

`soft_projected_kl` avoids the hard bin boundary in `projected_kl`. For target
peak `i` at m/z `m_i` and candidate fragment `j` at m/z `u_j`, define a Gaussian
kernel:

```text
K_ij = exp(-0.5 ((m_i - u_j) / sigma_i)^2)
```

The kernel width is tied to the existing tolerance setting:

```text
sigma_i = tolerance_i / 2
```

For absolute tolerance, `tolerance_i` is `--mass-tolerance`. For relative
tolerance, it is:

```text
tolerance_i = --mass-tolerance * max(m_i, --mass-tolerance-min-mz)
```

Each target peak distributes its normalized intensity over candidate fragments:

```text
a_ij = K_ij / sum_l K_il
q_j = sum_i y_i a_ij
```

where `y_i` is the target intensity normalized over measured peaks. The loss is:

```text
soft_projected_kl = KL(q || p)
                  = sum_j q_j (log q_j - log p_j)
```

Here `p_j` is the model softmax probability over fragment candidates. Exact or
near m/z matches receive most target mass, but nearby candidates still receive
smooth partial supervision. This gives KL-style gradients without the
discontinuity of a hard tolerance window.

## Soft-Binned KL

`soft_binned_kl` distributes both target peaks and predicted fragment candidates
into neighboring m/z bins with the same Gaussian kernel:

```text
K_ik = exp(-0.5 ((m_i - c_k) / sigma_i)^2)
sigma_i = tolerance_i / 2
```

where `c_k` is the center of bin `k`. Kernel weights are normalized per peak, so
each target peak or fragment candidate preserves its total probability mass
while spreading it over nearby bins.

This gives a soft target distribution `y_soft` and a soft predicted distribution
`p_soft` over bins:

```text
soft_kl = KL(y_soft || p_soft)
        = sum_k y_soft_k (log y_soft_k - log p_soft_k)
```

This keeps the fast convergence behavior of KL while reducing hard bin-boundary
artifacts. Unlike `soft_binned_coverage_kl`, it does not construct the
target-peak by fragment-candidate coverage matrix, so it is much cheaper for
large fragment sets.

## Soft-Binned Coverage KL

`soft_binned_coverage_kl` adds an explicit peak-coverage term to
`soft_binned_kl`.

The coverage term measures how much model probability lies within tolerance of
each target peak:

```text
coverage_i = sum_j p_j 1[|m_i - u_j| <= tolerance_i]
coverage_loss = -sum_i y_i log coverage_i
```

The combined objective is:

```text
soft_binned_coverage_kl = soft_kl + lambda * coverage_loss
```

`lambda` is controlled by `--coverage-weight` or `COVERAGE_WEIGHT`. This loss
keeps the fast convergence behavior of KL, removes hard bin-boundary artifacts,
and directly rewards assigning probability to fragments near observed peaks.

## FraGNNet Cross Entropy

`fragnnet_ce` follows FraGNNet's sparse cross-entropy objective. The model
predicts a probability distribution over generated fragment candidates plus one
extra out-of-support, or OOS, outcome per spectrum.

For target peak `i` and predicted fragment `j`, the model likelihood is a
Gaussian mixture centered on predicted fragment m/z values:

```text
p(m_i) = sum_j p_j Normal(m_i | u_j, sigma_j^2)
```

For absolute tolerance:

```text
sigma_j = --mass-tolerance
```

For relative tolerance, matching FraGNNet's usual 10 ppm setup:

```text
sigma_j = --mass-tolerance * max(u_j, --mass-tolerance-min-mz)
```

Only target peaks with at least one fragment within the configured tolerance are
included in the in-support cross entropy:

```text
ios_ce = -sum_i y_i log p(m_i)
```

Target intensity assigned to peaks with no nearby generated fragment is trained
against the OOS probability:

```text
oos_ce = -y_oos log p_oos
fragnnet_ce = ios_ce + oos_ce
```

To reproduce FraGNNet's default tolerance setting, use relative 10 ppm matching:

```bash
make -C resources/massspecgym train LOSS=fragnnet_ce RELATIVE_MASS_TOLERANCE=1 MASS_TOLERANCE=1e-5
```

## Available Losses

Use:

```bash
make -C resources/massspecgym train LOSS=soft_projected_kl
```

Other losses remain available:

- `kl`: sparse binned KL with unreachable target bins assigned to OOS.
- `projected_kl`: hard projected KL on reachable target bins.
- `soft_binned_kl`: smooth-binned KL without the coverage penalty.
- `soft_binned_coverage_kl`: smooth-binned KL plus peak coverage penalty.
- `fragnnet_ce`: FraGNNet-style Gaussian sparse cross entropy with OOS mass.
- `cosine`: direct sparse binned cosine loss.
- `sqrt_cosine`: sparse binned cosine after square-root intensity transform.
- `tolerance_cosine`: cosine with m/z tolerance matching.
- `kl_cosine`: weighted combination of `kl` and `cosine`.
