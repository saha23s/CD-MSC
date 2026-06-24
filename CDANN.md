# Conditional DANN (C-DANN) — Technical Notes

## The Problem with Regular DANN on This Dataset

Take **An. dirus** as the clearest example. It has 76 training samples, almost all
in D5 (lab), and its unseen test domain is D4 (field). It's the most extreme case
of species-domain correlation in this dataset:

- Regular DANN collapses species-discriminative features for rare species
- C-DANN fixes this by conditioning the domain discriminator on the species label

---

## Regular DANN: What Goes Wrong

The domain discriminator sees only `z` — the 32-dim embedding — and tries to predict
domain. Over training it learns a pattern like:

> *"Whenever z has these particular values in dimensions 4, 17, 23... the domain is D4."*

But those dimensions encode **what An. dirus sounds like** — its wingbeat frequency,
harmonic structure, etc. The discriminator has no choice but to use them, because D4
samples are almost exclusively An. dirus.

The GRL then reverses that gradient and tells the backbone:

> *"Suppress whatever is in dimensions 4, 17, 23 — the discriminator is using them
> to identify D4."*

The backbone obeys — and in doing so destroys the features that identify An. dirus.
The model loses the ability to recognise An. dirus entirely, precisely to make its
domain unpredictable.

This is the fundamental flaw of regular DANN when **species and domain are correlated**:
removing domain information also removes species-discriminative information for
rare species whose training data is concentrated in one domain.

---

## C-DANN: The Fix

The domain discriminator now sees `[z ; onehot(species)]` — a 41-dim vector (32 + 9)
that includes the species label for free. For an An. dirus sample, the one-hot is
`[0,0,0,0,0,1,0,0,0]`.

The discriminator knows it's An. dirus before it even looks at z. So instead of
using species-correlated features to predict domain, it learns:

> *"I already know it's An. dirus. Given that, what in z tells me whether this is a
> D4 field recording vs D5 lab recording? Probably background noise texture,
> microphone frequency response, room acoustics..."*

The GRL then tells the backbone:

> *"Suppress the recording-environment artefacts — the discriminator is using those
> to distinguish D4 from D5 for An. dirus specifically."*

Species-discriminative features survive untouched because the discriminator already
has them from the one-hot label — it doesn't need z to encode them.

---

## What Changes Architecturally

```
Regular DANN:
  z [B,32] → GRL → [B,32] ────────────────────→ W_d [5×32] → domain_logits [B,5]

C-DANN:
  z [B,32] → GRL → [B,32] ──┐
                               cat → [B,41] → W_d [5×41] → domain_logits [B,5]
  species [B] → onehot → [B,9] ──┘
```

The species one-hot is appended **after** the GRL — it's a fixed discrete label,
not a learned feature. The gradient reversal still acts only on z. The discriminator
is stronger (has more information per sample), which means the adversarial pressure
on z is better targeted: it only has to remove what the discriminator couldn't already
explain from the species label alone.

### Mathematical contrast

Regular DANN targets the **marginal** domain distribution:
```
p(domain | z) ≈ uniform
```

C-DANN targets the **conditional** domain distribution:
```
p(domain | z, species) ≈ uniform    for all species
```

These are different objectives. For common species uniformly spread across domains
they behave similarly. For rare species concentrated in one domain (An. dirus → D4,
An. minimus → D2, An. stephensi → D4), they diverge significantly — regular DANN
destroys their features, C-DANN preserves them.

---

## At Inference Time

The domain head is **never used for predictions** — only `species_logits` matter
when classifying a new clip. C-DANN conditioning is therefore only active during
training. At inference, `species_labels=None` is passed and the model falls back
gracefully.

---

## Implementation in This Repo

Config flag: `"cdann": true` in the experiment JSON.

- `framework/model.py` — `MTRCNNClassifier.__init__()` reads `config["cdann"]`;
  sets `domain_classifier` input to 41 when True, 32 when False.
  `forward()` accepts `species_labels` and concatenates one-hot when cdann is on.
- `framework/engine.py` — `train_one_epoch()` and `evaluate_model()` both pass
  `species_labels` to the model.
- `train.py` — experiment name gets `_cdann` suffix to separate output directories.
- `colab_dann.ipynb` — `CDANN = True/False` parameter in the parameters cell.

Output directory for seed 42, alpha=0.3, cdann=True:
```
outputs/MTRCNN_seed42_B64_E100_earlystop_min10_pati5_dann0.3_cdann/
```

---

## Why alpha_max=0.3 for C-DANN

Same reasoning as regular DANN. The D5 dominance (99.4% of batches) makes domain
gradients disproportionately large. With alpha_max=0.3 the maximum adversarial
pressure is 30% of the domain gradient at any point in training. The Ganin schedule
ramps gradually so species features are established before adversarial pressure builds.

At epoch 13 (13% of training): λ ≈ 0.17 — gentle enough for stable training.

---

## What to Expect vs Regular DANN

The core hypothesis is that C-DANN improves specifically on the **rare species**
(An. dirus, An. minimus, An. stephensi) whose unseen domain is their only test
signal. For common species spread across domains, results should be similar.

If `BAunseen` improves further vs regular DANN alpha=0.3 (0.2225), C-DANN's
per-species conditioning is paying off. If results are similar, the bottleneck
may be batch imbalance (D5 dominance in batches) rather than the conditioning —
in which case batch balancing is the next intervention.
