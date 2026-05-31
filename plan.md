# Research Plan — BioDCASE 2026 CD-MSC

**Challenge:** Cross-Domain Mosquito Species Classification  
**Primary metric:** `species_balanced_accuracy` on unseen domains  
**DSG** (`|BA_unseen - BA_seen|`) — lower is better  
**Baseline (10-seed best-ckpt):** BA_seen=0.879, BA_unseen=0.185, DSG=0.694

---

## Goal

Improve cross-domain generalization (reduce DSG) while maintaining or improving BA_unseen.

---

## Experiments

| ID | Description | Status | BA_seen | BA_unseen | DSG | Notes |
|----|-------------|--------|---------|-----------|-----|-------|
| B0 | Baseline MTRCNN (10-seed) | done | 0.879 | 0.185 | 0.694 | released checkpoint |

---

## Next Steps

- [ ] Reproduce baseline locally
- [ ] Feature extraction on Development_data

---

## Ideas / Directions

- Domain-invariant training (adversarial domain head, gradient reversal)
- Data augmentation for domain shift (SpecAugment, pitch shift, noise)
- Class-balanced sampling to address species imbalance
- Domain-balanced sampling (D5 dominates with 265k/271k clips)
- **LODO cross-validation** (`train_lodo.py` implemented) — Leave-One-Domain-Out CV, `--fold D1..D5`, trains on 4 domains, validates on held-out domain; gives true out-of-domain BA per fold. Not yet run.

---

## Milestones

| Date | Milestone |
|------|-----------|
| — | Baseline reproduced locally |
| — | First improved run > BA_unseen 0.185 |
| — | Challenge submission |
