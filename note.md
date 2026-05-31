# Notes — BioDCASE 2026 CD-MSC

---

## 2026-05-31

- Repo cloned and set up on Mila cluster
- Data lives in `/network/scratch/s/sahas/CD-MSC/Development_data/` (symlinked via `data/` and `Development_data/`)
- Unzipping `Development_data.zip` (3.6 GB, ~271k clips) — in progress
- Baseline numbers (10-seed, best checkpoint): BA_seen=0.879, BA_unseen=0.185, DSG=0.694
- Key imbalance: D5 has 265k/271k total clips; species 6/8/9 have <700 clips each

