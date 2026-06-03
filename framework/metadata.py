"""Species and domain metadata for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import re
from pathlib import Path
from typing import List, Tuple, Union

import torch


SPECIES_ID_TO_NAME = {
    "1": "Aedes aegypti",
    "2": "Aedes albopictus",
    "3": "Culex quinquefasciatus",
    "4": "Anopheles gambiae",
    "5": "Anopheles arabiensis",
    "6": "Anopheles dirus",
    "7": "Culex pipiens",
    "8": "Anopheles minimus",
    "9": "Anopheles stephensi",
}
SPECIES_NAMES = list(SPECIES_ID_TO_NAME.values())
SPECIES_TO_INDEX = {name: idx for idx, name in enumerate(SPECIES_NAMES)}
DOMAIN_ID_TO_NAME = {"1": "D1", "2": "D2", "3": "D3", "4": "D4", "5": "D5"}

# Female wingbeat fundamental frequencies (Hz) from literature, ordered by SPECIES_NAMES index.
# Sources: Mukundarajan et al. 2017 (Sci. Transl. Med.), Brogdon 1994, Kiskin et al. 2020 (HumBugDB).
# Normalized to [0, 1] over the 400–700 Hz reference range used in _wingbeat_mel_bins.
# Index:           0     1     2     3     4     5     6     7     8
# Species:       Aaeg  Aalb  Cqui  Agam  Aara  Adiu  Cpip  Amin  Astr
_WINGBEAT_HZ = [500.0, 480.0, 450.0, 580.0, 560.0, 620.0, 400.0, 650.0, 460.0]
_WB_LO, _WB_HI = 400.0, 700.0
SPECIES_WINGBEAT_TARGETS: torch.Tensor = torch.tensor(
    [(_hz - _WB_LO) / (_WB_HI - _WB_LO) for _hz in _WINGBEAT_HZ],
    dtype=torch.float32,
)
DOMAIN_NAMES = list(DOMAIN_ID_TO_NAME.values())
DOMAIN_TO_INDEX = {name: idx for idx, name in enumerate(DOMAIN_NAMES)}
FILE_PATTERN = re.compile(r"^S_(\d+)_D_(\d+)_(\d+)$")


def load_id_list(path: Union[str, Path]) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_file_id(file_id: str) -> Tuple[str, str]:
    match = FILE_PATTERN.match(file_id)
    if not match:
        raise ValueError(f"Unrecognized file id: {file_id}")
    species_id, domain_id, _ = match.groups()
    if species_id not in SPECIES_ID_TO_NAME:
        raise ValueError(f"Unknown species id in file id: {file_id}")
    if domain_id not in DOMAIN_ID_TO_NAME:
        raise ValueError(f"Unknown domain id in file id: {file_id}")
    return SPECIES_ID_TO_NAME[species_id], DOMAIN_ID_TO_NAME[domain_id]
