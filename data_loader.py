from pathlib import Path
import csv
import numpy as np

CHANNELS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
            "P8", "T8", "FC6", "F4", "F8", "AF4")


def _read_csv(path):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))[1:]
    return np.asarray([[float(value) for value in row[1:]] for row in rows],
                      dtype=np.float64).T


CLASS_NAMES = ("Sí", "No", "Baño", "Hambre", "Sed", "Ayuda", "Dolor", "Gracias")


def load_eight_class(data_dir="data", return_groups=False):
    """Load the eight imagined-speech classes, excluding rest (class_0)."""
    root = Path(data_dir) / "raw"
    # Dataset folder IDs from data/readme.txt.
    class_ids = (8, 6, 2, 5, 7, 1, 3, 4)
    pairs = []
    for label, class_id in enumerate(class_ids):
        pairs.extend((path, label) for path in
                     sorted((root / f"class_{class_id}").glob("*.csv"),
                            key=lambda p: int(p.stem)))
    signals = np.stack([_read_csv(path) for path, _ in pairs])
    labels = np.asarray([label for _, label in pairs], dtype=np.int64)
    if signals.shape[1:] != (14, 128):
        raise ValueError(f"expected (N, 14, 128), got {signals.shape}")
    if return_groups:
        # The source release contains no subject/trial identifiers. Twelve
        # consecutive chunks are one 8-second trial (stride 80 at 128 Hz).
        groups = np.asarray([class_id * 1000 + int(path.stem) // 12
                             for (path, _), class_id in zip(pairs,
                                                             np.repeat(class_ids,
                                                                       [sum(1 for _ in (root / f"class_{c}").glob("*.csv"))
                                                                        for c in class_ids]))])
        return signals, labels, groups
    return signals, labels


load_binary = load_eight_class
