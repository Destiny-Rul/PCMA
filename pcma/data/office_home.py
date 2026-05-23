"""Office-Home dataset wrapper.

Office-Home contains 65 categories collected across four domains:
``Art`` (``art``), ``Clipart`` (``clipart``), ``Product`` (``product``),
and ``Real-World`` (``real_world``). The class list below preserves the
canonical order shipped with the original benchmark; do not re-sort it,
as label indices are tied to this ordering throughout the project.
"""

from __future__ import annotations

import os
from os import path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


OFFICE_HOME_DOMAINS = ("art", "clipart", "product", "real_world")

OFFICE_HOME_CLASSES = (
    "Drill", "Exit_Sign", "Bottle", "Glasses", "Computer", "File_Cabinet",
    "Shelf", "Toys", "Sink", "Laptop", "Kettle", "Folder", "Keyboard",
    "Flipflops", "Pencil", "Bed", "Hammer", "ToothBrush", "Couch", "Bike",
    "Postit_Notes", "Mug", "Webcam", "Desk_Lamp", "Telephone", "Helmet",
    "Mouse", "Pen", "Monitor", "Mop", "Sneakers", "Notebook", "Backpack",
    "Alarm_Clock", "Push_Pin", "Paper_Clip", "Batteries", "Radio", "Fan",
    "Ruler", "Pan", "Screwdriver", "Trash_Can", "Printer", "Speaker",
    "Eraser", "Bucket", "Chair", "Calendar", "Calculator", "Flowers",
    "Lamp_Shade", "Spoon", "Candles", "Clipboards", "Scissors", "TV",
    "Curtains", "Fork", "Soda", "Table", "Knives", "Oven", "Refrigerator",
    "Marker",
)


def _readable_class_name(raw: str) -> str:
    """Convert ``"File_Cabinet"`` to ``"file cabinet"`` for prompt rendering."""
    return raw.replace("_", " ").lower()


class OfficeHome(Dataset):
    """Office-Home directory-organised dataset.

    Args:
        root_dir: Path to the dataset root that contains the four domain
            subdirectories (``art``, ``clipart``, ``product``,
            ``real_world``).
        domain: One of :data:`OFFICE_HOME_DOMAINS`.
        transform: Optional torchvision transform; defaults to
            :class:`torchvision.transforms.ToTensor`.
    """

    classes = OFFICE_HOME_CLASSES

    def __init__(
        self,
        root_dir: str,
        domain: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if domain not in OFFICE_HOME_DOMAINS:
            raise ValueError(
                f"Unknown domain '{domain}'. Expected one of {OFFICE_HOME_DOMAINS}."
            )
        self.domain_dir = path.join(root_dir, domain)
        self.transform = transform or transforms.ToTensor()

        self.image_paths = []
        self.labels = []
        for label, category in enumerate(OFFICE_HOME_CLASSES):
            class_dir = path.join(self.domain_dir, category)
            if not path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                self.image_paths.append(path.join(class_dir, fname))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        return self.transform(img), idx, label


def build_office_home(domain: str, root_dir: str, transform: Optional[Callable] = None) -> OfficeHome:
    """Convenience factory that mirrors the project naming convention."""
    return OfficeHome(root_dir=root_dir, domain=domain, transform=transform)


def get_class_prompts() -> list:
    """Class names rendered as natural-language strings for CLIP prompts."""
    return [_readable_class_name(c) for c in OFFICE_HOME_CLASSES]
