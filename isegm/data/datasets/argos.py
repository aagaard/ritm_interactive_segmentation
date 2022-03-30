import cv2
import numpy as np
from pathlib import Path

from isegm.data.base import ISDataset
from isegm.data.sample import DSample



class ArgosDataset(ISDataset):
    def __init__(
        self,
        dataset_root_path: str,
        images_dir_name='images', masks_dir_name='semantic_segmentation_mask',
        cvat_annotation_file_glob: str = "cvat_argos_segmentation_annotation/**/annotations.xml",
        images_glob: str = "images/*/color/*_color.png",
        **kwargs
    ):
    super(ArgosDataset).__init__(**kwargs)
    self.dataset_path = Path(dataset_path)
    self._images_path = self.dataset_path / images_dir_name
    self._insts_path = self.dataset_path / masks_dir_name
