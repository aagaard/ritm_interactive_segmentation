import cv2
import numpy as np
from pathlib import Path

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class ImagesDirDataset(ISDataset):
    def __init__(self, dataset_path, images_dir_name="images", masks_dir_name="masks", **kwargs):
        super(ImagesDirDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)

        # Root path to images
        self._images_path = self.dataset_path / images_dir_name

        # Root path to instance mask (many instance mask files to one image file)
        self._insts_path = self.dataset_path / masks_dir_name

        image_filepaths = [x for x in sorted(self._images_path.glob("*.*"))]

        samples = {image_filepath.stem: {"image": image_filepath, "masks": []} for image_filepath in image_filepaths}

        for mask_filepath in self._insts_path.glob("*.*"):
            mask_filename = mask_filepath.stem
            if mask_filename in samples:
                samples[mask_filename]["masks"].append(mask_filepath)
                continue

            # template: ['000' "image_filename.png"]
            mask_filename_split = mask_filename.split("_")
            if mask_filename_split[-1].isdigit():
                #
                mask_filename = "_".join(mask_filename_split[:-1])
                assert mask_filename in samples
                samples[mask_filename]["masks"].append(mask_filepath)

        for image_masks_dict in samples.values():
            try:
                assert len(image_masks_dict["masks"]) > 0, image_masks_dict["image"]
            except AssertionError:
                print(f"Image {x['image']} does not have any masks")

        self.dataset_samples = [v for k, v in sorted(samples.items())]

    def get_sample(self, index) -> DSample:
        sample = self.dataset_samples[index]
        image_path = str(sample["image"])

        objects = []
        ignored_regions = []
        masks = []
        for indx, mask_path in enumerate(sample["masks"]):
            gt_mask = cv2.imread(str(mask_path))[:, :, 0].astype(np.int32)
            instances_mask = np.zeros_like(gt_mask)
            instances_mask[gt_mask == 128] = 2
            instances_mask[gt_mask > 128] = 1
            masks.append(instances_mask)
            objects.append((indx, 1))
            ignored_regions.append((indx, 2))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return DSample(image, np.stack(masks, axis=2), objects_ids=objects, ignore_ids=ignored_regions, sample_id=index)
