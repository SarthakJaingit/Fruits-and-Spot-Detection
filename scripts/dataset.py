import fruit_dataframe
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor
import glob
import cv2
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np


class FruitDetectDataset(object):
    def __init__(self, id_labels, id_bounding_boxes, transforms, mode, noisy_dataset_path = None):
        assert len(id_labels) == len(id_bounding_boxes)
        assert sorted(id_labels.keys()) == sorted(id_bounding_boxes.keys())

        self.imgs_key = sorted(id_labels.keys())

        if noisy_dataset_path:
          self.noisy_fp = [fp for fp in glob.glob(os.path.join(noisy_dataset_path, "*.JPEG"))]

          print("Noisy Has been subsetted")
          #Go to this code if you want to subset.
          self.noisy_fp = self.noisy_fp[:60]

        else:
          print("Dataset getting configured without noise loader")
          self.noisy_fp = list()

    # np.random.shuffle(self.imgs_key)
        if (mode == "train"):
          self.imgs_key = self.imgs_key[:int(len(self.imgs_key) * 0.8)]
          if noisy_dataset_path:
            print("Extended {} noisy images to train set".format(int(len(self.noisy_fp) * 0.8)))
            self.imgs_key.extend(self.noisy_fp[:int(len(self.noisy_fp) * 0.8)])
        elif (mode == "test"):
          self.imgs_key = self.imgs_key[int(len(self.imgs_key) * 0.8):]
          if noisy_dataset_path:
            print("Extended {} noisy images to test set".format(int(len(self.noisy_fp) * 0.2)))
            self.imgs_key.extend(self.noisy_fp[int(len(self.noisy_fp) * 0.8):])
        else:
          raise ValueError("Invalid Mode choose from train or test")

        self.id_labels = id_labels
        self.id_bounding_boxes = id_bounding_boxes
        self.full_image_file_paths = glob.glob("../ScriptFruitDet/ScriptDataset/Train/*/*/*.jpeg")

        self.transforms = transforms

    @staticmethod
    def ffile_path(image_id, full_image_file_paths):
        for image_path in full_image_file_paths:
            if image_id in image_path:
                return image_path

    @staticmethod
    def find_area_bb(bb_coord):
        bb_coord = bb_coord.numpy()
        area_of_each_bb = list()
        for pair_of_coord in bb_coord:
            area_of_each_bb.append(
            (pair_of_coord[2] - pair_of_coord[0]) * (pair_of_coord[3] - pair_of_coord[1])
            )
        return torch.tensor(area_of_each_bb, dtype=torch.int32)

    @staticmethod
    def convert_min_max(bb_coord):
        for pair_of_coord in bb_coord:
            pair_of_coord[2], pair_of_coord[3] = (pair_of_coord[0] + pair_of_coord[-2]), (pair_of_coord[1] + pair_of_coord[-1])
        return bb_coord

    def __getitem__(self, idx):

        img_key = self.imgs_key[idx]
        if img_key in self.noisy_fp:
          img_path = img_key
          boxes = torch.zeros((0, 4), dtype=torch.float32)
          labels = torch.as_tensor([], dtype = torch.int64)
        else:
          img_path = self.ffile_path(self.imgs_key[idx], self.full_image_file_paths)
          boxes = self.convert_min_max(torch.as_tensor(self.id_bounding_boxes[self.imgs_key[idx]], dtype=torch.float32))
          labels = torch.as_tensor(self.id_labels[self.imgs_key[idx]], dtype=torch.int64)

        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        image_id = torch.tensor([idx])
        area = self.find_area_bb(boxes)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area

        #Query about transforms for labels of images
        if self.transforms:
          sample = {
                    'image': img,
                    'bboxes': target['boxes'],
                    'labels': labels
                }

          sample = self.transforms(**sample)
          img = sample['image']

          if img_key not in self.noisy_fp:
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)


        return img, target

    def __len__(self):
        return len(self.imgs_key)

class NoiseDataset(object):

    def __init__(self, noise_file_path, size, camera_size):

        self.size = size
        self.noise_file_path = [fp for fp in glob.glob(os.path.join(noise_file_path, "*.JPEG"))]
        self.transforms = transforms.Compose([
                                              transforms.Resize((camera_size, camera_size)),
                                              transforms.ToTensor()])
    def __getitem__(self, idx):

        current_file_path = self.noise_file_path[idx]
        img = Image.open(current_file_path).convert("RGB")

        img = self.transforms(img)
        return img

    def __len__(self):
        if self.size:
          return self.size
        return len(self.noise_file_path)


def get_transforms(mode):
    if (mode == "train"):
        return A.Compose([
                          A.OneOf([
                          A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                         val_shift_limit=0.2, p=0.9),
                          A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2, p=0.9)],p=0.9),
                          A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.5),
                          A.HorizontalFlip(),
                          A.VerticalFlip(),
                          # A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), p=1),
                          # ToTensor(),
                          ToTensorV2()
                          ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    elif (mode == "test"):
        return A.Compose([
                          # A.Resize(512, 512),
                          # A.Normalize(mean=(0.485, 0.456, 0.406),
                          # std=(0.229, 0.224, 0.225), p=1),
                          # ToTensor()
                          ToTensorV2()
                          ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    elif (mode == "effdet_train"):
        return A.Compose([
                          A.OneOf([
                          A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                         val_shift_limit=0.2, p=0.9),
                          A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2, p=0.9)],p=0.9),
                          A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.5),
                          A.HorizontalFlip(),
                          A.VerticalFlip(),
                          A.Resize(height = 512, width=512),
                          ToTensorV2()
                          ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    elif (mode == "effdet_test"):
        return A.Compose([
                          A.Resize(height = 512, width = 512),
                          ToTensorV2()])
    else:
        raise ValueError("mode is wrong value can either be train or test")


if __name__ == "__main__":

    bounding_box_dict, labels_dict = fruit_dataframe.get_dict(["Placeholder", "Apples", "Strawberry", "Apple_Bad_Spot", "Strawberry_Bad_Spot"])
    noise_file_path = "ScriptDataset/noisy_dataset"
    train_dataset = FruitDetectDataset(labels_dict, bounding_box_dict, get_transforms(mode = "train"), mode = "train", noisy_dataset_path= noise_file_path)
    test_dataset = FruitDetectDataset(labels_dict, bounding_box_dict, get_transforms(mode = "test"), mode = "test", noisy_dataset_path=noise_file_path)

    size, camera_size = 100, 512
    noise_dataset = NoiseDataset(noise_file_path, size, camera_size)

    print(len(train_dataset))
    print(len(test_dataset))
    print(len(noise_dataset))
