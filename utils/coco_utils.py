import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Subset, DataLoader
from detr.datasets import transforms as coco_transforms
from detr.datasets.coco import CocoDetection as DetrCocoDetection
from detr.util.misc import collate_fn, NestedTensor
from torchvision import transforms
from pathlib import Path


default_transforms = coco_transforms.Compose([coco_transforms.Resize(size=(640, 480)), coco_transforms.ToTensor(),
                                              coco_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])
COCO_DIR = Path('placeholder')  # set coco directory here

coco_annotations_file = COCO_DIR / 'annotations' / 'instances_val2017.json'
coco = COCO(coco_annotations_file)
coco_cats = coco.loadCats(coco.getCatIds())
coco_class_to_name = {cat['id']: cat['name'] for cat in coco_cats}
categories = list(coco_class_to_name.keys())

default_hue_filters = ['none', 'red', 'green', 'blue', '120', '240']


def get_dataset(transform, return_mask, dataset_type):
    if dataset_type == 'val':
        dataset = DetrCocoDetection(img_folder=COCO_DIR / 'val2017',
                                    ann_file=COCO_DIR / 'annotations' / 'instances_val2017.json',
                                    transforms=transform, return_masks=return_mask)
    elif dataset_type == 'test':
        dataset = DetrCocoDetection(img_folder=COCO_DIR / 'test2017',
                                    ann_file=COCO_DIR / 'annotations' / 'instances_val2017.json',
                                    transforms=transform, return_masks=return_mask)
    elif dataset_type == 'train':
        dataset = DetrCocoDetection(img_folder=COCO_DIR / 'train2017',
                                    ann_file=COCO_DIR / 'annotations' / 'instances_val2017.json',
                                    transforms=transform, return_masks=return_mask)
    else:
        raise NameError(f'Unknown dataset type: {dataset_type}')
    return dataset


def get_dataloader_for_dataset(dataset, batch_size, dataset_type):
    if dataset_type == 'train':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                sampler=torch.utils.data.RandomSampler(dataset), collate_fn=collate_fn)
    elif dataset_type == 'val':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                sampler=torch.utils.data.SequentialSampler(dataset), collate_fn=collate_fn)
    elif dataset_type == 'test':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                sampler=torch.utils.data.SequentialSampler(dataset), collate_fn=collate_fn)
    else:
        raise NameError(f'Unknown dataset type: {dataset_type}')
    return dataloader


def get_dataloader(transform=default_transforms, category_ids=(), img_ids=(), batch_size=1, return_mask=False,
                   dataset_type='val'):
    dataset = get_dataset(transform=transform, return_mask=return_mask, dataset_type=dataset_type)
    if len(category_ids) > 0 or len(img_ids) > 0:
        annotation_ids = coco.getAnnIds(catIds=category_ids, imgIds=img_ids)
        annotations = coco.loadAnns(annotation_ids)
        image_ids = list(set([ann['image_id'] for ann in annotations]))
        id_to_index = {img_id: idx for idx, img_id in enumerate(dataset.ids)}
        subset_indices = [id_to_index[img_id] for img_id in image_ids if img_id in id_to_index]
        dataset = Subset(dataset, subset_indices)
    dataloader = get_dataloader_for_dataset(dataset=dataset, dataset_type=dataset_type, batch_size=batch_size)
    return dataloader


def get_colored_dataloader(category, image_ids=(), batch_size=1, color='none'):
    if color == 'none':
        transform = coco_transforms.Compose([coco_transforms.Resize(size=(640, 480)),
                                             coco_transforms.ToTensor(),
                                             coco_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif color == 'red':
        transform = coco_transforms.Compose([coco_transforms.Resize(size=(640, 480)),
                                             coco_transforms.ToTensor(),
                                             RecolorObjectHUE(color_filter=0, category=category),
                                             coco_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif color == 'green':
        transform = coco_transforms.Compose([coco_transforms.Resize(size=(640, 480)),
                                             coco_transforms.ToTensor(),
                                             RecolorObjectHUE(color_filter=60, category=category),
                                             coco_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif color == 'blue':
        transform = coco_transforms.Compose([coco_transforms.Resize(size=(640, 480)),
                                             coco_transforms.ToTensor(),
                                             RecolorObjectHUE(color_filter=120, category=category),
                                             coco_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif color == '120':
        transform = coco_transforms.Compose([coco_transforms.Resize(size=(640, 480)),
                                             coco_transforms.ToTensor(),
                                             RecolorObjectHUE(color_filter=60, category=category, add_filter=True),
                                             coco_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif color == '240':
        transform = coco_transforms.Compose([coco_transforms.Resize(size=(640, 480)),
                                             coco_transforms.ToTensor(),
                                             RecolorObjectHUE(color_filter=120, category=category, add_filter=True),
                                             coco_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif color == 'gray':
        transform = coco_transforms.Compose([coco_transforms.Resize(size=(640, 480)),
                                             coco_transforms.ToTensor(),
                                             TorchTransformWrapper(transforms.Grayscale(num_output_channels=3)),
                                             coco_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        raise NameError('unknown')
    return get_dataloader(transform=transform, batch_size=batch_size, return_mask=True, img_ids=image_ids,
                          category_ids=(category,))


class RecolorObjectHUE:
    def __init__(self, color_filter, category=None, whiten_surroundings=False, single_object=False, add_filter=False):
        self.color_filter = color_filter
        self.category = category
        self.whiten_surroundings = whiten_surroundings
        self.single_object = single_object
        self.add_filter = add_filter

    def __call__(self, image, target):
        labels, masks = target['labels'], target['masks']
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        for i in range(labels.shape[0]):
            if labels[i] == self.category:
                if self.color_filter is not None:
                    if self.add_filter:
                        img_hsv = img_hsv.astype(np.int16)
                        img_hsv[masks[i], 0] += self.color_filter
                        img_hsv[masks[i], 0] = img_hsv[masks[i], 0] % 180
                        img_hsv = img_hsv.astype(np.uint8)
                    else:
                        img_hsv[masks[i], 0] = self.color_filter
                if self.whiten_surroundings:
                    img_hsv[~masks[i], 1] = 0
                    img_hsv[~masks[i], 2] = 255
                if self.single_object:
                    break

        img_rgb_np = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        img_rgb_tensor = torch.from_numpy(img_rgb_np).float() / 255.0
        img_rgb_tensor = img_rgb_tensor.permute(2, 0, 1).clamp(0, 1)  # Convert to [3, H, W] format
        return img_rgb_tensor, target


def get_img_ids_from_dataloader(dataloader):
    if hasattr(dataloader.dataset, 'dataset'):
        return [dataloader.dataset.dataset.ids[i] for i in dataloader.dataset.indices]
    else:
        return dataloader.dataset.ids


class TorchTransformWrapper(object):
    def __init__(self, torch_transform):
        self.torch_transform = torch_transform

    def __call__(self, img, target):
        return self.torch_transform(img), target


def get_batch(batch_id, dataloader):
    for dataloader_batch_id, batch in enumerate(dataloader):
        if batch_id == dataloader_batch_id:
            return batch


def filter_large_objects_by_mask(inputs, targets, category, threshold=92160):  # 30 % of img
    filter_ids = []
    for i in range(len(targets)):
        sum_cat = 0
        labels = targets[i]['labels']
        masks = targets[i]['masks']
        for j in range(labels.shape[0]):
            if labels[j].item() == category:
                sum_cat += torch.sum(masks[j])
        if sum_cat >= threshold:
            filter_ids.append(i)
    if len(filter_ids) == 0:
        return None
    indices = torch.tensor(filter_ids)
    print(indices)
    return NestedTensor(inputs.tensors[indices], inputs.mask[indices])
