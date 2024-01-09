import os
import xml.etree.ElementTree as ET
import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection

# Define the path to the dataset
data_dir = os.path.dirname(os.path.abspath(__file__))
print(data_dir)

# Define the transforms to be applied to the images
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the dataset
train_dataset = VOCDetection(root=data_dir, year='2012', image_set='train', download=False, transform=transform)

# function to extract the bounding boxes and labels from the annotations
def extract_boxes_labels(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    # labels is a string list, it needs to be mapped first in a dict
    #return torch.tensor(boxes), torch.tensor(labels)
    return torch.tensor(boxes), labels

# function to get the image and its annotations
def get_image_annotation(idx):
    img, _ = train_dataset.__getitem__(idx)
    annotation_file = train_dataset.annotations[idx]
    print(annotation_file)
    path = os.path.join(data_dir, 'VOCdevkit', 'VOC2012', 'Annotations', annotation_file)
    boxes, labels = extract_boxes_labels(path)
    return img, boxes, labels

# main function to get the preprocessed data to train
def get_preprocessed_data():
    # img, boxes, labels = get_image_annotation(0)
    img = []
    boxes = []
    labels = []

    for i in range(1, len(train_dataset)):
        img_temp, boxes_temp, labels_temp = get_image_annotation(i)
        img.append(img_temp)
        boxes.append(boxes_temp)
        labels.append(labels_temp)

    return img, boxes, labels
