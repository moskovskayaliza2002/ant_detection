from torchvision.models.detection.rpn import AnchorGenerator
import os, json, cv2, numpy as np, matplotlib.pyplot as plt
import torch
from datetime import datetime
from engine import train_one_epoch, evaluate
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A # Library for augmentations
import transforms, utils, engine, train
from utils import collate_fn
import torchvision
import argparse


def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )


class ClassDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):                
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for visualization purposes)
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.bboxes_files = sorted(os.listdir(os.path.join(root, "bboxes")))
        self.keypoints_files = sorted(os.listdir(os.path.join(root, "keypoints")))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        bbox_path = os.path.join(self.root, "bboxes", self.bboxes_files[idx])
        keypoints_path = os.path.join(self.root, "keypoints", self.keypoints_files[idx])

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)        
        
        bboxes_original = []
        keypoints_original = []
        
        with open(bbox_path) as f:
            for i in f:
                x_min, y_min, x_max, y_max = map(int, i[:-1].split(' '))
                bboxes_original.append([x_min, y_min, x_max, y_max])
                
        bboxes_labels_original = ['Ant' for _ in bboxes_original] 
        
        with open(keypoints_path) as f:
            for i in f:
                x_center_body, y_center_body, x_center_head, y_center_head = map(int, i[:-1].split(' '))
                visibility = 1
                keypoints_original.append([[x_center_body, y_center_body, visibility], [x_center_head, y_center_head, visibility]])

        if self.transform:   
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints            
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format            
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            
            # Apply augmentations
            transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            img = transformed['image']
            bboxes = transformed['bboxes']
            
            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1,2,2)).tolist()

            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened): # Iterating over objects
                obj_keypoints = []
                for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)
        
        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original        
        
        # Convert everything into a torch tensor        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)       
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are glue tubes
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)        
        img = F.to_tensor(img)
        
        
        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original], dtype=torch.int64) # all objects are glue tubes
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)        
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs_files)


def get_model(num_keypoints, weights_path=None):
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2,
                                                                   rpn_anchor_generator=anchor_generator)
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)        
        
    return model


def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 12
    keypoints_classes_ids2names = {0: 'A', 1: 'H'}
    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (255,0,0), 2)
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 2, (255,0,0), 10)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40,40))
        plt.imshow(image)
        plt.show(block=True)

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 2, (0,255,0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)
        plt.show(block=True)


def train_rcnn(num_epochs, root):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    #KEYPOINTS_FOLDER_TEST = root + '/test_data'
    #KEYPOINTS_FOLDER_TRAIN = root + '/train_data'
    KEYPOINTS_FOLDER_TEST = root + '/TEST_on_real'
    KEYPOINTS_FOLDER_TRAIN = root + '/1to4'
    SAVING_WEIGHTS_PATH = root + '/rcnn_models/'
    
    if not os.path.exists(SAVING_WEIGHTS_PATH):
        os.mkdir(SAVING_WEIGHTS_PATH)
    
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = os.path.join(SAVING_WEIGHTS_PATH, time_str)
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    SAVING_WEIGHTS_PATH += time_str
    
    dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=False)
    dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

    data_loader_train = DataLoader(dataset_train, batch_size=3, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    model = get_model(num_keypoints = 2)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    
    all_clas_loss = []
    all_bbox_loss = []
    all_kps_loss = []
    total_loss = []
    ep = []
    min_loss = float('inf')
    plt.ion()
    plt.show(block=False)
    for epoch in range(num_epochs):
        _, dictionary = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
        print(dictionary)
        
        loss_class = dictionary['loss_classifier'].item()
        loss_bbox = dictionary['loss_box_reg'].item()
        loss_keypoints = dictionary['loss_keypoint'].item()
        total = loss_class + loss_bbox + loss_keypoints
        all_clas_loss.append(loss_class)
        all_bbox_loss.append(loss_bbox)
        all_kps_loss.append(loss_keypoints)
        total_loss.append(total)

        ep.append(epoch)
        plt.cla()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(ep, all_clas_loss, label="Classification", linestyle = '--', color = 'green')
        plt.plot(ep, all_bbox_loss, label="Regression", linestyle = '-.', color = 'darkblue')
        plt.plot(ep, all_kps_loss, label="Keypoints", linestyle = '--', color = 'red')
        plt.plot(ep, total_loss, label="Total", linestyle = ':', color = 'black')
        plt.legend(loc="best")
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.3)

        lr_scheduler.step()
        evaluate(model, data_loader_test, device)
        if total < min_loss:
            min_loss = total
            torch.save(model.state_dict(), SAVING_WEIGHTS_PATH + '/best_weights.pth')
    plt.savefig(SAVING_WEIGHTS_PATH + '/loss.png')
    # Save model weights after training
    torch.save(model.state_dict(), SAVING_WEIGHTS_PATH + '/full_weights.pth')
    return model, SAVING_WEIGHTS_PATH


def visualizate_predictions(model, data_loader_test):
    iterator = iter(data_loader_test)
    images, targets = next(iterator)
    images = list(image.to(device) for image in images)

    with torch.no_grad():
        model.to(device)
        model.eval()
        output = model(images)
        
    image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
    scores = output[0]['scores'].detach().cpu().numpy()

    high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

    keypoints = []
    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:2])) for kp in kps])

    bboxes = []
    for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))
    
    visualize(image, bboxes, keypoints)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', nargs='?', default='/home/ubuntu/ant_detection', help="Specify main directory", type=str)
    parser.add_argument('num_epoch', nargs='?', default=4, help="Specify number of epoch", type=int)
    args = parser.parse_args()
    
    root_path = args.root_path
    num_epoch = args.num_epoch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model, weights_path = train_rcnn(num_epoch, root_path)
    
    #KEYPOINTS_FOLDER_TEST = root_path + '/test_data'
    #dataset = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)
    #data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    #iterator = iter(data_loader)
    #batch = next(iterator)
    
    #test_model = get_model(2, weights_path + '/best_weights.pth')

    #visualizate_predictions(test_model, data_loader)
