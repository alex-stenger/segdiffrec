import os
import cv2
import numpy as np
from skimage import metrics
from tqdm import tqdm
import argparse

def calculate_mean_iou_and_dice(pred_mask, gt_mask, class1_value, class2_value):
    class1_pred_mask = (pred_mask == class1_value)
    class2_pred_mask = (pred_mask == class2_value)
    class1_gt_mask = (gt_mask == class1_value)
    class2_gt_mask = (gt_mask == class2_value)

    intersection_class1 = np.logical_and(class1_pred_mask, class1_gt_mask)
    intersection_class2 = np.logical_and(class2_pred_mask, class2_gt_mask)

    union_class1 = np.logical_or(class1_pred_mask, class1_gt_mask)
    union_class2 = np.logical_or(class2_pred_mask, class2_gt_mask)

    iou_class1 = (np.sum(intersection_class1)+1e-6) / (np.sum(union_class1) + 1e-6)
    iou_class2 = (np.sum(intersection_class2)+1e-6) / (np.sum(union_class2) + 1e-6)

    mean_iou = (iou_class1 + iou_class2) / 2

    # Dice coefficient calculation
    dice_class1 = (2 * np.sum(intersection_class1)+1e-6) / (np.sum(class1_pred_mask) + np.sum(class1_gt_mask) + 1e-6)
    dice_class2 = (2 * np.sum(intersection_class2)+1e-6) / (np.sum(class2_pred_mask) + np.sum(class2_gt_mask) + 1e-6)

    mean_dice = (dice_class1 + dice_class2) / 2

    return mean_iou, mean_dice

def calculate_iou_and_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    pred_mask = [[1 if element == 255 else element for element in row] for row in pred_mask]
    gt_mask = [[1 if element == 255 else element for element in row] for row in gt_mask]
    iou = (np.sum(intersection)+1e-6) / (np.sum(union)+1e-6)
    dice = (2 * np.sum(intersection)+1e-6) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-6)

    return iou, dice

def test(pred_path, gt_path, threshold_factor, n_mask=10, retour=False):
    global_iou = 0
    global_mean_iou = 0
    global_dice = 0
    global_mean_dice = 0
    total_samples = 0

    for gt_file in tqdm(os.listdir(gt_path)):
        if not gt_file.endswith('.png'):
            continue

        gt_name = gt_file.split('.')[0]
        gt_img = cv2.imread(os.path.join(gt_path, gt_file), cv2.IMREAD_GRAYSCALE)

        pred_masks = []
        for pred_file in os.listdir(pred_path):
            if not pred_file.startswith(gt_name) or not pred_file.endswith('.png'):
                continue

            pred_img = cv2.imread(os.path.join(pred_path, pred_file), cv2.IMREAD_GRAYSCALE)

            # Resize the prediction to match the size of the ground truth
            if pred_img is None :
                pred_resized = np.random.randint(2,gt_img.shape)
            else :
                pred_resized = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

            pred_masks.append(pred_resized)

        if pred_masks:
            # Calculate mean prediction
            pred_mean = np.mean(pred_masks[0:n_mask-1], axis=0).astype(np.uint8)

            # Save mean prediction before thresholding
            mean_save_path = os.path.join(pred_path, f"{gt_name}_mean.png")
            cv2.imwrite(mean_save_path, pred_mean)

            # Threshold the mean prediction
            threshold_value = int(threshold_factor * 255)
            _, pred_thresholded = cv2.threshold(pred_mean, threshold_value, 255, cv2.THRESH_BINARY)

            # Save mean prediction after thresholding
            mean_thresholded_save_path = os.path.join(pred_path, f"{gt_name}_mean_thresholded.png")
            cv2.imwrite(mean_thresholded_save_path, pred_thresholded)

            # Calculate IoU and Dice score for each pair
            iou, dice = calculate_iou_and_dice(pred_thresholded, gt_img)
            mean_iou,mean_dice = calculate_mean_iou_and_dice(pred_thresholded, gt_img, class1_value=255, class2_value=0)

            print(f"GT: {gt_name}, iou: {iou}, dice: {dice}")

            # Accumulate global scores
            global_iou += iou
            global_mean_iou += mean_iou
            global_dice += dice
            global_mean_dice += mean_dice
            total_samples += 1

    if total_samples > 0:
        # Calculate global scores
        global_iou /= total_samples
        global_dice /= total_samples
        global_mean_iou /= total_samples
        global_mean_dice /= total_samples

        print(f"Global IoU: {global_iou}, Global mean IoU: {global_mean_iou}, Global Dice: {global_dice}, Global mean Dice: {global_mean_dice}")

    if retour :
        return global_iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate IoU and Dice scores for predicted masks compared to ground truth masks.")
    parser.add_argument("--pred_path", type=str, help="Path to the directory containing predicted masks.")
    parser.add_argument("--gt_path", type=str, help="Path to the directory containing ground truth masks.")
    parser.add_argument("--threshold_factor", type=float, default=0.5, help="Threshold factor between 0 and 1.")
    parser.add_argument("--n_mask", type=int, default=10, help="Number of generated mask to compute the mean prediction")
    args = parser.parse_args()

    test(args.pred_path, args.gt_path, args.threshold_factor, args.n_mask)

