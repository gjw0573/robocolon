import os
import cv2
import csv
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamPredictor

from groundingdino.util.evaluation import visualize_predictions
from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.datasets.dataset import GroundingDINODataset
from groundingdino.util.inference import load_model
from config import ConfigurationManager  # type: ignore
from metrics import dice_coefficient, iou_score, recall_score, hd95

def setup_data_loader(data_config, batch_size):
    eval_dataset = GroundingDINODataset(
        data_config.val_dir,
        data_config.val_ann
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=lambda x: tuple(zip(*x))
    )
    return eval_loader

def prepare_batch(batch, device="cuda"):
    images, targets = batch

    # Convert list of images to NestedTensor and move to device
    if isinstance(images, (list, tuple)):
        images = nested_tensor_from_tensor_list(images)
    images = images.to(device)

    # Process targets
    captions = []
    for target in targets:
        target['boxes'] = target['boxes'].to(device)
        target['size'] = target['size'].to(device)
        target['labels'] = target['labels'].to(device)
        captions.append(target['caption'])

    return images, targets, captions


def sam_seg(results, save_dir, predictor, data_config):
    total_dice = 0.0
    total_iou = 0.0
    total_recall = 0.0
    hd95_list = []
    predict_times_ms = []  # 记录每张图片的 SAM 推理耗时（毫秒）

    # 新增：为均值±标准差准备原始样本列表
    dice_list = []
    iou_list = []
    recall_list = []

    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, mode="w", newline="") as f:  # 创建CSV
        csv_writer = csv.writer(f)
        csv_writer.writerow(["name", "dice", "iou", "hd95", "recall"])

    for idx, result in tqdm(enumerate(results), total=len(results), desc="Segment", dynamic_ncols=True):  # 每一个样本表示一张图片

        img = result['image'].copy()  # (H,W,C),uint8,numpy数组
        H, W = img.shape[:2]
        pred_boxes = result['pred_boxes'].astype(int)
        pred_scores = result['pred_scores']
        gt_boxes = result['gt_boxes'].astype(int)
        caption = result['caption']
        names = result['image_names'][0]
        class_name = result['class_name'][0][0]
        mask_dir = data_config.val_mask

        gt_path = os.path.join(mask_dir, f"{names}.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = (gt > 0).astype(np.uint8)  # 二值化

        # === 若无检测框，则使用整图默认框，并用其中心点 ===
        if pred_boxes.size == 0:
            chosen_box = np.array([0, 0, W - 1, H - 1], dtype=np.int32)
        else:
            # 保持与你原逻辑一致：取第一个框
            # 如需按最高分选择，可改为：
            # best_idx_by_score = int(np.argmax(pred_scores))
            # chosen_box = pred_boxes[best_idx_by_score]
            chosen_box = pred_boxes[0].astype(np.int32)

        # === 计算该 BOX 的中心点，作为前景点（point_labels=1）===
        x0, y0, x1, y1 = chosen_box.tolist()
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        point_coords = np.array([[cx, cy]], dtype=np.float32)  # SAM 需要像素坐标（浮点）
        point_labels = np.array([1], dtype=np.int32)  # 1=前景，0=背景

        # ==== 仅计 predict() 纯前向 ====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        predictor.set_image(img)

        masks, scores, logits = predictor.predict(
            box=chosen_box,  # [x0,y0,x1,y1]
            point_coords=point_coords,  # [[cx, cy]]
            point_labels=point_labels,  # [1]
            multimask_output=True  # 输出多个 mask
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        predict_times_ms.append(dt_ms)
        # ============================

        # === 选取 scores 最大的 mask ===
        best_idx = int(np.argmax(scores))  # 找分数最大的索引
        best_mask = masks[best_idx].astype(np.uint8)

        # 计算对应的指标
        best_dice = dice_coefficient(best_mask, gt)
        best_iou = iou_score(best_mask, gt)
        best_recall = recall_score(best_mask, gt)
        best_hd95 = hd95(best_mask, gt)

        tqdm.write(f"{names}: BestIndex:{best_idx}, DICE:{best_dice:.4f}, IOU:{best_iou:.4f},"
                   f"HD95:{best_hd95:.2f}, RECALL:{best_recall:.4f}")

        # 保存最佳 mask
        plt.imsave(os.path.join(save_dir, f"{names}.png"), best_mask, cmap='gray')

        with open(csv_path, mode="a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([names,
                                 f"{best_dice:.6f}",
                                 f"{best_iou:.6f}",
                                 f"{best_hd95:.6f}",
                                 f"{best_recall:.6f}",])

            # 累加
        total_dice += best_dice
        total_iou += best_iou
        total_recall += best_recall
        hd95_list.append(best_hd95)

        # 新增：收集样本（用于均值±标准差）
        dice_list.append(best_dice)
        iou_list.append(best_iou)
        recall_list.append(best_recall)

    # n = len(predict_times_ms) if len(predict_times_ms) > 0 else 1
    # avg_dice = total_dice / len(results)
    # avg_iou = total_iou / len(results)
    # avg_time_ms = sum(predict_times_ms) / n

    # tqdm.write(f"AvgDICE: {avg_dice:.4f}, AvgIOU: {avg_iou:.4f}")
    # tqdm.write(f"Forward Time: {avg_time_ms:.2f} ms/image (N={len(predict_times_ms)})")

    # with open(csv_path, mode="a", newline="") as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(["average", f"{avg_dice:.6f}", f"{avg_iou:.6f}"])

    n_time = len(predict_times_ms) if len(predict_times_ms) > 0 else 1

    # ===== 新增：统一的“均值±标准差”格式函数 =====
    def mean_std_str(values, finite_only=False):
        arr = np.array(values, dtype=float)
        if finite_only:
            arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return "nan±nan"
        mean = arr.mean()
        std = arr.std(ddof=1) if arr.size > 1 else 0.0  # 样本标准差
        return f"{mean:.6f}**{std:.6f}"

    # 计算四个指标的“均值±标准差”
    dice_ms = mean_std_str(dice_list)
    iou_ms = mean_std_str(iou_list)
    recall_ms = mean_std_str(recall_list)
    hd95_ms = mean_std_str(hd95_list, finite_only=True)  # 过滤 inf/NaN

    avg_time_ms = sum(predict_times_ms) / n_time

    tqdm.write(f"AvgDICE: {dice_ms}, AvgIOU: {iou_ms},"
               f"AvgHD95: {hd95_ms}, AvgRecall: {recall_ms}")
    tqdm.write(f"Forward Time: {avg_time_ms:.2f} ms/image (N={len(predict_times_ms)})")

    with open(csv_path, mode="a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["average",
                             dice_ms,
                             iou_ms,
                             hd95_ms,
                             recall_ms])


if __name__ == "__main__":
    # DINO模型初始化
    config_path = "configs/dino_sam.yaml"
    device = "cuda"

    data_config, model_config, training_config = ConfigurationManager.load_config(config_path)

    weights_name = os.path.basename(model_config.lora_weights).split(".")[0]
    parent_dir = os.path.dirname(data_config.val_ann)  # 获取父目录路径
    dataset_name = os.path.basename(parent_dir)  # 提取父目录名称
    model = load_model(model_config, training_config.use_lora, device=device).to(device)
    eval_loader = setup_data_loader(data_config, batch_size=1)  # 批量大小设为1
    prepare_batch_fn = lambda batch: prepare_batch(batch, device=device)

    # SAM模型初始化
    sam_checkpoint = "/home/gjw_20307130119/code/AViD/weights/sam_vit_b_01ec64.pth"  # SAM 权重
    model_type = "vit_b"

    # sam_checkpoint = "/home/gjw_20307130119/code/AViD/weights/sam_vit_l_0b3195.pth"  # SAM 权重
    # model_type = "vit_l"

    # sam_checkpoint = "/home/gjw_20307130119/code/AViD/weights/sam_vit_h_4b8939.pth"  # SAM 权重
    # model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    save_dir = os.path.join("dino_sam_output", f"{model_type}_{weights_name}_{dataset_name}")
    os.makedirs(save_dir, exist_ok=True)

    # DINO推理
    results = visualize_predictions(
        model,
        eval_loader,
        prepare_batch_fn,
        num_samples=float('inf'),  # 不指定可视化数目
        score_threshold=0.25,
        device=device
    )
    # 此时返回的是一个列表，表中每个元素代表一张图片的图像，预测框，真实框，名称等信息
    # results({
    #             "image": img,
    #             "pred_boxes": pred_boxes_xyxy.cpu().numpy(),
    #             "pred_scores": filtered_scores.cpu().numpy(),
    #             "gt_boxes": gt_boxes_xyxy.cpu().numpy(),
    #             "caption": captions[0],
    #             "image_names": image_names,
    #             "class_name": class_name
    #         })

    # SAM推理
    sam_seg(
        results=results,
        save_dir=save_dir,
        predictor=predictor,
        data_config=data_config
    )

