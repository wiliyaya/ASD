import os
import glob
import json
from ultralytics import YOLO
from tqdm import tqdm

def main():
    input_dir = 'input_frames'
    output_dir = 'output_data'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载双模型
    print("正在加载 YOLO 模型...")
    pose_model = YOLO('yolov8n-pose.pt')
    obj_model = YOLO('yolov8n.pt')
    target_classes = [37, 41, 77]

    # 获取 input_frames 下的所有子文件夹
    # 如果根目录直接有图片，也把它当做一个单独的任务处理
    video_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    if not video_folders:
        video_folders = [''] # 兼容直接把图片放在 input_frames 根目录的情况

    # 遍历每个视频文件夹
    for folder in video_folders:
        current_input_dir = os.path.join(input_dir, folder)
        image_paths = sorted(glob.glob(os.path.join(current_input_dir, '*.jpg')))
        
        if not image_paths:
            print(f"跳过空文件夹: {folder}")
            continue

        print(f"\n========================================")
        print(f"🎬 正在处理视频目录: {folder if folder else '根目录'} (共 {len(image_paths)} 帧)")
        
        result_data = []
        fps = 10.0 # 假设帧率为 10fps

        for frame_index, img_path in enumerate(tqdm(image_paths, desc="提取进度")):
            frame_info = {
                "frame": frame_index,
                "time": round(frame_index / fps, 2),
                "left_wrist": [0, 0],
                "right_wrist": [0, 0],
                "left_shoulder": [0, 0],
                "right_shoulder": [0, 0],
                "toy": [0, 0]
            }

            # 提取姿态
            pose_results = pose_model(img_path, verbose=False)
            if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                keypoints_data = pose_results[0].keypoints.xy.cpu().numpy()
                if len(keypoints_data) > 0 and len(keypoints_data[0]) >= 11:
                    kpts = keypoints_data[0]
                    frame_info["left_shoulder"] = [int(kpts[5][0]), int(kpts[5][1])]
                    frame_info["right_shoulder"] = [int(kpts[6][0]), int(kpts[6][1])]
                    frame_info["left_wrist"] = [int(kpts[9][0]), int(kpts[9][1])]
                    frame_info["right_wrist"] = [int(kpts[10][0]), int(kpts[10][1])]

            # 提取玩具
            obj_results = obj_model(img_path, classes=target_classes, verbose=False)
            if len(obj_results) > 0 and obj_results[0].boxes is not None:
                boxes = obj_results[0].boxes
                if len(boxes) > 0:
                    coords = boxes[0].xyxy[0].cpu().numpy().tolist()
                    frame_info["toy"] = [int((coords[0] + coords[2]) / 2), int((coords[1] + coords[3]) / 2)]
            
            result_data.append(frame_info)

        # 保存该视频专属的 JSON 文件
        output_filename = f"{folder}.json" if folder else "raw_features.json"
        output_file = os.path.join(output_dir, output_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        
        print(f"✅ {folder} 处理完成！数据保存至 {output_file}")

if __name__ == "__main__":
    main()