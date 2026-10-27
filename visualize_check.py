import os
import json
import cv2

def main():
    input_dir = 'input_frames'
    output_dir = 'output_debug'
    json_file = 'output_data/raw_features.json'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(json_file):
        print(f"找不到 JSON 文件：{json_file}，请先运行 process_frames.py")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for frame in data.get("frames", []):
        img_name = frame["image_name"]
        img_path = os.path.join(input_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        image = cv2.imread(img_path)
        if image is None:
            continue

        # 画人的关键点
        for person in frame.get("persons", []):
            kpts = person.get("keypoints", {})
            
            # 用红点画手腕
            for part in ["left_wrist", "right_wrist"]:
                pt = kpts.get(part)
                if pt and pt[0] > 0 and pt[1] > 0:
                    cv2.circle(image, (int(pt[0]), int(pt[1])), radius=5, color=(0, 0, 255), thickness=-1)
            
            # 用蓝点画肩膀
            for part in ["left_shoulder", "right_shoulder"]:
                pt = kpts.get(part)
                if pt and pt[0] > 0 and pt[1] > 0:
                    cv2.circle(image, (int(pt[0]), int(pt[1])), radius=5, color=(255, 0, 0), thickness=-1)

        # 画玩具的框 (绿色)
        for obj in frame.get("objects", []):
            bbox = obj.get("bbox")
            cls_name = obj.get("class")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        save_path = os.path.join(output_dir, f"debug_{img_name}")
        cv2.imwrite(save_path, image)
        print(f"已生成校验图: {save_path}")

if __name__ == "__main__":
    main()