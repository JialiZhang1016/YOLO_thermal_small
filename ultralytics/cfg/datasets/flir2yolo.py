import os
import json

def process_annotations(json_path, output_dir, selected_labels, type="train", width=640.0, height=512.0):
    """
    读取 index.json 文件，并将其中的对象类别转换为 YOLO 格式标签，输出到指定路径。
    同时统计每个类别的数量，并保存类别映射和统计信息。

    :param json_path: 输入的 index.json 文件路径
    :param output_dir: 输出的 YOLO 标签文件的保存路径
    :param selected_labels: 要处理的对象类别列表
    :param width: 图像宽度（默认值为640.0）
    :param height: 图像高度（默认值为512.0）
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录每个类别的计数
    class_count = {label: 0 for label in selected_labels}
    
    # 读取 JSON 文件
    with open(json_path) as f:
        data = json.load(f)
        frames = data['frames']

    # 处理每一帧
    for frame in frames:
        image_name = "video-" + frame["videoMetadata"]["videoId"] + "-frame-" + str(frame["videoMetadata"]["frameIndex"]).zfill(6) + "-" + frame["datasetFrameId"] + ".jpg"
        converted_results = []

        for anno in frame["annotations"]:
            label = anno['labels'][0]  # 获取标签
            if label in selected_labels:  # 检查标签是否在用户选择的类别中
                bbox_height = anno["boundingBox"]["h"]
                bbox_width = anno["boundingBox"]["w"]
                x = anno["boundingBox"]["x"]
                y = anno["boundingBox"]["y"]
                cat_id = selected_labels.index(label)  # 获取类别ID
                
                # 统计该类别的数量
                class_count[label] += 1

                # 计算 YOLO 格式的相对坐标
                x_center, y_center = (x + bbox_width / 2, y + bbox_height / 2)
                x_rel, y_rel = (x_center / width, y_center / height)
                w_rel, h_rel = (bbox_width / width, bbox_height / height)
                
                # 将结果追加到列表
                converted_results.append((cat_id, x_rel, y_rel, w_rel, h_rel))
        
        # 如果有结果，保存到文件
        if converted_results:
            output_file = os.path.join(output_dir, str(image_name)[:-4] + '.txt')
            with open(output_file, 'w') as file:
                file.write('\n'.join('%d %.6f %.6f %.6f %.6f' % res for res in converted_results))

    # 保存类别统计和映射信息
    mapping_file = os.path.join(output_dir,"..", str(type) + '.txt')
    with open(mapping_file, 'w') as f:
        f.write("YOLO Label Mapping and Class Counts:\n")
        for i, label in enumerate(selected_labels):
            f.write(f"{i}: {label}, Count: {class_count[label]}\n")

    print(f"处理完成，共处理了 {len(frames)} 帧。类别统计已保存到 {mapping_file}")

# 使用示例
# type = "train"
# json_path = "/Users/captainzhang/Documents/Research/datasets/FLIR_ADAS_v2/images_thermal_train/index.json"  # 你的 index.json 文件路径
# output_dir = "/Users/captainzhang/Documents/Research/datasets/FLIR_YOLO/thermal/label/train"  # 输出标签的目录
# selected_labels = ['person','bike','car','motor','bus','truck','light', 'hydrant','sign','other vehicle']  # 用户选择的对象类别

# 使用示例
type = "val"
json_path = "/Users/captainzhang/Documents/Research/datasets/FLIR_ADAS_v2/images_thermal_val/index.json"  # 你的 index.json 文件路径
output_dir = "/Users/captainzhang/Documents/Research/datasets/FLIR_YOLO/thermal/label/val"  # 输出标签的目录
selected_labels = ['person','bike','car','motor','bus','truck','light', 'hydrant','sign','other vehicle']  # 用户选择的对象类别

process_annotations(json_path, output_dir, selected_labels, type)
