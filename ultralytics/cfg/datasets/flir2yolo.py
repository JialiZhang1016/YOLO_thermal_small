import os
import json

def process_annotations(json_path, output_dir, selected_labels, type="train", width=640.0, height=512.0):
    """
    Read the index.json file and convert the object categories to YOLO format labels, output to the specified path.
    Also count the number of instances for each category and save the category mapping and statistics.

    :param json_path: Path to the input index.json file
    :param output_dir: Path to save the output YOLO label files
    :param selected_labels: List of object categories to process
    :param width: Image width (default value is 640.0)
    :param height: Image height (default value is 512.0)
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Record the count for each category
    class_count = {label: 0 for label in selected_labels}
    
    # Read the JSON file
    with open(json_path) as f:
        data = json.load(f)
        frames = data['frames']

    # Process each frame
    for frame in frames:
        image_name = "video-" + frame["videoMetadata"]["videoId"] + "-frame-" + str(frame["videoMetadata"]["frameIndex"]).zfill(6) + "-" + frame["datasetFrameId"] + ".jpg"
        converted_results = []

        for anno in frame["annotations"]:
            label = anno['labels'][0]  # Get the label
            if label in selected_labels:  # Check if the label is in the user-selected categories
                bbox_height = anno["boundingBox"]["h"]
                bbox_width = anno["boundingBox"]["w"]
                x = anno["boundingBox"]["x"]
                y = anno["boundingBox"]["y"]
                cat_id = selected_labels.index(label)  # Get the category ID
                
                # Count the instances of this category
                class_count[label] += 1

                # Calculate relative coordinates in YOLO format
                x_center, y_center = (x + bbox_width / 2, y + bbox_height / 2)
                x_rel, y_rel = (x_center / width, y_center / height)
                w_rel, h_rel = (bbox_width / width, bbox_height / height)
                
                # Append the result to the list
                converted_results.append((cat_id, x_rel, y_rel, w_rel, h_rel))
        
        # If there are results, save to file
        if converted_results:
            output_file = os.path.join(output_dir, str(image_name)[:-4] + '.txt')
            with open(output_file, 'w') as file:
                file.write('\n'.join('%d %.6f %.6f %.6f %.6f' % res for res in converted_results))

    # Save category statistics and mapping information
    mapping_file = os.path.join(output_dir,"..", str(type) + '.txt')
    with open(mapping_file, 'w') as f:
        f.write("YOLO Label Mapping and Class Counts:\n")
        for i, label in enumerate(selected_labels):
            f.write(f"{i}: {label}, Count: {class_count[label]}\n")

    print(f"Processing completed, processed {len(frames)} frames. Category statistics have been saved to {mapping_file}")

# Usage example
# type = "train"
# json_path = "/Users/captainzhang/Documents/Research/datasets/FLIR_ADAS_v2/images_thermal_train/index.json"  # Path to your index.json file
# output_dir = "/Users/captainzhang/Documents/Research/datasets/FLIR_YOLO/thermal/label/train"  # Directory for output labels
# selected_labels = ['person','bike','car','motor','bus','truck','light', 'hydrant','sign','other vehicle']  # User-selected object categories

# Usage example
type = "val"
json_path = "/Users/captainzhang/Documents/Research/datasets/FLIR_ADAS_v2/images_thermal_val/index.json"  # Path to your index.json file
output_dir = "/Users/captainzhang/Documents/Research/datasets/FLIR_YOLO/thermal/label/val"  # Directory for output labels
selected_labels = ['person','bike','car','motor','bus','truck','light', 'hydrant','sign','other vehicle']  # User-selected object categories

process_annotations(json_path, output_dir, selected_labels, type)
