

# ============================================================================
# PART 2: DATASET PREPARATION FOR YOLO
# ============================================================================

class YOLODatasetBuilder:
    """Converts CARLA data to YOLO format"""

    def __init__(self, carla_dataset_dir, output_dir='yolo_dataset'):
        self.carla_dir = carla_dataset_dir
        self.output_dir = output_dir
        self.class_names = ['vehicle_close_call', 'vehicle_safe']

    def build_detection_dataset(self, train_split=0.7, val_split=0.15):
        """Build YOLO object detection dataset"""
        print("\n📦 Building YOLO detection dataset...")

        # Create directory structure
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'labels', split), exist_ok=True)

        # Get all metadata files
        metadata_dir = os.path.join(self.carla_dir, 'metadata')
        metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]

        # Shuffle and split
        random.shuffle(metadata_files)
        n_train = int(len(metadata_files) * train_split)
        n_val = int(len(metadata_files) * val_split)

        splits = {
            'train': metadata_files[:n_train],
            'val': metadata_files[n_train:n_train + n_val],
            'test': metadata_files[n_train + n_val:]
        }

        stats = {'train': 0, 'val': 0, 'test': 0}

        for split_name, files in splits.items():
            for meta_file in files:
                # Load metadata
                with open(os.path.join(metadata_dir, meta_file), 'r') as f:
                    metadata = json.load(f)

                filename = metadata['filename']

                # Copy image
                src_img = os.path.join(self.carla_dir, 'raw_images', f"{filename}.jpg")
                dst_img = os.path.join(self.output_dir, 'images', split_name, f"{filename}.jpg")
                shutil.copy(src_img, dst_img)

                # Create YOLO annotation
                annotations = []
                img = cv2.imread(src_img)
                img_height, img_width = img.shape[:2]

                for vehicle in metadata['vehicles']:
                    bbox = vehicle['bbox_2d']
                    class_id = 0 if vehicle['is_close_call'] else 1

                    # Convert to YOLO format (normalized)
                    x_center = ((bbox[0] + bbox[2]) / 2) / img_width
                    y_center = ((bbox[1] + bbox[3]) / 2) / img_height
                    width = (bbox[2] - bbox[0]) / img_width
                    height = (bbox[3] - bbox[1]) / img_height

                    annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # Save annotation
                label_path = os.path.join(self.output_dir, 'labels', split_name, f"{filename}.txt")
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annotations))

                stats[split_name] += 1

        # Create dataset.yaml
        yaml_content = f"""path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val
test: images/test

nc: 2
names: {self.class_names}
"""

        with open(os.path.join(self.output_dir, 'dataset.yaml'), 'w') as f:
            f.write(yaml_content)

        print(f"\n✓ Dataset created:")
        print(f"  Train: {stats['train']} images")
        print(f"  Val:   {stats['val']} images")
        print(f"  Test:  {stats['test']} images")

        return os.path.join(self.output_dir, 'dataset.yaml')

    def build_classification_dataset(self, train_split=0.7, val_split=0.15):
        """Build YOLO classification dataset"""
        print("\n📦 Building YOLO classification dataset...")

        # Create directory structure
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, split, 'close_call'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'safe'), exist_ok=True)

        # Get all images
        raw_images_dir = os.path.join(self.carla_dir, 'raw_images')
        image_files = [f for f in os.listdir(raw_images_dir) if f.endswith('.jpg')]

        # Shuffle and split
        random.shuffle(image_files)
        n_train = int(len(image_files) * train_split)
        n_val = int(len(image_files) * val_split)

        splits = {
            'train': image_files[:n_train],
            'val': image_files[n_train:n_train + n_val],
            'test': image_files[n_train + n_val:]
        }

        stats = {'train': {'close_call': 0, 'safe': 0},
                'val': {'close_call': 0, 'safe': 0},
                'test': {'close_call': 0, 'safe': 0}}

        for split_name, files in splits.items():
            for img_file in files:
                # Determine class from filename
                class_name = 'close_call' if img_file.startswith('close_call') else 'safe'

                # Copy image
                src = os.path.join(raw_images_dir, img_file)
                dst = os.path.join(self.output_dir, split_name, class_name, img_file)
                shutil.copy(src, dst)

                stats[split_name][class_name] += 1

        print(f"\n✓ Classification dataset created:")
        for split in ['train', 'val', 'test']:
            print(f"  {split.capitalize()}:")
            print(f"    Close call: {stats[split]['close_call']}")
            print(f"    Safe:       {stats[split]['safe']}")

        return self.output_dir
