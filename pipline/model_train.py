
class YOLOTrainer:
    """Trains YOLOv8 models"""

    def __init__(self, task='detect'):
        """
        task: 'detect' for object detection or 'classify' for classification
        """
        self.task = task

    def train_detection_model(self, data_yaml, model_size='n', epochs=100,
                            batch_size=16, img_size=640):
        """Train YOLOv8 detection model"""
        try:
            from ultralytics import YOLO
        except ImportError:
            print("❌ ultralytics not installed. Install with: pip install ultralytics")
            return None

        print(f"\n🚀 Training YOLOv8{model_size} detection model...")

        # Load pretrained model
        model = YOLO(f'yolov8{model_size}.pt')

        # Train
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=20,
            save=True,
            plots=True,
            device='cuda:0',  # Use 'cpu' if no GPU
            workers=4,
            optimizer='Adam',
            lr0=0.001,

            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1
        )

        print(f"\n✓ Training complete! Model saved to: {model.trainer.save_dir}")
        return model

    def train_classification_model(self, data_dir, model_size='n',
                                  epochs=100, batch_size=32, img_size=224):
        """Train YOLOv8 classification model"""
        try:
            from ultralytics import YOLO
        except ImportError:
            print("❌ ultralytics not installed. Install with: pip install ultralytics")
            return None

        print(f"\n🚀 Training YOLOv8{model_size} classification model...")

        # Load pretrained model
        model = YOLO(f'yolov8{model_size}-cls.pt')

        # Train
        results = model.train(
            data=data_dir,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=20,
            save=True,
            plots=True,
            device='cuda:0',  # Use 'cpu' if no GPU
            workers=4,
            optimizer='Adam',
            lr0=0.001
        )

        print(f"\n✓ Training complete! Model saved to: {model.trainer.save_dir}")
        return model
