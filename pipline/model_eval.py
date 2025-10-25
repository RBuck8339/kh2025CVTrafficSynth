
class YOLOEvaluator:
    """Evaluates trained YOLO models"""

    def __init__(self, model_path):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"✓ Loaded model from {model_path}")
        except ImportError:
            print("❌ ultralytics not installed. Install with: pip install ultralytics")
            self.model = None

    def evaluate_detection(self, data_yaml, split='test'):
        """Evaluate detection model"""
        if self.model is None:
            return None

        print(f"\n📊 Evaluating detection model on {split} set...")

        # Run validation
        metrics = self.model.val(data=data_yaml, split=split)

        print(f"\n📈 Detection Metrics:")
        print(f"  mAP50:     {metrics.box.map50:.4f}")
        print(f"  mAP50-95:  {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall:    {metrics.box.mr:.4f}")

        return metrics

    def evaluate_classification(self, data_dir):
        """Evaluate classification model"""
        if self.model is None:
            return None

        print(f"\n📊 Evaluating classification model...")

        # Run validation
        metrics = self.model.val(data=data_dir, split='test')

        print(f"\n📈 Classification Metrics:")
        print(f"  Top-1 Accuracy: {metrics.top1:.4f}")
        print(f"  Top-5 Accuracy: {metrics.top5:.4f}")

        return metrics

    def test_on_images(self, image_dir, output_dir='predictions', confidence=0.25):
        """Test model on images and save predictions"""
        if self.model is None:
            return

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🔍 Running inference on images in {image_dir}...")

        image_files = [f for f in os.listdir(image_dir)
                      if f.endswith(('.jpg', '.png', '.jpeg'))]

        predictions_summary = []

        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)

            # Predict
            results = self.model(img_path, conf=confidence)

            # Save annotated image
            annotated = results[0].plot()
            output_path = os.path.join(output_dir, f"pred_{img_file}")
            cv2.imwrite(output_path, annotated)

            # Extract predictions
            if hasattr(results[0], 'boxes'):  # Detection
                boxes = results[0].boxes
                pred_info = {
                    'image': img_file,
                    'detections': len(boxes),
                    'close_calls': sum(boxes.cls == 0),
                    'safe_vehicles': sum(boxes.cls == 1)
                }
            else:  # Classification
                probs = results[0].probs
                pred_info = {
                    'image': img_file,
                    'predicted_class': results[0].names[probs.top1],
                    'confidence': float(probs.top1conf)
                }

            predictions_summary.append(pred_info)

        # Save summary
        summary_path = os.path.join(output_dir, 'predictions_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(predictions_summary, f, indent=2)

        print(f"✓ Predictions saved to {output_dir}")
        print(f"✓ Summary saved to {summary_path}")

        return predictions_summary
