
def main():
    """Main execution pipeline"""

    print("=" * 70)
    print(" CARLA CLOSE-CALL DETECTION WITH YOLOv8 - COMPLETE PIPELINE")
    print("=" * 70)

    # Configuration
    COLLECT_DATA = False  # Set to True if you have CARLA running
    BUILD_DATASET = True
    TRAIN_MODEL = True
    EVALUATE_MODEL = True

    TASK = 'detect'  # 'detect' or 'classify'
    MODEL_SIZE = 'n'  # 'n', 's', 'm', 'l', 'x'
    EPOCHS = 50

    # Step 1: Data Collection (requires CARLA simulator)
    if COLLECT_DATA:
        print("\n" + "=" * 70)
        print("STEP 1: DATA COLLECTION FROM CARLA")
        print("=" * 70)

        collector = CARLADataCollector(
            output_dir='carla_dataset',
            close_call_threshold=5.0
        )

        if collector.connect_to_carla():
            collector.spawn_ego_vehicle_with_camera()
            collector.spawn_traffic(num_vehicles=50)
            collector.collect_data(num_frames=1000, save_interval=10)
            collector.cleanup()
        else:
            print("\n⚠ Skipping data collection - CARLA not available")
            print("You can use pre-collected data or run CARLA simulator")

    # Step 2: Build YOLO Dataset
    if BUILD_DATASET:
        print("\n" + "=" * 70)
        print("STEP 2: BUILDING YOLO DATASET")
        print("=" * 70)

        builder = YOLODatasetBuilder(
            carla_dataset_dir='carla_dataset',
            output_dir='yolo_dataset'
        )

        if TASK == 'detect':
            data_path = builder.build_detection_dataset()
        else:
            data_path = builder.build_classification_dataset()

    # Step 3: Train Model
    if TRAIN_MODEL:
        print("\n" + "=" * 70)
        print("STEP 3: TRAINING MODEL")
        print("=" * 70)

        trainer = YOLOTrainer(task=TASK)

        if TASK == 'detect':
            model = trainer.train_detection_model(
                data_yaml='yolo_dataset/dataset.yaml',
                model_size=MODEL_SIZE,
                epochs=EPOCHS
            )
            model_path = 'runs/detect/train/weights/best.pt'
        else:
            model = trainer.train_classification_model(
                data_dir='yolo_dataset',
                model_size=MODEL_SIZE,
                epochs=EPOCHS
            )
            model_path = 'runs/classify/train/weights/best.pt'

    # Step 4: Evaluate Model
    if EVALUATE_MODEL:
        print("\n" + "=" * 70)
        print("STEP 4: EVALUATING MODEL")
        print("=" * 70)

        # Use the trained model path
        evaluator = YOLOEvaluator(model_path)

        if TASK == 'detect':
            metrics = evaluator.evaluate_detection('yolo_dataset/dataset.yaml')
            evaluator.test_on_images(
                'yolo_dataset/images/test',
                output_dir='predictions'
            )
        else:
            metrics = evaluator.evaluate_classification('yolo_dataset')
            evaluator.test_on_images(
                'yolo_dataset/test',
                output_dir='predictions'
            )

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review training plots in runs/ directory")
    print("2. Check predictions in predictions/ directory")
    print("3. Adjust hyperparameters and retrain if needed")
    print("4. Deploy model for real-time inference")


if __name__ == "__main__":
    main()
