def run_model_pipeline(data_bundle):
    print("\n===== MODEL PIPELINE =====")
    print("Data received successfully.")
    print("Train dataframe shape:", data_bundle["train_df"].shape)
    print("Train images path:", data_bundle["train_dir"])
    print("Test images path:", data_bundle["test_dir"])
    print("Next step: build ConvNeXt-Tiny training pipeline.")