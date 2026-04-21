from DAL.core_data import run_data_pipeline
from Models.core_models import run_model_pipeline


def main():
    try:
        print("\n===== STARTING FULL PIPELINE =====")

        pipeline_bundle = run_data_pipeline()

        print("\n===== DATA PIPELINE FINISHED =====")
        print("Reports dir:", pipeline_bundle["reports_dir"])
        print("Metadata path:", pipeline_bundle["metadata_path"])

        model_outputs = run_model_pipeline(pipeline_bundle)

        print("\n===== MODEL PIPELINE FINISHED =====")
        return model_outputs

    except Exception as e:
        print("\n[ERROR] Pipeline failed:")
        print(str(e))
        raise


if __name__ == "__main__":
    main()