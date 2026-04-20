from DAL.core_data import run_data_pipeline
from Models.core_models import run_model_pipeline


def main():
    data_bundle = run_data_pipeline()
    run_model_pipeline(data_bundle)


if __name__ == "__main__":
    main()