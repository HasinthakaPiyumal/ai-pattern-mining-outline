"""Entry point wrapper for training or prediction.

Use model_trainer.py for training/OOF validation and model saving.
Use model_runner.py for inference on unverified data with saved models.
"""

from ensemble_common import CONFIG
from model_trainer import run_training_pipeline
from model_runner import run_prediction_pipeline


def main():
    if CONFIG["predict_mode"]:
        run_prediction_pipeline(CONFIG)
    else:
        run_training_pipeline(CONFIG)


if __name__ == "__main__":
    main()