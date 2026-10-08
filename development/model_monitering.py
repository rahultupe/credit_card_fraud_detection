import os
import pandas as pd

def save_prediction(
  input_data,
  prediction,
  probability,
  model_version,
  log_file      
):
    log_data=input_data.copy()

    log_data["prediction"] = prediction
    log_data["probability"] = probability
    log_data["model_version"] = model_version
    log_data["Actual_Label"] = None

    file_exists = os.path.exists(log_file)

    log_data.to_csv(
        log_file,
        mode = "a",
        header = not file_exists,
        index = False
    )

    print("Prediction saved to:", log_file)
    print("Rows saved:", len(log_data))

def add_actual_labels(log_file,actual_labels):

    df = pd.read_csv(log_file)

    df["Actual_Label"] = actual_labels

    df.to_csv(
        log_file,
        index = False
    )