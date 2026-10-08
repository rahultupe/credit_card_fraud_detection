from model_monitering import add_actual_labels


log_file = "prediction_logs.csv"


actual_labels = [
  1,0,0
]


add_actual_labels(
    log_file,
    actual_labels
)

print("Actual labels added successfully.")