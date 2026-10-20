from features import load_and_process
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import pandas as pd
import pickle

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

X_test, y_test, test_ids = load_and_process(DATA_DIR / 'test.csv')
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

# Load pre-trained models stored in /models
with open(PROJECT_DIR / 'models/fault_model.pkl', 'rb') as file:
    fault_model = pickle.load(file)

with open(PROJECT_DIR / 'models/type_model.pkl', 'rb') as file:
    type_model = pickle.load(file)

for name, model, truth in [("Fault model (0 = healthy, 1 = faulty)", fault_model,
                            (y_test != "Normal").astype(int)),
                           ("Type model (fault type + location)", type_model, y_test)]:
    pred = model.predict(X_test)
    labels = sorted(set(truth) | set(pred))
    print(f"=== {name} ===")
    print(classification_report(truth, pred, zero_division=0))
    print(pd.DataFrame(confusion_matrix(truth, pred, labels=labels),
                       index=[f"true {l}" for l in labels],
                       columns=[f"pred {l}" for l in labels]), "\n")