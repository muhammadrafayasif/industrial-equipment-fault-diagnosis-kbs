import pickle
from features import load_and_process
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Train on train_data.csv

def train():
    fault_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    type_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)

    X_train, y_train, train_ids = load_and_process(DATA_DIR / "train.csv")
    fault_model.fit(X_train, (y_train != "Normal").astype(int))
    type_model.fit(X_train, y_train)
    
    print("Trained on", len(X_train), "windows.\n")

    # Saving models as a .pkl file
    with open('fault_model.pkl', 'wb') as file:
        pickle.dump(fault_model, file)

    with open('type_model.pkl', 'wb') as file:
        pickle.dump(type_model, file)

if __name__ == '__main__':
    train()        