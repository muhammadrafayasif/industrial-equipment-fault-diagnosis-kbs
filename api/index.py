from pathlib import Path
import pickle
import tempfile

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from knowledge_base.bearing_faults import get_maintenance_action
from src.features import load_and_process


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REQUIRED_COLUMNS = {"DE_Time", "FE_Time", "RecordingID"}
MINIMUM_ROWS = 2048


def merge_diagnostics(diagnostics):
    merged = {}
    for diagnostic in diagnostics:
        key = (
            diagnostic["recording_id"],
            diagnostic["fault"],
            diagnostic["fault_type"],
        )
        if key not in merged:
            merged[key] = {
                **diagnostic,
                "samples": MINIMUM_ROWS,
                "_confidence_count": 1,
            }
            continue

        merged[key]["samples"] += MINIMUM_ROWS
        confidence_count = merged[key]["_confidence_count"]
        new_confidence_count = confidence_count + 1
        for model_name in ("fault", "fault_type"):
            merged[key]["confidence"][model_name] = (
                merged[key]["confidence"][model_name] * confidence_count
                + diagnostic["confidence"][model_name]
            ) / new_confidence_count
            merged[key]["confidence"][model_name] = round(
                merged[key]["confidence"][model_name], 6
            )
        merged[key]["_confidence_count"] = new_confidence_count

    return [
        {key: value for key, value in diagnostic.items() if key != "_confidence_count"}
        for diagnostic in merged.values()
    ]


def prediction_confidences(model, predictions, features):
    probabilities = model.predict_proba(features)
    class_indexes = {
        str(model_class): index
        for index, model_class in enumerate(model.classes_)
    }
    confidences = []
    for probability_row, prediction in zip(probabilities, predictions):
        class_index = class_indexes.get(str(prediction))
        if class_index is None:
            raise ValueError(f"Model prediction is not present in model classes: {prediction}")
        confidences.append(float(probability_row[class_index]))
    return confidences


def load_model(name):
    model_path = MODEL_DIR / name
    try:
        with model_path.open("rb") as model_file:
            return pickle.load(model_file)
    except (OSError, pickle.PickleError) as exc:
        raise RuntimeError(f"Unable to load model: {model_path}") from exc


fault_model = load_model("fault_model.pkl")
type_model = load_model("type_model.pkl")
app = FastAPI(
    title="KBS Maintenance Diagnostics API",
    description="Generate bearing maintenance actions from vibration CSV windows.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/diagnostics")
async def diagnostics(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    temporary_path = None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_path = Path(temporary_file.name)

        try:
            input_df = pd.read_csv(temporary_path, low_memory=False)
        except (
            OSError,
            UnicodeDecodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as exc:
            raise HTTPException(
                status_code=400, detail="The uploaded file is not a valid CSV."
            ) from exc

        missing_columns = sorted(REQUIRED_COLUMNS - set(input_df.columns))
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"CSV is missing required columns: {', '.join(missing_columns)}.",
            )
        if len(input_df) < MINIMUM_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain at least {MINIMUM_ROWS} data rows.",
            )

        try:
            features, _, recording_ids = load_and_process(temporary_path)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to process the CSV: {exc}",
            ) from exc

        if len(features) == 0:
            raise HTTPException(
                status_code=400,
                detail="The CSV contains no complete 2048-sample window within a recording.",
            )

        try:
            fault_predictions = fault_model.predict(features)
            type_predictions = type_model.predict(features)
            fault_confidences = prediction_confidences(
                fault_model, fault_predictions, features
            )
            type_confidences = prediction_confidences(
                type_model, type_predictions, features
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Model prediction failed: {exc}"
            ) from exc

        diagnostics = []
        window_numbers = {}
        for recording_id, fault, fault_type, fault_confidence, type_confidence in zip(
            recording_ids,
            fault_predictions,
            type_predictions,
            fault_confidences,
            type_confidences,
        ):
            recording_key = str(recording_id)
            window_number = window_numbers.get(recording_key, 0)
            window_numbers[recording_key] = window_number + 1
            diagnostics.append(
                {
                    "recording_id": recording_key,
                    "window": window_number,
                    "fault": int(fault),
                    "fault_type": str(fault_type),
                    "confidence": {
                        "fault": fault_confidence,
                        "fault_type": type_confidence,
                    },
                    "maintenance_action": get_maintenance_action(
                        int(fault), str(fault_type)
                    ),
                }
            )

        merged_diagnostics = merge_diagnostics(diagnostics)
        return merged_diagnostics
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)