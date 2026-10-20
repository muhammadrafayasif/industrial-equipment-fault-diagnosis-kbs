from pathlib import Path
import pickle
import tempfile

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from knowledge_base.bearing_faults import get_maintenance_action
from src.features import load_and_process


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REQUIRED_COLUMNS = {"DE_Time", "FE_Time", "RecordingID"}
MINIMUM_ROWS = 2048


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
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Model prediction failed: {exc}"
            ) from exc

        diagnostics = []
        window_numbers = {}
        for recording_id, fault, fault_type in zip(
            recording_ids, fault_predictions, type_predictions
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
                    "maintenance_action": get_maintenance_action(
                        int(fault), str(fault_type)
                    ),
                }
            )

        return diagnostics
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)