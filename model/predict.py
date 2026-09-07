import os

import joblib
import numpy as np
import pandas as pd

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_MODEL_DIR, "career_model.joblib")
ENCODER_PATH = os.path.join(_MODEL_DIR, "label_encoder.joblib")


class CareerModel:
    def __init__(self):
        self.pipeline = joblib.load(MODEL_PATH)
        self.label_encoder = joblib.load(ENCODER_PATH)

    def predict(self, skills: str, interests: str, education: str, experience: str):
        # Build a one-row DataFrame matching the structured feature columns
        # the model was trained on (see model/train_model.py's
        # build_feature_frame) -- skills/interests as separate text columns,
        # education as categorical text, experience as numeric. This must
        # stay in sync with build_feature_frame()'s column names/dtypes,
        # since the pipeline's ColumnTransformer selects columns by name.
        try:
            experience_num = float(experience)
        except (TypeError, ValueError):
            experience_num = 0.0

        row = pd.DataFrame(
            [
                {
                    "skills": str(skills).strip().lower(),
                    "interests": str(interests).strip().lower(),
                    "education": str(education).strip().lower(),
                    "experience": experience_num,
                }
            ]
        )

        probs = self.pipeline.predict_proba(row)[0]

        # Top predictions with genuine model probabilities
        sorted_idx = np.argsort(probs)[::-1]
        career_labels = self.label_encoder.inverse_transform(sorted_idx)
        scores = probs[sorted_idx]

        return list(zip(career_labels, scores))
