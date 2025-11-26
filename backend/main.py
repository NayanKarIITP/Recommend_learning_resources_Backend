# backend/main.py
import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.decomposition import NMF

# --------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
N_COMPONENTS = 50 
TOP_N_DEFAULT = 5
# ---------------------------

app = FastAPI(title="Student Resource Recommender")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
interaction_matrix: Optional[pd.DataFrame] = None
pred_matrix: Optional[np.ndarray] = None
student_ids: List[int] = []
resource_ids: List[int] = []
student_index_map = {}

vle_df: Optional[pd.DataFrame] = None          # metadata about VLE activities (title, activity_type, etc.)
student_assessment: Optional[pd.DataFrame] = None
engagement_df: Optional[pd.DataFrame] = None   # aggregated engagement for charts
student_info_df: Optional[pd.DataFrame] = None # optional, if studentInfo.csv exists


# ---------------------------------------------------------
# LOAD DATA + TRAIN MODEL
# ---------------------------------------------------------
def load_oulad_and_train():
    global pred_matrix, interaction_matrix
    global student_ids, resource_ids, student_index_map

    print("🔄 Loading pre-trained NMF model...")

    # ---- Load Pretrained Matrices ----
    W_path = os.path.join(DATA_PATH, "W.npy")
    H_path = os.path.join(DATA_PATH, "H.npy")
    students_path = os.path.join(DATA_PATH, "students.csv")
    resources_path = os.path.join(DATA_PATH, "resources.csv")

    # Safety checks
    if not (os.path.exists(W_path) and os.path.exists(H_path)):
        raise RuntimeError("❌ Missing W.npy or H.npy — train_model.py must be run first!")

    if not (os.path.exists(students_path) and os.path.exists(resources_path)):
        raise RuntimeError("❌ Missing students.csv or resources.csv — run train_model.py first!")

    # Load model parts
    W = np.load(W_path)
    H = np.load(H_path)

    print("✔ Loaded W, H")

    # Build prediction matrix
    pred = W @ H
    pred_matrix = pred

    # Load student and resource IDs
    student_ids_list = pd.read_csv(students_path).iloc[:, 0].tolist()
    resource_ids_list = pd.read_csv(resources_path).iloc[:, 0].tolist()

    student_ids[:] = student_ids_list
    resource_ids[:] = resource_ids_list

    # Build lookup map
    student_index_map.clear()
    student_index_map.update({sid: i for i, sid in enumerate(student_ids)})

    print(f"✔ Loaded {len(student_ids)} students")
    print(f"✔ Loaded {len(resource_ids)} resources")

    # OPTIONAL: create dummy interaction matrix for compatibility
    # This avoids KeyErrors if code expects interaction_matrix
    interaction_matrix = pd.DataFrame(
        0,
        index=student_ids,
        columns=resource_ids
    )

    print("Pre-trained model loaded successfully!")



# ---------------------------------------------------------
# RECOMMENDATION FUNCTIONS
# ---------------------------------------------------------
def recommend_resources_nmf(student_id: int, top_n: int):
    if student_id not in student_index_map:
        raise KeyError("Student not found")

    if pred_matrix is None or interaction_matrix is None:
        raise RuntimeError("Model not initialized")

    s_idx = student_index_map[student_id]

    raw_scores = pred_matrix[s_idx]

    # Sanitize scores (NaN, inf → 0)
    scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)

    already_used = interaction_matrix.loc[student_id]
    already_used = already_used[already_used > 0].index.to_list()

    scored_items: List[tuple[int, float]] = []
    for j, score in enumerate(scores):
        rid = resource_ids[j]
        if rid not in already_used:
            safe_score = float(score) if np.isfinite(score) else 0.0
            scored_items.append((int(rid), safe_score))

    scored_items.sort(key=lambda x: x[1], reverse=True)

    return scored_items[:top_n]


def recommend_with_details(student_id: int, top_n: int):
    if vle_df is None:
        return []

    recs = recommend_resources_nmf(student_id, top_n)

    if not recs:
        return []

    rec_ids = [r[0] for r in recs]
    scores_map = {rid: score for rid, score in recs}

    subset = vle_df[vle_df["id_site"].isin(rec_ids)].copy()

    # Map scores
    subset["score"] = subset["id_site"].map(scores_map)

    # Clean NaN / inf
    subset = subset.replace([np.inf, -np.inf], 0)
    subset = subset.fillna(0)

    subset = subset.sort_values("score", ascending=False)

    result = []
    for _, row in subset.iterrows():
        result.append(
            {
                "id_site": int(row["id_site"]),
                # ✅ For UI: show title instead of raw ID
                "title": str(row.get("title", "")),
                "activity_type": str(row.get("activity_type", "")),
                "week_from": int(row["week_from"]) if not pd.isna(row.get("week_from")) else None,
                "week_to": int(row["week_to"]) if not pd.isna(row.get("week_to")) else None,
                # Extra optional fields if present (for "More Details" modal)
                "num_clicks": int(row["sum_click"]) if "sum_click" in row and not pd.isna(row["sum_click"]) else None,
                "score": float(row["score"]) if np.isfinite(row["score"]) else 0.0,
            }
        )
    return result


# ---------------------------------------------------------
# PERFORMANCE
# ---------------------------------------------------------
def performance_summary(student_id: int):
    if student_assessment is None:
        return None

    sa = student_assessment
    rows = sa[sa["id_student"] == student_id]
    if rows.empty:
        return None

    avg_score = float(rows["score"].mean())
    pass_rate = float((rows["score"] >= 40).mean())
    attempts = int(rows.shape[0])

    return {
        "student_id": student_id,
        "attempts": attempts,
        "avg_score": avg_score,
        "pass_rate": pass_rate,
    }


def performance_detail(student_id: int):
    """
    Detailed records for charts:
    each assessment attempt with its score and pass/fail.
    """
    if student_assessment is None:
        return []

    sa = student_assessment
    rows = sa[sa["id_student"] == student_id].copy()
    if rows.empty:
        return []

    # Clean NaNs
    rows = rows.replace([np.inf, -np.inf], 0)
    rows = rows.fillna(0)

    details = []
    for _, r in rows.iterrows():
        details.append(
            {
                "id_assessment": int(r["id_assessment"]),
                "score": float(r["score"]),
                "is_pass": bool(r["score"] >= 40),
                # include date_submitted if present
                "date_submitted": int(r["date_submitted"]) if "date_submitted" in r and not pd.isna(r["date_submitted"]) else None,
            }
        )
    return details


# ---------------------------------------------------------
# ENGAGEMENT (for charts)
# ---------------------------------------------------------
def engagement_timeseries(student_id: int):
    """
    Returns engagement over time (per week_from) for a student.
    """
    if engagement_df is None:
        return []

    df = engagement_df
    if "week_from" in df.columns:
        rows = df[df["id_student"] == student_id].copy()
        if rows.empty:
            return []
        rows = rows.replace([np.inf, -np.inf], 0).fillna(0)
        result = []
        for _, r in rows.iterrows():
            result.append(
                {
                    "week": int(r["week_from"]) if not pd.isna(r["week_from"]) else None,
                    "sum_click": float(r["sum_click"]),
                }
            )
        return result
    else:
        # fallback aggregate
        rows = df[df["id_student"] == student_id].copy()
        if rows.empty:
            return []
        rows = rows.replace([np.inf, -np.inf], 0).fillna(0)
        result = []
        for _, r in rows.iterrows():
            result.append(
                {
                    "week": None,
                    "sum_click": float(r.get("total_clicks", 0)),
                }
            )
        return result


# ---------------------------------------------------------
# STUDENT LIST (for dropdown)
# ---------------------------------------------------------
def get_student_list():
    """
    Returns a list of valid students for the dropdown.
    If studentInfo is available, uses extra metadata for labels.
    """
    result = []

    if not student_ids:
        return result

    # If we have studentInfo.csv, use nicer labels
    if student_info_df is not None and "id_student" in student_info_df.columns:
        info = student_info_df[student_info_df["id_student"].isin(student_ids)].copy()

        # Only keep a few safe columns if they exist
        cols = ["id_student"]
        for c in ["age_band", "gender", "region", "final_result"]:
            if c in info.columns:
                cols.append(c)

        info = info[cols].drop_duplicates(subset=["id_student"])

        for _, r in info.iterrows():
            sid = int(r["id_student"])
            label_parts = [f"Student {sid}"]
            if "age_band" in r and isinstance(r["age_band"], str):
                label_parts.append(str(r["age_band"]))
            if "gender" in r and isinstance(r["gender"], str):
                label_parts.append(str(r["gender"]))
            label = " | ".join(label_parts)
            result.append(
                {
                    "id_student": sid,
                    "label": label,
                    # extra metadata if UI wants it
                    "age_band": r.get("age_band", None),
                    "gender": r.get("gender", None),
                    "region": r.get("region", None),
                    "final_result": r.get("final_result", None),
                }
            )
    else:
        # Fallback: just list IDs
        for sid in student_ids:
            result.append(
                {
                    "id_student": int(sid),
                    "label": f"Student {sid}",
                }
            )

    # Sort by id for stable ordering
    result.sort(key=lambda x: x["id_student"])
    return result


# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------
@app.on_event("startup")
def startup_event():
    load_oulad_and_train()


# ---------- Recommendation ----------
@app.get("/recommend")
def recommend_endpoint(student_id: int, top_n: int = TOP_N_DEFAULT):
    # Case 1: Student ID does NOT exist at all
    if student_id not in student_index_map:
        return {
            "student_id": student_id,
            "top_n": top_n,
            "recommendations": [],
            "message": "❗ This student is not found in the dataset. Please enter a valid student ID.",
        }

    # Try generating recommendations
    try:
        recs = recommend_with_details(student_id, top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Case 2: Student exists but has NO resource recommendations
    if len(recs) == 0:
        return {
            "student_id": student_id,
            "top_n": top_n,
            "recommendations": [],
            "message": "ℹ️ This student has no VLE activity or no new resources to recommend.",
        }

    # Case 3: Normal successful recommendation
    return {
        "student_id": student_id,
        "top_n": top_n,
        "recommendations": recs,
        "message": "✅ Recommendations successfully generated.",
    }


# ---------- Performance ----------
@app.get("/performance")
def performance_endpoint(student_id: int):
    summary = performance_summary(student_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="No assessment data")
    return summary


@app.get("/performance/detail")
def performance_detail_endpoint(student_id: int):
    details = performance_detail(student_id)
    return {
        "student_id": student_id,
        "records": details,
    }


# ---------- Engagement ----------
@app.get("/engagement")
def engagement_endpoint(student_id: int):
    series = engagement_timeseries(student_id)
    return {
        "student_id": student_id,
        "engagement": series,
    }


# ---------- Student list (dropdown) ----------
@app.get("/students/list")
def students_list_endpoint():
    students = get_student_list()
    return {
        "count": len(students),
        "students": students,
    }


# Optional simple health check
@app.get("/health")
def health():
    return {"status": "ok"}



@app.get("/students/search")
def search_students(query: str = "", limit: int = 10):
    """
    Fast search for student IDs.
    Example: /students/search?query=28
    Returns top 10 matches.
    """
    query = query.strip()

    if not query:
        return {"students": []}

    # Filter matching IDs (string startswith)
    matches = [sid for sid in student_ids if str(sid).startswith(query)]

    # Build results with labels (uses studentInfo if available)
    results = []
    count = 0
    for sid in matches:
        if count >= limit:
            break

        label = f"Student {sid}"
        if student_info_df is not None:
            row = student_info_df[student_info_df["id_student"] == sid]
            if not row.empty:
                gender = row.iloc[0].get("gender", None)
                age_band = row.iloc[0].get("age_band", None)
                if isinstance(gender, str):
                    label += f" | {gender}"
                if isinstance(age_band, str):
                    label += f" | {age_band}"

        results.append({"id_student": sid, "label": label})
        count += 1

    return {"students": results}

