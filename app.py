#!/usr/bin/env python
import os
import re
import time
import secrets
from pathlib import Path
from math import radians, sin, cos, asin, sqrt
from functools import wraps
from collections import defaultdict

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "derived_data"

# Use geocoded hospital metrics file
HOSPITAL_CSV = DATA_DIR / "hospital_metrics_geocoded.csv"
PROVIDER_CSV = DATA_DIR / "provider_metrics.csv"

app = Flask(__name__)

# Secret key for sessions - in production, use environment variable
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))

# -------------------------------------------------------------------
# Authentication Configuration
# -------------------------------------------------------------------
# Store hashed password (never store plaintext!)
AUTH_USERNAME = "neuroinnovate"
AUTH_PASSWORD_HASH = generate_password_hash("mobydon", method='pbkdf2:sha256')

# Rate limiting configuration
MAX_FAILED_ATTEMPTS = 10
LOCKOUT_DURATION_SECONDS = 10 * 60  # 10 minutes

# In-memory store for rate limiting (use Redis in production for multi-worker)
# Structure: { ip_address: { 'attempts': int, 'lockout_until': float, 'last_attempt': float } }
login_attempts = defaultdict(lambda: {'attempts': 0, 'lockout_until': 0, 'last_attempt': 0})


# -------------------------------------------------------------------
# Authentication Helpers
# -------------------------------------------------------------------
def get_client_ip():
    """Get the client's IP address, accounting for proxies."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr or 'unknown'


def is_locked_out(ip):
    """Check if an IP is currently locked out."""
    record = login_attempts[ip]
    if record['lockout_until'] > time.time():
        return True
    return False


def get_lockout_remaining(ip):
    """Get remaining lockout time in seconds."""
    record = login_attempts[ip]
    remaining = record['lockout_until'] - time.time()
    return max(0, int(remaining))


def record_failed_attempt(ip):
    """Record a failed login attempt and potentially trigger lockout."""
    record = login_attempts[ip]
    record['attempts'] += 1
    record['last_attempt'] = time.time()
    
    if record['attempts'] >= MAX_FAILED_ATTEMPTS:
        record['lockout_until'] = time.time() + LOCKOUT_DURATION_SECONDS
        return True  # Locked out
    return False


def reset_attempts(ip):
    """Reset failed attempts on successful login."""
    login_attempts[ip] = {'attempts': 0, 'lockout_until': 0, 'last_attempt': 0}


def get_remaining_attempts(ip):
    """Get remaining login attempts before lockout."""
    record = login_attempts[ip]
    return max(0, MAX_FAILED_ATTEMPTS - record['attempts'])


def login_required(f):
    """Decorator to require authentication for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


# -------------------------------------------------------------------
# Authentication Routes
# -------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login page and authentication."""
    ip = get_client_ip()
    error = None
    remaining_attempts = get_remaining_attempts(ip)
    
    # Check if locked out
    if is_locked_out(ip):
        remaining_time = get_lockout_remaining(ip)
        minutes = remaining_time // 60
        seconds = remaining_time % 60
        return render_template('login.html', 
                             error=f"Too many failed attempts. Access blocked for {minutes}m {seconds}s.",
                             locked_out=True,
                             remaining_time=remaining_time), 429
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Validate credentials
        if username == AUTH_USERNAME and check_password_hash(AUTH_PASSWORD_HASH, password):
            # Successful login
            reset_attempts(ip)
            session['authenticated'] = True
            session['username'] = username
            session.permanent = True  # Use permanent session
            
            # Redirect to next page or home
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            # Failed login
            locked = record_failed_attempt(ip)
            remaining_attempts = get_remaining_attempts(ip)
            
            if locked:
                remaining_time = get_lockout_remaining(ip)
                minutes = remaining_time // 60
                seconds = remaining_time % 60
                return render_template('login.html',
                                     error=f"Too many failed attempts. Access blocked for {minutes}m {seconds}s.",
                                     locked_out=True,
                                     remaining_time=remaining_time), 429
            else:
                error = f"Invalid username or password. {remaining_attempts} attempt(s) remaining."
    
    return render_template('login.html', 
                         error=error, 
                         remaining_attempts=remaining_attempts,
                         locked_out=False)


@app.route('/logout')
def logout():
    """Handle logout."""
    session.clear()
    return redirect(url_for('login'))


#
# PROCEDURE GROUPS (by ICD-10-PCS prefix)
#
PROCEDURE_GROUPS = {
    # Coronary artery bypass graft (CABG)
    "CABG": {
        "label": "Coronary artery bypass graft (CABG)",
        "prefixes": ["0210", "0211", "0212", "0213"],
    },

    # Coronary stenting / PCI
    "PCI_STENTING": {
        "label": "Coronary stenting / PCI",
        "prefixes": ["0270", "0271", "0272", "0273"],
    },

    # Knee replacement (right + left, including surfaces)
    "KNEE_REPLACEMENT": {
        "label": "Knee replacement (TKA / partial)",
        "prefixes": ["0SRC", "0SRD"],
    },

    # Hip replacement
    "HIP_REPLACEMENT": {
        "label": "Hip replacement",
        "prefixes": ["0SR9", "0SRB", "0SRA", "0SRR", "0SRS"],
    },

    # Hip fracture surgery
    "HIP_FRACTURE": {
        "label": "Hip fracture fixation / ORIF",
        "prefixes": ["0QS6", "0QS7", "0SS9", "0SSB"],
    },

    # Varicose vein surgery – lower limb veins
    "VARICOSE_VEINS": {
        "label": "Varicose vein surgery (lower extremity)",
        "prefixes": ["06BP", "06BQ", "06BT", "06BV", "06BY"],
    },

    # Spinal fusion
    "SPINAL_FUSION": {
        "label": "Spinal fusion (lumbar / lumbosacral)",
        "prefixes": ["0SG0", "0SG1", "0SG3", "0SG5", "0SG7"],
    },

    # Laminectomy / discectomy WITHOUT fusion
    "LAMINECTOMY_DISCECTOMY": {
        "label": "Laminectomy / discectomy (no fusion)",
        "prefixes": ["0SB0", "0SB1", "0SB2"],
    },

    # Cholecystectomy
    "CHOLECYSTECTOMY": {
        "label": "Cholecystectomy (gallbladder removal)",
        "prefixes": ["0FB4", "0FT4"],
    },

    # Colorectal resection
    "COLORECTAL_RESECTION": {
        "label": "Colorectal resection (colectomy, etc.)",
        "prefixes": [
            "0DTE",
            "0DTF",
            "0DTG",
            "0DTH",
            "0DTK",
            "0DTL",
            "0DTM",
            "0DTN",
        ],
    },

    # Cataract surgery
    "CATARACT": {
        "label": "Cataract surgery (lens replacement)",
        "prefixes": ["08RJ", "08RK"],
    },

    # Hernia repair
    "HERNIA_REPAIR": {
        "label": "Hernia repair (inguinal / abdominal wall)",
        "prefixes": ["0YQ", "0WQF"],
    },

    # Colonoscopy
    "COLONOSCOPY": {
        "label": "Colonoscopy",
        "prefixes": ["0DJD8"],
    },

    # Upper GI endoscopy
    "UPPER_GI_ENDOSCOPY": {
        "label": "Upper GI endoscopy (EGD)",
        "prefixes": ["0DJ08"],
    },

    # Incontinence surgery
    "INCONTINENCE_SURGERY": {
        "label": "Incontinence surgery (sling / bladder neck)",
        "prefixes": ["0TSC", "0TUC"],
    },
}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _safe_float(row, col):
    """Return float(row[col]) or None if missing/NaN."""
    if col not in row or pd.isna(row[col]):
        return None
    try:
        return float(row[col])
    except Exception:
        return None





def volume_category(cases: float | int | None) -> str:
    if cases is None:
        return "Unknown"
    try:
        c = float(cases)
    except Exception:
        return "Unknown"

    if c >= 200:
        return "High"
    if c >= 50:
        return "Medium"
    if c > 0:
        return "Low"
    return "Unknown"


def complication_tier(rate: float | None) -> str:
    """
    Simple tiering: lower % is better.
    Using any/overall complication rate from your aggregated table.
    """
    if rate is None or pd.isna(rate):
        return "Unknown"
    r = float(rate)
    if r < 5:
        return "Low estimated complications"
    if r < 15:
        return "Average estimated complications"
    return "Higher estimated complications"


def get_city_col(df: pd.DataFrame) -> str | None:
    for c in ["hospital_city", "city"]:
        if c in df.columns:
            return c
    return None


def get_state_col(df: pd.DataFrame) -> str | None:
    for c in ["hospital_state", "state"]:
        if c in df.columns:
            return c
    return None


def get_zip_col(df: pd.DataFrame) -> str | None:
    for c in ["hospital_zip", "postal_code", "zip", "ZIP"]:
        if c in df.columns:
            return c
    return None


def stringify_zip(s):
    if pd.isna(s):
        return None
    return str(s).strip()[:5] or None


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Vectorized Haversine distance in kilometers between arrays of (lat1, lon1)
    and scalar (lat2, lon2).
    """
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0  # Earth radius in km
    return r * c


def weighted_mean(series: pd.Series, weights: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = (~s.isna()) & (~w.isna()) & (w > 0)
    if not mask.any():
        return None
    s = s[mask]
    w = w[mask]
    return float(np.average(s, weights=w))


# -------------------------------------------------------------------
# Load data once at startup
# -------------------------------------------------------------------
if not HOSPITAL_CSV.exists():
    raise FileNotFoundError(f"Missing {HOSPITAL_CSV}")
if not PROVIDER_CSV.exists():
    raise FileNotFoundError(f"Missing {PROVIDER_CSV}")

hospital_df = pd.read_csv(
    HOSPITAL_CSV,
    dtype={"ORG_NPI_NUM": str},
    low_memory=False,
)

provider_df = pd.read_csv(
    PROVIDER_CSV,
    dtype={"ORG_NPI_NUM": str, "OP_PHYSN_NPI": str},
    low_memory=False,
)

# Normalize NPI formats
hospital_df["ORG_NPI_NUM"] = hospital_df["ORG_NPI_NUM"].astype(str).str.zfill(10)
provider_df["ORG_NPI_NUM"] = provider_df["ORG_NPI_NUM"].astype(str).str.zfill(10)
provider_df["OP_PHYSN_NPI"] = provider_df["OP_PHYSN_NPI"].astype(str).str.zfill(10)

# Normalize ZIP column if present
zip_col_h = get_zip_col(hospital_df)
if zip_col_h:
    hospital_df[zip_col_h] = hospital_df[zip_col_h].apply(stringify_zip)

# Ensure lat/lng columns exist and drop rows with missing coords
if "lat" not in hospital_df.columns or "lng" not in hospital_df.columns:
    raise ValueError(
        "hospital_metrics_geocoded.csv must contain 'lat' and 'lng' columns."
    )
hospital_df = hospital_df.dropna(subset=["lat", "lng"])

# -------------------------------------------------------------------
# Routes (Protected)
# -------------------------------------------------------------------
@app.route("/")
@login_required
def index():
    # expects templates/index.html
    return render_template("index.html")


@app.route("/methodology")
@login_required
def methodology():
    return render_template("methodology.html")


@app.route("/api/procedure-groups")
@login_required
def api_procedure_groups():
    """
    Return the available procedure groups for the autocomplete.
    """
    groups = []
    for key, meta in PROCEDURE_GROUPS.items():
        groups.append(
            {
                "key": key,
                "label": meta["label"],
                "prefixes": meta["prefixes"],
            }
        )
    # sort by label for nicer display
    groups.sort(key=lambda g: g["label"])
    return jsonify({"groups": groups})


@app.route("/api/search")
@login_required
def api_search():
    """
    Query params:
      - lat: latitude (required)
      - lng: longitude (required)
      - radius_km: radius in kilometers (default 50)
      - group: optional procedure group key from PROCEDURE_GROUPS

    IMPORTANT:
    * Nearby hospitals are determined FIRST by distance, BEFORE any procedure-group filtering.
    * Procedure-group filtering only affects which procedures/metrics are included in the stats,
      not which hospitals appear.
    * If the hospital has < 11 Medicare cases in this scope, its detailed panel will be suppressed
      on the frontend and metrics are nulled server-side for safety.
    * Surgeons' caseloads are only exposed if they have > 10 Medicare cases at that hospital
      (within this scope) to avoid low-volume privacy issues.
    """
    lat_raw = request.args.get("lat")
    lng_raw = request.args.get("lng")
    radius_raw = request.args.get("radius_km", "50")
    group_raw = (request.args.get("group", "") or "").strip().upper()

    if not lat_raw or not lng_raw:
        return jsonify({"error": "Missing lat/lng parameters."}), 400

    try:
        origin_lat = float(lat_raw)
        origin_lng = float(lng_raw)
    except ValueError:
        return jsonify({"error": "Invalid lat/lng parameters."}), 400

    try:
        radius_km = float(radius_raw)
        if radius_km <= 0:
            radius_km = 50.0
    except ValueError:
        radius_km = 50.0

    df_all = hospital_df.copy()

    # ------------------------------------------------------------------
    # Distance filtering FIRST (as requested)
    # ------------------------------------------------------------------
    df_all["distance_km"] = haversine_km(
        df_all["lat"], df_all["lng"], origin_lat, origin_lng
    )
    df_radius = df_all[df_all["distance_km"] <= radius_km].copy()

    if df_radius.empty:
        return jsonify(
            {
                "hospitals": [],
                "total": 0,
                "search_scope": "group" if group_raw else "all",
                "procedure_group_key": group_raw or None,
                "procedure_group_label": PROCEDURE_GROUPS.get(group_raw, {}).get(
                    "label"
                )
                if group_raw in PROCEDURE_GROUPS
                else None,
                "radius_km": radius_km,
            }
        )

    # ------------------------------------------------------------------
    # Procedure group metadata (optional)
    # ------------------------------------------------------------------
    procedure_group_key = None
    procedure_group_label = None
    prefixes = None

    if group_raw and group_raw in PROCEDURE_GROUPS:
        procedure_group_key = group_raw
        procedure_group_label = PROCEDURE_GROUPS[group_raw]["label"]
        prefixes = PROCEDURE_GROUPS[group_raw]["prefixes"]

        if "OPERATION_PCS" not in df_radius.columns:
            return (
                jsonify(
                    {"error": "Hospital metrics file missing OPERATION_PCS column."}
                ),
                500,
            )

    # ------------------------------------------------------------------
    # Ensure metric columns exist
    # ------------------------------------------------------------------
    for metric in [
        "cases",
        "avg_cost",
        "median_cost",
        "readmit_30_rate",
        "complication_rate",
        "mortality_rate",
        "avg_LOS",
        "median_LOS",
        "avg_cost_pctile_rank",
        "avg_LOS_pctile_rank",
        "readmit_30_rate_pctile_rank",
        "complication_rate_pctile_rank",
        "mortality_rate_pctile_rank",
        "short_title",
        "OPERATION_PCS",
    ]:
        if metric not in df_radius.columns:
            df_radius[metric] = pd.NA

    city_col = get_city_col(df_radius)
    state_col = get_state_col(df_radius)
    zip_col = get_zip_col(df_radius)

    # Unique hospitals within radius (for metadata)
    hospital_npis = df_radius["ORG_NPI_NUM"].unique().tolist()

    # Provider subset restricted to hospitals in radius
    provider_sub = provider_df[provider_df["ORG_NPI_NUM"].isin(hospital_npis)].copy()
    if procedure_group_key and prefixes:
        if "OPERATION_PCS" not in provider_sub.columns:
            provider_sub["OPERATION_PCS"] = ""
        op_p = provider_sub["OPERATION_PCS"].astype(str).str.upper()
        mask_p = pd.Series(False, index=provider_sub.index)
        for p in prefixes:
            mask_p |= op_p.str.startswith(p)
        provider_sub = provider_sub[mask_p].copy()

    hospitals = []

    # ------------------------------------------------------------------
    # Build per-hospital aggregates manually so that
    # the hospital LIST is purely radius-driven, while
    # metrics respect the procedure group (if any).
    # ------------------------------------------------------------------
    for org_npi in hospital_npis:
        org_npi_str = str(org_npi).zfill(10)
        hosp_rows_all = df_radius[df_radius["ORG_NPI_NUM"] == org_npi_str].copy()

        if hosp_rows_all.empty:
            continue

        # Scope rows: either all procedures, or those in the procedure group
        hosp_scope = hosp_rows_all.copy()
        if procedure_group_key and prefixes:
            op_series = hosp_scope["OPERATION_PCS"].astype(str).str.upper()
            mask = pd.Series(False, index=hosp_scope.index)
            for p in prefixes:
                mask |= op_series.str.startswith(p)
            hosp_scope = hosp_scope[mask].copy()

        # Aggregate metrics for this hospital in the current scope
        if hosp_scope.empty:
            total_cases = 0.0
            cases_series = pd.Series([], dtype=float)
            avg_cost = median_cost = readmit_30_rate = complication_rate = None
            mortality_rate_val = avg_los = median_los = None
            avg_cost_pctile_rank = avg_los_pctile_rank = None
            readmit_30_rate_pctile_rank = complication_rate_pctile_rank = None
            mortality_rate_pctile_rank = None
        else:
            cases_series = pd.to_numeric(
                hosp_scope["cases"], errors="coerce"
            ).fillna(0.0)
            total_cases = float(cases_series.sum())
            avg_cost = weighted_mean(hosp_scope["avg_cost"], cases_series)
            median_cost = weighted_mean(hosp_scope["median_cost"], cases_series)
            readmit_30_rate = weighted_mean(
                hosp_scope["readmit_30_rate"], cases_series
            )
            complication_rate = weighted_mean(
                hosp_scope["complication_rate"], cases_series
            )
            mortality_rate_val = weighted_mean(
                hosp_scope["mortality_rate"], cases_series
            )
            avg_los = weighted_mean(hosp_scope["avg_LOS"], cases_series)
            median_los = weighted_mean(hosp_scope["median_LOS"], cases_series)

            avg_cost_pctile_rank = weighted_mean(
                hosp_scope["avg_cost_pctile_rank"], cases_series
            ) or weighted_mean(
                hosp_scope["avg_cost_pctile_rank"],
                pd.Series([1] * len(hosp_scope)),
            )
            avg_los_pctile_rank = weighted_mean(
                hosp_scope["avg_LOS_pctile_rank"], cases_series
            ) or weighted_mean(
                hosp_scope["avg_LOS_pctile_rank"],
                pd.Series([1] * len(hosp_scope)),
            )
            readmit_30_rate_pctile_rank = weighted_mean(
                hosp_scope["readmit_30_rate_pctile_rank"], cases_series
            ) or weighted_mean(
                hosp_scope["readmit_30_rate_pctile_rank"],
                pd.Series([1] * len(hosp_scope)),
            )
            complication_rate_pctile_rank = weighted_mean(
                hosp_scope["complication_rate_pctile_rank"], cases_series
            ) or weighted_mean(
                hosp_scope["complication_rate_pctile_rank"],
                pd.Series([1] * len(hosp_scope)),
            )
            mortality_rate_pctile_rank = weighted_mean(
                hosp_scope["mortality_rate_pctile_rank"], cases_series
            ) or weighted_mean(
                hosp_scope["mortality_rate_pctile_rank"],
                pd.Series([1] * len(hosp_scope)),
            )

        # Low volume suppression: < 11 (or unknown/0) → hide panel, null out metrics
        cases_value = total_cases if not np.isnan(total_cases) else None
        low_volume_suppressed = (cases_value is None) or (cases_value < 11)

        if low_volume_suppressed:
            # Preserve total_cases for the front-end so it can decide to mask,
            # but zero out all detailed metrics for privacy.
            avg_cost = median_cost = None
            readmit_30_rate = complication_rate = None
            mortality_rate_val = avg_los = median_los = None
            avg_cost_pctile_rank = avg_los_pctile_rank = None
            readmit_30_rate_pctile_rank = complication_rate_pctile_rank = None
            mortality_rate_pctile_rank = None

        # Basic metadata from first row (any procedure)
        first_row = hosp_rows_all.iloc[0]
        city_val = first_row.get(city_col) if city_col else first_row.get(
            "hospital_city"
        )
        state_val = first_row.get(state_col) if state_col else first_row.get(
            "hospital_state"
        )
        zip_val = (
            stringify_zip(first_row.get(zip_col))
            if zip_col
            else stringify_zip(first_row.get("hospital_zip"))
        )
        full_address = first_row.get("full_address")
        distance_km_val = _safe_float(first_row, "distance_km")

        # Top procedures in-scope (limit 5)
        hosp_scope_for_procs = hosp_scope.copy()
        hosp_scope_for_procs["cases_num"] = pd.to_numeric(
            hosp_scope_for_procs["cases"], errors="coerce"
        ).fillna(0.0)
        top_procedures = []
        if not hosp_scope_for_procs.empty:
            top_proc_rows = hosp_scope_for_procs.sort_values(
                "cases_num", ascending=False
            ).head(5)
            for _, pr in top_proc_rows.iterrows():
                top_procedures.append(
                    {
                        "operation_pcs": pr.get("OPERATION_PCS"),
                        "short_title": pr.get("short_title"),
                        "cases": _safe_float(pr, "cases"),
                        "avg_cost": _safe_float(pr, "avg_cost"),
                        "complication_rate": _safe_float(pr, "complication_rate"),
                        "readmit_30_rate": _safe_float(pr, "readmit_30_rate"),
                        "mortality_rate": _safe_float(pr, "mortality_rate"),
                    }
                )

        # Surgeons at this hospital (within procedure scope if any)
        docs = provider_sub[provider_sub["ORG_NPI_NUM"] == org_npi_str].copy()
        doctors = []
        if not docs.empty:
            # Filter to surgeons with > 10 cases at this hospital (in this scope)
            docs["cases_num"] = pd.to_numeric(
                docs["cases"], errors="coerce"
            ).fillna(0.0)
            docs_filtered = docs[docs["cases_num"] > 10].copy()

            for _, d in docs_filtered.iterrows():
                numeric_cases = _safe_float(d, "cases")
                if numeric_cases is None:
                    continue

                doctors.append(
                    {
                        "npi": d.get("OP_PHYSN_NPI"),
                        "name": (
                            ""
                            if d.get("provider_full_name")
                            != d.get("provider_full_name")
                            else d.get("provider_full_name")
                        ),
                        "credential": d.get("provider_credential")
                        if not pd.isna(d.get("provider_credential"))
                        else None,
                        "cases": int(numeric_cases),
                        "avg_cost": _safe_float(d, "avg_cost"),
                        "median_cost": _safe_float(d, "median_cost"),
                        "readmit_30_rate": _safe_float(d, "readmit_30_rate"),
                        "complication_rate": _safe_float(d, "complication_rate"),
                        "mortality_rate": _safe_float(d, "mortality_rate"),
                        "avg_LOS": _safe_float(d, "avg_LOS"),
                        "median_LOS": _safe_float(d, "median_LOS"),
                        "readmit_30_rate_pctile_rank": _safe_float(
                            d, "readmit_30_rate_pctile_rank"
                        ),
                        "complication_rate_pctile_rank": _safe_float(
                            d, "complication_rate_pctile_rank"
                        ),
                        "mortality_rate_pctile_rank": _safe_float(
                            d, "mortality_rate_pctile_rank"
                        ),
                    }
                )

        volume_cat = volume_category(cases_value)
        comp_tier = (
            complication_tier(complication_rate)
            if not low_volume_suppressed
            else "Suppressed (low volume)"
        )

        hospitals.append(
            {
                "org_npi_num": org_npi_str,
                "hospital_name": first_row.get("hospital_name"),
                "full_address": full_address,
                "city": city_val,
                "state": state_val,
                "postal_code": zip_val,
                "lat": _safe_float(first_row, "lat"),
                "lng": _safe_float(first_row, "lng"),
                "distance_km": distance_km_val,
                "cases": cases_value,
                "avg_cost": avg_cost,
                "median_cost": median_cost,
                "readmit_30_rate": readmit_30_rate,
                "complication_rate": complication_rate,
                "mortality_rate": mortality_rate_val,
                "avg_LOS": avg_los,
                "median_LOS": median_los,
                "avg_cost_pctile_rank": avg_cost_pctile_rank,
                "avg_LOS_pctile_rank": avg_los_pctile_rank,
                "readmit_30_rate_pctile_rank": readmit_30_rate_pctile_rank,
                "complication_rate_pctile_rank": complication_rate_pctile_rank,
                "mortality_rate_pctile_rank": mortality_rate_pctile_rank,
                "volume_category": volume_cat,
                "complication_tier": comp_tier,
                "low_volume_suppressed": low_volume_suppressed,
                "doctors": doctors,
                "top_procedures": top_procedures,
                "procedure_group_key": procedure_group_key,
                "procedure_group_label": procedure_group_label,
            }
        )

    # Sort by quality first (complications), then mortality, then distance
    hospitals.sort(
        key=lambda h: (
            h["complication_rate"] if h["complication_rate"] is not None else 1e9,
            h["mortality_rate"] if h["mortality_rate"] is not None else 1e9,
            h["distance_km"] if h["distance_km"] is not None else 1e9,
        )
    )

    return jsonify(
        {
            "hospitals": hospitals,
            "total": len(hospitals),
            "search_scope": "group" if procedure_group_key else "all",
            "procedure_group_key": procedure_group_key,
            "procedure_group_label": procedure_group_label,
            "radius_km": radius_km,
        }
    )


if __name__ == "__main__":
    # Dev only – use gunicorn or similar in production
    app.run(debug=True, host="0.0.0.0", port=5001)