"""
main.py — FastAPI application for the CAT Modeling Data Pipeline.

Endpoints:
  POST   /upload                    Upload CSV/XLSX, create session
  GET    /suggest-columns/{id}      Get column mapping suggestions
  POST   /confirm-columns/{id}      Confirm column mapping
  POST   /geocode/{id}              Run geocoding
  POST   /map-codes/{id}            Run 4-stage code mapping
  POST   /normalize/{id}            Run normalization
  GET    /review/{id}               Get accumulated flags
  POST   /correct/{id}              Apply corrections
  GET    /download/{id}             Download output XLSX or CSV
  GET    /session/{id}              Session info
  GET    /sessions                  List all active sessions (debug)
  DELETE /session/{id}              Delete session
"""
import io
import json
import logging
import os
import pathlib
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ── Load env early ─────────────────────────────────────────────────────────────
load_dotenv()

# ── Local imports ──────────────────────────────────────────────────────────────
import session as session_store
import code_mapper
import geocoder
from column_mapper import suggest_columns, validate_required_fields
from normalizer import normalize_all_rows
from output_builder import build_xlsx, build_csv
from rules import BusinessRulesConfig
from models import (
    UploadResponse, SuggestColumnsResponse, ColumnSuggestion,
    ConfirmColumnsRequest, ConfirmColumnsResponse,
    GeocodeResponse, MapCodesResponse, NormalizeResponse,
    ReviewResponse, FlagEntry, CorrectRequest, CorrectResponse,
    SessionInfoResponse,
)

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("main")

# ── Lifespan: pre-build TF-IDF and start TTL cleanup ──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CAT pipeline — loading TF-IDF indexes…")
    code_mapper.build_tfidf_indexes()
    geocoder.load_reference_data()
    session_store.start_ttl_cleanup()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CAT Modeling Data Pipeline",
    description="Upload, map, geocode, classify, and normalise property exposure data for AIR/RMS CAT models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_session_or_404(upload_id: str) -> dict:
    try:
        return session_store.require_session(upload_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{upload_id}' not found or expired.")


def _require_stage(session: dict, stage: str, endpoint: str) -> None:
    if not session.get("stages_complete", {}).get(stage):
        raise HTTPException(
            status_code=422,
            detail=f"Stage '{stage}' must be completed before calling '{endpoint}'.",
        )


def _load_iso4217() -> set:
    p = pathlib.Path(__file__).parent / "reference" / "iso4217_currency.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(data.keys())
    return set()


_VALID_CURRENCIES = _load_iso4217()


# ── Step 1: Upload ─────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, tags=["Pipeline"])
async def upload(
    file: UploadFile = File(...),
    target_format: str = Query("AIR", regex="^(AIR|RMS)$"),
    rules_config: Optional[str] = Form(None),
):
    """
    Upload a CSV or XLSX file and create a processing session.
    No row limit — all rows are ingested.
    """
    content = await file.read()
    fname = (file.filename or "").lower()

    # Parse rules config
    try:
        rules_dict = json.loads(rules_config) if rules_config and rules_config.strip() else {}
        rules = BusinessRulesConfig(**rules_dict)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid rules_config: {exc}")

    # Parse file
    try:
        if fname.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, encoding="latin-1")
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content), dtype=str, keep_default_na=False, sheet_name=0)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload CSV or XLSX.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"File parsing error: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File contains no data rows.")

    # Deduplicate column names
    seen: Dict[str, int] = {}
    new_cols = []
    for col in df.columns:
        col = str(col).strip()
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            new_cols.append(col)
    df.columns = new_cols

    # Clean values: strip whitespace, replace empty string with None
    df = df.map(lambda v: v.strip() if isinstance(v, str) and v.strip() != "" else None)

    headers = list(df.columns)
    raw_rows = df.to_dict(orient="records")
    sample = raw_rows[:5]

    upload_id = session_store.create_session({
        "target_format": target_format,
        "rules_config": rules.model_dump(),
        "raw_rows": raw_rows,
        "headers": headers,
        "sample": sample,
        "column_map": {},
        "unmapped_cols": [],
        "geo_rows": [],
        "code_map": {},
        "final_rows": [],
    })

    logger.info(f"Session {upload_id}: uploaded {len(raw_rows)} rows, format={target_format}")
    return UploadResponse(
        upload_id=upload_id,
        row_count=len(raw_rows),
        headers=headers,
        sample=sample,
        target_format=target_format,
    )


# ── Step 2a: Suggest columns ───────────────────────────────────────────────────

@app.get("/suggest-columns/{upload_id}", response_model=SuggestColumnsResponse, tags=["Pipeline"])
def suggest_columns_endpoint(upload_id: str):
    """Stateless: analyse column names and return ranked mapping suggestions."""
    session = _get_session_or_404(upload_id)
    raw_rows = session["raw_rows"]
    headers = session["headers"]

    # Build sample values dict (first 3 non-null per column)
    sample_values: Dict[str, List[Any]] = {}
    for col in headers:
        vals = [r[col] for r in raw_rows[:20] if r.get(col) is not None][:3]
        sample_values[col] = vals

    result = suggest_columns(
        source_columns=headers,
        sample_values=sample_values,
        target_format=session["target_format"],
        fuzzy_threshold=session["rules_config"].get("fuzzy_llm_fallback_threshold", 72),
        cutoff=session["rules_config"].get("fuzzy_score_cutoff", 50),
    )

    suggestions_typed = {
        col: [ColumnSuggestion(**s) for s in sug_list]
        for col, sug_list in result["suggestions"].items()
    }
    return SuggestColumnsResponse(
        suggestions=suggestions_typed,
        unmapped=result["unmapped"],
    )


# ── Step 2b: Confirm columns ───────────────────────────────────────────────────

@app.post("/confirm-columns/{upload_id}", response_model=ConfirmColumnsResponse, tags=["Pipeline"])
def confirm_columns(upload_id: str, body: ConfirmColumnsRequest):
    """Confirm and persist the column mapping. Validates required fields and 1:1 uniqueness."""
    session = _get_session_or_404(upload_id)

    column_map = body.column_map

    # ── Enforce 1:1 mapping: one source → one canonical, one canonical ← one source ──
    # Collect which source columns map to each canonical (ignoring None/unmapped).
    canonical_to_sources: Dict[str, List[str]] = {}
    for src_col, canonical in column_map.items():
        if canonical is None:
            continue
        canonical_to_sources.setdefault(canonical, []).append(src_col)

    # Find violations: any canonical claimed by more than one source column
    duplicate_violations = {
        canonical: sources
        for canonical, sources in canonical_to_sources.items()
        if len(sources) > 1
    }

    if duplicate_violations:
        details = "; ".join(
            f"'{canonical}' claimed by: {sources}"
            for canonical, sources in duplicate_violations.items()
        )
        raise HTTPException(
            status_code=422,
            detail=f"Mapping violation — each canonical field must be mapped by exactly one source column. "
                   f"Duplicates found: {details}",
        )

    unmapped = [k for k, v in column_map.items() if v is None]
    mapped_count = len(column_map) - len(unmapped)

    missing_required = validate_required_fields(column_map, session["target_format"])
    warnings = [f"Required field '{f}' is not mapped." for f in missing_required]

    session_store.update_session(upload_id, {
        "column_map": column_map,
        "unmapped_cols": unmapped,
    })
    session_store.session_mark_stage(upload_id, "column_map")

    logger.info(f"Session {upload_id}: column map confirmed, {mapped_count} mapped, {len(unmapped)} unmapped")
    return ConfirmColumnsResponse(
        upload_id=upload_id,
        mapped_count=mapped_count,
        unmapped_cols=unmapped,
        missing_required=missing_required,
        warnings=warnings,
    )


# ── Step 3: Geocode ────────────────────────────────────────────────────────────

@app.post("/geocode/{upload_id}", response_model=GeocodeResponse, tags=["Pipeline"])
def geocode_endpoint(upload_id: str):
    """Geocode all rows using Geoapify. Skips rows with pre-existing coordinates."""
    session = _get_session_or_404(upload_id)
    _require_stage(session, "column_map", "/geocode")

    column_map = session["column_map"]
    rules_config = BusinessRulesConfig(**session["rules_config"])

    # Remap raw rows using confirmed column_map
    raw_rows = session["raw_rows"]
    remapped = _apply_column_map(raw_rows, column_map)

    geocoded_count = provided_count = failed_count = 0
    new_flags: List[dict] = []
    geo_rows: List[dict] = []

    for idx, row in enumerate(remapped):
        geo_fields = geocoder.process_row_geocoding(
            row, column_map,
            target_format=session.get("target_format", "AIR"),
        )
        row.update(geo_fields)
        geo_rows.append(row)

        status = geo_fields.get("GeocodingStatus", "")
        source = geo_fields.get("Geosource", "")

        if source == "Provided":
            provided_count += 1
        elif status == "OK":
            geocoded_count += 1
            # State code validation flag
            if geo_fields.get("StateCodeValidation") == "UNRECOGNIZED":
                new_flags.append({
                    "row_index": idx,
                    "field": "Statecode_Final",
                    "issue": "unrecognized_state_code",
                    "current_value": geo_fields.get("Statecode_Final"),
                    "confidence": None,
                    "alternatives": [],
                    "message": f"State code '{geo_fields.get('Statecode_Final')}' not found in ISO 3166 reference",
                })
        else:
            failed_count += 1
            new_flags.append({
                "row_index": idx,
                "field": "GeocodingStatus",
                "issue": "geocoding_failed",
                "current_value": status,
                "confidence": None,
                "alternatives": [],
                "message": f"Geocoding failed for row {idx}: {status}",
            })

    session_store.update_session(upload_id, {"geo_rows": geo_rows})
    session_store.append_flags(upload_id, new_flags)
    session_store.session_mark_stage(upload_id, "geocoding")

    logger.info(f"Session {upload_id}: geocoded={geocoded_count}, provided={provided_count}, failed={failed_count}")
    return GeocodeResponse(
        upload_id=upload_id,
        total_rows=len(raw_rows),
        geocoded=geocoded_count,
        provided=provided_count,
        failed=failed_count,
        flags_added=len(new_flags),
    )


# ── Step 4: Map codes ──────────────────────────────────────────────────────────

@app.post("/map-codes/{upload_id}", response_model=MapCodesResponse, tags=["Pipeline"])
def map_codes_endpoint(upload_id: str):
    """
    Run 4-stage code mapping (deterministic → Gemini LLM → TF-IDF → default)
    for both occupancy and construction codes.
    """
    session = _get_session_or_404(upload_id)
    _require_stage(session, "geocoding", "/map-codes")

    geo_rows = session["geo_rows"]
    column_map = session["column_map"]
    rules_config = BusinessRulesConfig(**session["rules_config"])
    target = session["target_format"]

    # Identify schema columns
    occ_scheme_col, occ_value_col = _find_code_columns(column_map, "occupancy", target)
    const_scheme_col, const_value_col = _find_code_columns(column_map, "construction", target)

    target_occ_scheme = "OccupancyCodeType" if target == "AIR" else "OCCSCHEME"
    target_const_scheme = "ConstructionCodeType" if target == "AIR" else "BLDGSCHEME"

    new_flags: List[dict] = []
    enriched_rows = []
    code_map: Dict[str, Dict] = {}

    # ── Occupancy mapping ────────────────────────────────────────────────────
    if occ_value_col:
        occ_items = code_mapper.extract_unique_pairs(geo_rows, occ_scheme_col, occ_value_col)
        occ_results = code_mapper.map_codes(occ_items, target, "occupancy", rules_config)
        for item in occ_items:
            key = code_mapper.build_row_key(item["scheme"], item["value"])
            result = occ_results.get(str(item["index"]), {})
            if result:
                code_map[f"occ|{key}"] = result

    # ── Construction mapping ─────────────────────────────────────────────────
    if const_value_col:
        const_items = code_mapper.extract_unique_pairs(geo_rows, const_scheme_col, const_value_col)
        const_results = code_mapper.map_codes(const_items, target, "construction", rules_config)
        for item in const_items:
            key = code_mapper.build_row_key(item["scheme"], item["value"])
            result = const_results.get(str(item["index"]), {})
            if result:
                code_map[f"const|{key}"] = result

    # ── Enrich geo_rows with code results ────────────────────────────────────
    occ_by_method: Dict[str, int] = {}
    const_by_method: Dict[str, int] = {}

    enriched_rows = []
    for idx, row in enumerate(geo_rows):
        row = dict(row)

        # Apply occupancy
        if occ_value_col:
            scheme = str(row.get(occ_scheme_col) or "").strip()
            value = str(row.get(occ_value_col) or "").strip()
            key = f"occ|{code_mapper.build_row_key(scheme, value)}"
            result = code_map.get(key, {})
            if result:
                # ── Write mapped integer code to the canonical column the output builder reads ──
                row[occ_value_col] = result["code"]      # e.g. row["OccupancyCode"] = 302
                # Write scheme label only if not already set by the source data
                if not row.get(target_occ_scheme):
                    row[target_occ_scheme] = target
                # Preserve metadata in internal audit fields
                row["Occupancy_Code"]        = result["code"]
                row["Occupancy_Description"] = result["description"]
                row["Occupancy_Confidence"]  = result["confidence"]
                row["Occupancy_Method"]      = result["method"]
                row["Occupancy_Original"]    = result["original"]
                occ_by_method[result["method"]] = occ_by_method.get(result["method"], 0) + 1
                if result["confidence"] < rules_config.occ_confidence_threshold:
                    new_flags.append({
                        "row_index": idx, "field": occ_value_col,
                        "issue": "low_confidence",
                        "current_value": result["code"],
                        "confidence": result["confidence"],
                        "alternatives": result.get("alternatives", []),
                        "message": (f"Occupancy '{value}' mapped to {result['code']} "
                                    f"({result['description']}) with low confidence {result['confidence']:.2f}"),
                    })

        # Apply construction
        if const_value_col:
            scheme = str(row.get(const_scheme_col) or "").strip()
            value = str(row.get(const_value_col) or "").strip()
            key = f"const|{code_mapper.build_row_key(scheme, value)}"
            result = code_map.get(key, {})
            if result:
                # ── Write mapped integer code to the canonical column the output builder reads ──
                row[const_value_col] = result["code"]    # e.g. row["ConstructionCode"] = 215
                # Write scheme label only if not already set by the source data
                if not row.get(target_const_scheme):
                    row[target_const_scheme] = target
                # Preserve metadata in internal audit fields
                row["Construction_Code"]        = result["code"]
                row["Construction_Description"] = result["description"]
                row["Construction_Confidence"]  = result["confidence"]
                row["Construction_Method"]      = result["method"]
                row["Construction_Original"]    = result["original"]
                const_by_method[result["method"]] = const_by_method.get(result["method"], 0) + 1
                if result["confidence"] < rules_config.const_confidence_threshold:
                    new_flags.append({
                        "row_index": idx, "field": const_value_col,
                        "issue": "low_confidence",
                        "current_value": result["code"],
                        "confidence": result["confidence"],
                        "alternatives": result.get("alternatives", []),
                        "message": (f"Construction '{value}' mapped to {result['code']} "
                                    f"({result['description']}) with low confidence {result['confidence']:.2f}"),
                    })

        enriched_rows.append(row)

    session_store.update_session(upload_id, {
        "code_map": code_map,
        "final_rows": enriched_rows,
    })
    session_store.append_flags(upload_id, new_flags)
    session_store.session_mark_stage(upload_id, "code_mapping")

    logger.info(f"Session {upload_id}: code mapping complete. "
                f"occ_methods={occ_by_method}, const_methods={const_by_method}")

    return MapCodesResponse(
        upload_id=upload_id,
        unique_occ_pairs=len([k for k in code_map if k.startswith("occ|")]),
        unique_const_pairs=len([k for k in code_map if k.startswith("const|")]),
        occ_by_method=occ_by_method,
        const_by_method=const_by_method,
        flags_added=len(new_flags),
    )


# ── Step 5: Normalize ──────────────────────────────────────────────────────────

@app.post("/normalize/{upload_id}", response_model=NormalizeResponse, tags=["Pipeline"])
def normalize_endpoint(upload_id: str):
    """Run all normalization (year, stories, area, values, currency, modifiers)."""
    session = _get_session_or_404(upload_id)
    _require_stage(session, "code_mapping", "/normalize")

    rules_config = BusinessRulesConfig(**session["rules_config"])
    final_rows = session["final_rows"] or session["geo_rows"]

    # Reload valid currencies
    valid_currencies = _VALID_CURRENCIES

    target_format = session.get("target_format", "AIR")
    normalized, new_flags = normalize_all_rows(final_rows, rules_config, valid_currencies,
                                                target_format=target_format)

    # Apply global line of business if provided
    lob_col = "LineOfBusiness" if target_format == "AIR" else ""
    if lob_col and rules_config.line_of_business:
        for r in normalized:
            if not r.get(lob_col):
                r[lob_col] = rules_config.line_of_business

    session_store.update_session(upload_id, {"final_rows": normalized})
    session_store.append_flags(upload_id, new_flags)
    session_store.session_mark_stage(upload_id, "normalization")

    # Summarise what was done
    summary = {
        "year_flags": sum(1 for f in new_flags if "year" in f.get("issue", "")),
        "story_flags": sum(1 for f in new_flags if "story" in f.get("issue", "") or "stories" in f.get("issue", "")),
        "area_flags": sum(1 for f in new_flags if "area" in f.get("issue", "")),
        "value_flags": sum(1 for f in new_flags if "value" in f.get("issue", "")),
        "currency_flags": sum(1 for f in new_flags if "currency" in f.get("issue", "")),
    }

    logger.info(f"Session {upload_id}: normalization complete, {len(new_flags)} new flags")
    return NormalizeResponse(
        upload_id=upload_id,
        total_rows=len(normalized),
        flags_added=len(new_flags),
        normalization_summary=summary,
    )


# ── Review & Corrections ───────────────────────────────────────────────────────

@app.get("/review/{upload_id}", response_model=ReviewResponse, tags=["Review"])
def review(upload_id: str):
    """Return all accumulated flags for the session."""
    session = _get_session_or_404(upload_id)
    flags = [FlagEntry(**f) for f in session.get("flags", [])]
    return ReviewResponse(
        upload_id=upload_id,
        flags=flags,
        stages_complete=session.get("stages_complete", {}),
    )


@app.post("/correct/{upload_id}", response_model=CorrectResponse, tags=["Review"])
def correct(upload_id: str, body: CorrectRequest):
    """
    Apply manual corrections to final_rows.
    Each correction:
      • Updates final_rows[row_index][field]
      • Removes the matching flag
      • If occupancy/construction code, updates code_map for that key
    """
    session = _get_session_or_404(upload_id)
    final_rows = session.get("final_rows", [])
    code_map = session.get("code_map", {})
    rules_config = BusinessRulesConfig(**session["rules_config"])
    applied = 0
    flags_removed = 0

    for correction in body.corrections:
        idx = correction.row_index
        field = correction.field
        value = correction.new_value

        if idx < 0 or idx >= len(final_rows):
            continue

        final_rows[idx][field] = value

        # If this is a code override, update code_map so other rows with same source benefit
        if field in ("Occupancy_Code", "Construction_Code"):
            prefix = "occ|" if field == "Occupancy_Code" else "const|"
            scheme = final_rows[idx].get("OccupancyScheme" if field == "Occupancy_Code" else "ConstructionScheme", "")
            original = final_rows[idx].get(
                "Occupancy_Original" if field == "Occupancy_Code" else "Construction_Original", "")
            key = f"{prefix}{code_mapper.build_row_key(scheme, original)}"
            if key in code_map:
                code_map[key]["code"] = value
                code_map[key]["method"] = "user_override"
                code_map[key]["confidence"] = 1.0
                # Propagate to all matching rows
                for r in final_rows:
                    r_orig = r.get("Occupancy_Original" if field == "Occupancy_Code" else "Construction_Original", "")
                    r_scheme = r.get("OccupancyScheme" if field == "Occupancy_Code" else "ConstructionScheme", "")
                    if r_orig == original and r_scheme == scheme:
                        r[field] = value
                        r[field.replace("_Code", "_Method")] = "user_override"
                        r[field.replace("_Code", "_Confidence")] = 1.0

        # Remove flag
        session_store.remove_flag(upload_id, idx, field)
        flags_removed += 1
        applied += 1

    session_store.update_session(upload_id, {
        "final_rows": final_rows,
        "code_map": code_map,
    })

    return CorrectResponse(applied=applied, flags_removed=flags_removed)


# ── Download ───────────────────────────────────────────────────────────────────

@app.get("/download/{upload_id}", tags=["Output"])
def download(
    upload_id: str,
    format: str = Query("xlsx", regex="^(xlsx|csv)$"),
):
    """Download the processed output as XLSX or CSV."""
    session = _get_session_or_404(upload_id)
    _require_stage(session, "normalization", "/download")

    final_rows = session.get("final_rows", [])
    unmapped_cols = session.get("unmapped_cols", [])
    flags = session.get("flags", [])
    target = session.get("target_format", "AIR")
    short_id = upload_id[:8]

    if format == "csv":
        buf = build_csv(final_rows, unmapped_cols, target)
        filename = f"cat_output_{short_id}.csv"
        media_type = "text/csv"
    else:
        buf = build_xlsx(final_rows, unmapped_cols, flags, target, upload_id)
        filename = f"cat_output_{short_id}.xlsx"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    logger.info(f"Session {upload_id}: download requested ({format}), {len(final_rows)} rows")
    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Session management ─────────────────────────────────────────────────────────

@app.get("/session/{upload_id}", response_model=SessionInfoResponse, tags=["Session"])
def session_info(upload_id: str):
    session = _get_session_or_404(upload_id)
    return SessionInfoResponse(
        upload_id=upload_id,
        created_at=session["created_at"],
        target_format=session.get("target_format", "AIR"),
        row_count=len(session.get("raw_rows", [])),
        stages_complete=session.get("stages_complete", {}),
        flag_count=len(session.get("flags", [])),
        rules_config=session.get("rules_config", {}),
    )


@app.get("/sessions", tags=["Session"])
def list_sessions():
    return {"sessions": session_store.list_sessions()}


@app.delete("/session/{upload_id}", tags=["Session"])
def delete_session(upload_id: str):
    if session_store.delete_session(upload_id):
        return {"deleted": upload_id}
    raise HTTPException(status_code=404, detail="Session not found.")


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _apply_column_map(raw_rows: List[Dict], column_map: Dict[str, Optional[str]]) -> List[Dict]:
    """
    Return a new list of row dicts where keys are canonical field names.
    Enforces 1:1 at apply-time: if two source columns map to the same canonical,
    the first one encountered wins and the second is skipped with a warning.
    Source columns mapped to None are dropped (not passed through).
    """
    # Build a deduplicated ordered mapping: canonical → first src col that claimed it
    canonical_claimed_by: Dict[str, str] = {}
    for src_col, canonical in column_map.items():
        if canonical is None:
            continue
        if canonical not in canonical_claimed_by:
            canonical_claimed_by[canonical] = src_col
        else:
            logger.warning(
                f"_apply_column_map: canonical '{canonical}' already claimed by "
                f"'{canonical_claimed_by[canonical]}'; skipping duplicate source '{src_col}'."
            )

    result = []
    for row in raw_rows:
        new_row: Dict[str, Any] = {}
        for canonical, src_col in canonical_claimed_by.items():
            new_row[canonical] = row.get(src_col)
        result.append(new_row)
    return result


def _find_code_columns(column_map: Dict, field_type: str, target: str):
    """
    Identify scheme and value columns for occupancy or construction.
    Returns (scheme_col, value_col) as canonical names.
    """
    # AIR canonical names
    _AIR_OCC_VALUE   = ["OccupancyCode"]
    _AIR_OCC_SCHEME  = ["OccupancyCodeType"]
    _AIR_CONST_VALUE = ["ConstructionCode"]
    _AIR_CONST_SCHEME= ["ConstructionCodeType"]
    # RMS canonical names
    _RMS_OCC_VALUE   = ["OCCTYPE"]
    _RMS_OCC_SCHEME  = ["OCCSCHEME"]
    _RMS_CONST_VALUE = ["BLDGCLASS"]
    _RMS_CONST_SCHEME= ["BLDGSCHEME"]

    if field_type == "occupancy":
        scheme_options = _AIR_OCC_SCHEME  if target == "AIR" else _RMS_OCC_SCHEME
        value_options  = _AIR_OCC_VALUE   if target == "AIR" else _RMS_OCC_VALUE
    else:
        scheme_options = _AIR_CONST_SCHEME if target == "AIR" else _RMS_CONST_SCHEME
        value_options  = _AIR_CONST_VALUE  if target == "AIR" else _RMS_CONST_VALUE

    mapped_vals = set(v for v in column_map.values() if v)
    scheme_col = next((c for c in scheme_options if c in mapped_vals), "")
    value_col  = next((c for c in value_options  if c in mapped_vals), "")
    return scheme_col, value_col


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
