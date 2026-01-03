# operator_table_api_async.py
import os, json, time, uuid, re
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header
from pydantic import BaseModel

from operator_table_service import build_rows_for_location, validate_rows

# --- Optional: S3 persistence (recommended) ---
USE_S3 = os.getenv("OP_TABLE_USE_S3", "1") == "1"
S3_BUCKET = os.getenv("OP_TABLE_S3_BUCKET", "")
S3_PREFIX = os.getenv("OP_TABLE_S3_PREFIX", "operator_tables/")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

s3 = None
if USE_S3:
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)

# --- Simple bearer auth token ---
API_TOKEN = os.getenv("OP_TABLE_API_TOKEN")  # set to a long random string
def auth_or_403(authorization: Optional[str]):
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(403, "Forbidden")

# --- Helpers ---
def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def out_filename(city: str, country: str) -> str:
    return f"operator_table_{norm_name(city)}_{norm_name(country)}.json"

# --- App & in-memory job store ---
app = FastAPI(title="Operator Table Service (Async)")
JOBS: Dict[str, Dict[str, Any]] = {}  # replace with Redis/Celery in prod if needed

class JobCreate(BaseModel):
    location: str                   # "City, Country"
    # when not using S3, you can also include out_dir for local persistence:
    out_dir: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

def _do_build(job_id: str, location: str, out_dir: Optional[str]):
    try:
        parts = [p.strip() for p in location.split(",") if p.strip()]
        if not parts or len(parts) < 2:
            raise ValueError("location must be 'City, Country'")
        city, country = parts[0], parts[-1]

        rows = build_rows_for_location(location)
        if not rows:
            rows = [{
                "Band ": "N/A", "3GPP Band": "N/A",
                "Uplink Frequency (MHz)": "0 - 0",
                "Downlink Frequency (MHz)": "0 - 0",
                "Bandwidth": "N/A",
                "Technology": "N/A",
                "Operators": f"No live/curated data for {location}. Please upload a table."
            }]

        validate_rows(rows)
        payload = json.dumps(rows, indent=2, ensure_ascii=False).encode("utf-8")

        fn = out_filename(city, country)
        saved_path = None
        public_url = None

        if USE_S3 and S3_BUCKET:
            key = f"{S3_PREFIX}{fn}"
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=payload, ContentType="application/json")
            saved_path = f"s3://{S3_BUCKET}/{key}"
            # Optional: presigned URL for quick debug/download
            public_url = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=3600)
        else:
            # Local write (e.g., EC2 instance storage)
            out_dir_p = Path(out_dir or ".")
            out_dir_p.mkdir(parents=True, exist_ok=True)
            fp = out_dir_p / fn
            fp.write_bytes(payload)
            saved_path = str(fp.resolve())
            public_url = None

        JOBS[job_id].update({"status": "done", "path": saved_path, "url": public_url, "rows": rows})
    except Exception as e:
        JOBS[job_id].update({"status": "error", "error": str(e)})

@app.post("/jobs")
def create_job(req: JobCreate, bg: BackgroundTasks, authorization: Optional[str] = Header(None)):
    auth_or_403(authorization)
    if not req.location or "," not in req.location:
        raise HTTPException(400, "location must be like 'City, Country'")
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "created_at": time.time()}
    bg.add_task(_do_build, job_id, req.location, req.out_dir)
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
def job_status(job_id: str, authorization: Optional[str] = Header(None)):
    auth_or_403(authorization)
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return {"job_id": job_id, "status": j["status"], "path": j.get("path"), "url": j.get("url"), "error": j.get("error")}

@app.get("/jobs/{job_id}/result")
def job_result(job_id: str, authorization: Optional[str] = Header(None)):
    auth_or_403(authorization)
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    if j["status"] != "done":
        raise HTTPException(409, "job not finished")
    return {"job_id": job_id, "path": j["path"], "url": j.get("url"), "rows": j["rows"]}
