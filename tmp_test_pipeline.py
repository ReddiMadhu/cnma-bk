import json
import logging
from pathlib import Path
from fastapi.testclient import TestClient

# Must import main after adding to path
import sys
sys.path.insert(0, ".")
from main import app, lifespan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_pipeline")

client = TestClient(app)

def run_test():
    filepath = Path("SOv_input_combined.xlsx")
    if not filepath.exists():
        logger.error(f"File {filepath} not found!")
        return

    # Trigger lifespan startup manually since TestClient doesn't always run it reliably
    # Actually TestClient(app) does run lifespan if used as context manager
    with TestClient(app) as client:
        # 1. Upload
        logger.info("--- 1. Upload ---")
        with open(filepath, "rb") as f:
            resp = client.post("/upload", params={"target_format": "AIR"}, files={"file": ("SOv_input_combined.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        
        if resp.status_code != 200:
            logger.error(f"Upload failed: {resp.status_code} {resp.text}")
            return
        
        data = resp.json()
        upload_id = data["upload_id"]
        logger.info(f"Upload success. ID: {upload_id}. Rows: {data['row_count']}")
        
        # 2a. Suggest
        logger.info("--- 2a. Suggest Columns ---")
        resp = client.get(f"/suggest-columns/{upload_id}")
        if resp.status_code != 200:
            logger.error(f"Suggest failed: {resp.status_code} {resp.text}")
            return
        
        suggestions = resp.json()["suggestions"]
        
        # Auto-pick the highest confidence for each column
        column_map = {}
        for src, sugs in suggestions.items():
            if sugs and sugs[0]["score"] >= 0.5:
                column_map[src] = sugs[0]["canonical"]
            else:
                column_map[src] = None
        
        logger.info(f"Selected Column Map: {json.dumps(column_map, indent=2)}")

        # 2b. Confirm
        logger.info("--- 2b. Confirm Columns ---")
        resp = client.post(f"/confirm-columns/{upload_id}", json={"column_map": column_map})
        if resp.status_code != 200:
            logger.error(f"Confirm failed: {resp.status_code} {resp.text}")
            return
        logger.info(f"Confirm success: {resp.json()}")

        # 3. Geocode
        logger.info("--- 3. Geocode ---")
        resp = client.post(f"/geocode/{upload_id}")
        if resp.status_code != 200:
            logger.error(f"Geocode failed: {resp.status_code} {resp.text}")
            return
        logger.info(f"Geocode success: {resp.json()}")

        # 4. Map Codes
        logger.info("--- 4. Map Codes ---")
        resp = client.post(f"/map-codes/{upload_id}")
        if resp.status_code != 200:
            logger.error(f"Map codes failed: {resp.status_code} {resp.text}")
            return
        logger.info(f"Map codes success: {resp.json()}")

        # 5. Normalize
        logger.info("--- 5. Normalize ---")
        resp = client.post(f"/normalize/{upload_id}")
        if resp.status_code != 200:
            logger.error(f"Normalize failed: {resp.status_code} {resp.text}")
            return
        logger.info(f"Normalize success: {resp.json()}")
        
        # 6. Review
        logger.info("--- 6. Review Flags ---")
        resp = client.get(f"/review/{upload_id}")
        if resp.status_code != 200:
            logger.error(f"Review failed: {resp.status_code} {resp.text}")
            return
        flags = resp.json()["flags"]
        logger.info(f"Found {len(flags)} flags. Sample: {json.dumps(flags[:2], indent=2) if flags else '[]'}")

        # 7. Download
        logger.info("--- 7. Download ---")
        resp = client.get(f"/download/{upload_id}", params={"format": "csv"})
        if resp.status_code != 200:
            logger.error(f"Download failed: {resp.status_code} {resp.text}")
            return
            
        with open("TEST_OUTPUT.csv", "wb") as f:
            f.write(resp.content)
        logger.info("Pipeline Complete. Saved output to TEST_OUTPUT.csv.")

if __name__ == "__main__":
    run_test()
