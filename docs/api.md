# API Reference

The backend is a **FastAPI** application (`src/probabilistic_medseg/api.py`) that serves live segmentation predictions along with per-pixel uncertainty heatmaps.

- **Base URL (local):** `http://localhost:8000`
- **Base URL (production):** deployed on Render — see [Deployment](deployment.md)
- **Interactive docs:** once running, visit `/docs` (Swagger UI) or `/redoc` on the deployed instance for a live, testable schema

---

## `POST /predict`

Runs inference on an uploaded skin lesion image and returns the predicted segmentation mask together with aleatoric and epistemic uncertainty heatmaps.

### Request

`multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file (image) | Yes | Dermoscopic skin lesion image (e.g. JPEG/PNG) to segment |

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@lesion_example.jpg"
```

### Response

`200 OK` — `application/json`

The response returns the predicted mask along with both uncertainty heatmaps. Depending on implementation, images may be returned as base64-encoded PNGs.

```json
{
  "mask": "base64-encoded PNG of the predicted segmentation mask",
  "aleatoric_uncertainty": "base64-encoded PNG heatmap (Probabilistic U-Net latent variance)",
  "epistemic_uncertainty": "base64-encoded PNG heatmap (MC Dropout predictive variance)",
  "metadata": {
    "model": "probabilistic_unet | mc_dropout",
    "num_samples": 30
  }
}
```

!!! note
    Confirm the exact response field names and encoding (base64 vs. raw array vs. file URLs) against your current `api.py` implementation — update this schema to match if it has since evolved.

### Error responses

| Status | Meaning |
|---|---|
| `400 Bad Request` | Uploaded file is missing, empty, or not a valid image |
| `422 Unprocessable Entity` | Request does not match the expected schema (e.g. wrong field name) |
| `500 Internal Server Error` | Model failed to load or inference error |

---

## Health Check

If exposed, a simple health/status endpoint is recommended for Render's health checks:

```
GET /health -> {"status": "ok"}
```

!!! note
    Add this to `api.py` if not already present — Render's free tier periodically pings a health endpoint, and having one avoids cold-start/timeout issues.

---

## Next

- [Deployment](deployment.md) — how this API is hosted on Render and wired to the GitHub Pages frontend
