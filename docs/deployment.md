# Deployment (Planned)

!!! note "Status"
    The project is not yet deployed. This page documents the **planned** hosting setup — a FastAPI backend on Render and a static frontend dashboard on GitHub Pages — as a roadmap for when deployment happens, not a description of a live system.

The project will ship as two separately deployed pieces: a **FastAPI backend** (model inference) on Render, and a **static frontend dashboard** on GitHub Pages.

---

## Backend — Render (Planned)

The backend will be deployed as a **Render Web Service** (Free tier).

### Start command

```bash
uvicorn src.probabilistic_medseg.api:app --host 0.0.0.0 --port $PORT
```

Render injects the `$PORT` environment variable automatically — do not hardcode a port.

### Setup steps

1. Push the repository to GitHub (already at [`Alisha026/probabilistic_medseg`](https://github.com/Alisha026/probabilistic_medseg))
2. In the Render dashboard, create a **New Web Service** and connect the GitHub repo
3. Configure the service:

    | Setting | Value |
    |---|---|
    | **Environment** | Python 3 |
    | **Build Command** | `pip install -r requirements.txt` |
    | **Start Command** | `uvicorn src.probabilistic_medseg.api:app --host 0.0.0.0 --port $PORT` |
    | **Plan** | Free |

4. Deploy — Render will build and expose a public URL of the form `https://<service-name>.onrender.com`

!!! warning "Interpretation"
    Render's free tier spins down idle services and cold-starts on the next request, which can take 30–60 seconds. If the GitHub Pages frontend calls the API directly, surface a loading state for that first request rather than letting it appear to hang or fail.

### Model weights

Ensure trained model weights (`.pt`/`.pth` files) are either committed to the repo (if small enough), pulled from external storage at build/start time, or included via Git LFS — Render's free tier has limited disk, so avoid bundling very large checkpoints directly if avoidable.

### CORS

Since the frontend (GitHub Pages) and backend (Render) live on different origins, make sure `api.py` has CORS configured to allow requests from `https://alisha026.github.io`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://alisha026.github.io"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

---

## Frontend — GitHub Pages (Planned)

The static interactive dashboard (`index.html`) will be served directly from GitHub Pages once built and published.

- **Planned URL:** `https://alisha026.github.io/probabilistic_medseg/`

### Setup steps

1. In the repository settings on GitHub, go to **Settings → Pages**
2. Set the source branch (e.g. `main`) and folder (`/` root or `/docs`, depending on where `index.html` lives)
3. GitHub will build and publish automatically on push to that branch
4. Update the dashboard's API call target to point at the deployed Render URL

---

## Deploying These Docs

This documentation site itself is built with **MkDocs Material** and can be published to GitHub Pages independently of the dashboard (e.g. to a `gh-pages` branch).

```bash
# 1. Install MkDocs Material
pip install mkdocs-material

# 2. Build the static site (outputs to site/)
mkdocs build

# 3. Preview locally at http://127.0.0.1:8000
mkdocs serve

# 4. Deploy to GitHub Pages (gh-pages branch)
mkdocs gh-deploy
```

!!! note
    `mkdocs gh-deploy` pushes to a `gh-pages` branch by default. If the same repository already uses GitHub Pages for the interactive dashboard from a different branch/folder, decide whether the docs site and dashboard should live on separate GitHub Pages projects/paths, or consolidate under one Pages source to avoid a conflict.

---

## Summary / Roadmap

| Component | Planned Host | Status |
|---|---|---|
| Backend API | Render (Free Web Service) | Not yet deployed |
| Interactive dashboard | GitHub Pages | Not yet deployed |
| This documentation | GitHub Pages (`gh-pages`, via `mkdocs gh-deploy`) | Not yet deployed |

**Next steps:** build the static dashboard (`index.html`) that consumes the `/predict` endpoint, confirm model weights are ready to ship with the backend, then deploy the API to Render and the dashboard to GitHub Pages following the steps above.
