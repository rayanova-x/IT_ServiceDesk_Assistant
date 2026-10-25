# Deploying this project

## 1. Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo-name>.git
git push -u origin main
```

`.gitignore` already excludes `.venv/`, `.env`, `chroma_db/`, `local_model_cache/`,
and `sessions.db` — none of your secrets or local build artifacts get committed.
Before pushing, double check `git status` doesn't show `.env` as staged.

## 2. Deploy on Render (free web service, Docker runtime)

Vercel can't run this app — it's a stateful FastAPI server with a local embedding
model + reranker, not a stateless/serverless frontend function. Render's free web
service tier (512MB RAM) is tight for a RAG pipeline, which is why this repo
deploys with a lighter reranker in production (see "Why the reranker is different
in production" below) — with that swap, the whole app fits comfortably.

1. Go to https://dashboard.render.com/ → **New → Web Service**
2. Connect your GitHub repo
3. Render will detect `render.yaml` automatically (runtime: docker, plan: free)
4. Fill in the secrets it prompts for (from `render.yaml`'s env var list):
   - `GEMINI_API_KEY`
   - `GEMINI_MODEL` (e.g. `gemini-3.5-flash-lite`)
   - `GEMINI_JUDGE_MODEL`
   - `GEMINI_STRONG_JUDGE_MODEL`
   - `JIRA_SITE_URL`
   - `JIRA_EMAIL`
   - `JIRA_API_TOKEN`
   - `JIRA_PROJECT_KEY`
   (same values as your local `.env` — never commit these)
5. Click **Create Web Service**. The Docker build step takes a few minutes: it
   downloads the embedding + reranker models AND builds the Chroma index from
   `data/chunks/` — both baked into the image at build time (see the
   Dockerfile), not at container boot. Boot itself is fast (just reseeds
   `sessions.db` and starts the server — see `docker-entrypoint.sh`).

**Free tier note:** Render's free web services spin down after 15 minutes of no
traffic and take ~30-60s to wake on the next request (cold start). Fine for a
demo link, not for guaranteed instant response at all times.

### Why the Chroma index and models are baked into the image, not rebuilt at boot

Render's free tier disk is **ephemeral** — every redeploy/restart starts from a
blank filesystem, so a boot-time rebuild sounded right at first. In practice,
embedding all 299 KB chunks took 20+ minutes on Render's free-tier 0.1 vCPU — far
past the ~5 minute window Render allows before killing an unresponsive boot
(confirmed by a real failed deploy, not a guess). A Docker **build** step has no
such timeout, so the Dockerfile now runs `embed_and_store.py` (and downloads both
ML models) once per `git push`, baked into the image — boot itself only reseeds
`sessions.db` (`alice`/`bob`/`carol`/`dave`/`admin`), which is instant.

**What this means in practice:** a KB content change (editing `data/chunks/*.jsonl`)
needs a new push to take effect, not just a restart — already true for a
git-based deploy anyway. Chat sessions and custom eval cases still reset on every
container restart (that part of the ephemeral-disk behavior is unchanged — see
`sessions.db` above), so every visitor still gets a fresh demo regardless of
persona. The golden set version history (`golden_set_v1.jsonl`,
`golden_set_v2.jsonl`, `golden_set_meta.json`) and `eval_custom_cases.jsonl` in
`data/processed/` are committed to git, so that specific demonstration history
survives across redeploys too.

### Why the reranker is different in production

This project's reranker (`BAAI/bge-reranker-base`) measured ~1.2GB RSS once
loaded — on its own, more than double Render's free-tier 512MB RAM cap, before
FastAPI/Chroma/the Gemini SDK are even counted. Rather than drop reranking
entirely (it's load-bearing: Step 8's confidence check and the whole retrieval
quality story depend on it) or violate the project's "local embeddings only, no
third-party API" constraint by moving to a hosted embedding service, the
Dockerfile sets `RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-6-v2` — a much smaller
cross-encoder (measured ~290MB RSS total with the embedding model, fits the
free tier with real headroom). It's still a local, fastembed/ONNX model — the
"no third-party API" constraint is untouched, only the specific model size.

**Known, explicitly accepted gap:** the deployed app's confidence scores are on
a different scale than `BAAI/bge-reranker-base`'s. The calibrated 2.0 confidence
threshold (`step8_confidence.py`) and every golden-set/retrieval-eval number in
this project's history were measured against the original, heavier model and
have NOT been re-run against the lighter one. Local development (`python
ui/server.py` without `RERANKER_MODEL` set) still uses the original model, so
all of that history stays accurate for local work — it's specifically the
public deployment that trades model fidelity for free hosting.

### Your own login

Sign in with `admin` / `demo4321` (or alice/bob/carol/dave / `demo1234`) via the
sign-in form, not a persona card — persona cards create new throwaway guest
accounts every click, by design (see `src/auth.py`'s `guest_login()`).

### What NOT to re-run on the server

`src/seed_jira_tickets.py` is a one-off script that creates real tickets in your
actual Jira project via the live API. It is deliberately NOT part of
`docker-entrypoint.sh` — re-running it on every boot would create duplicate
tickets in your real Jira project. Tickets already exist there from running it
once locally; the app fetches them live via `src/jira_client.py`.
