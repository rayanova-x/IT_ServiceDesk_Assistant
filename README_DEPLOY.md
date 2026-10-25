# Deploying this project

## 1. Push to GitHub

```bash
git init   # already done if you're reading this in the repo
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo-name>.git
git push -u origin main
```

`.gitignore` already excludes `.venv/`, `.env`, `chroma_db/`, `local_model_cache/`,
and `sessions.db` — none of your secrets or local build artifacts get committed.
Before pushing, double check `git status` doesn't show `.env` as staged.

## 2. Deploy on Hugging Face Spaces (free, Docker runtime)

Vercel can't run this app — it's a stateful FastAPI server with a local embedding
model + reranker (~1.2GB download) and a persistent Chroma vector DB, none of
which fit Vercel's serverless model. Hugging Face Spaces' free Docker tier gives
**16GB RAM / 2 vCPU / 50GB disk**, which comfortably fits this.

1. Go to https://huggingface.co/new-space
2. Pick a name, set **SDK: Docker**, visibility as you prefer
3. Either:
   - Connect it directly to your GitHub repo (Settings → Repository → link GitHub), or
   - `git clone` the empty Space repo HF gives you and push this project's code to
     it directly (Spaces are themselves git repos)
4. Go to the Space's **Settings → Variables and secrets** and add these as
   **secrets** (never commit them — they're already gitignored via `.env`):
   - `GEMINI_API_KEY`
   - `GEMINI_MODEL` (e.g. `gemini-3.5-flash-lite`)
   - `GEMINI_JUDGE_MODEL`
   - `GEMINI_STRONG_JUDGE_MODEL`
   - `JIRA_SITE_URL`
   - `JIRA_EMAIL`
   - `JIRA_API_TOKEN`
   - `JIRA_PROJECT_KEY`
5. The Space will build the Dockerfile automatically and start. First boot takes
   a few minutes: it downloads the embedding/reranker models (~1.2GB) and
   rebuilds the Chroma index from `data/chunks/` — see `docker-entrypoint.sh`.

### Why the disk resets every time

HF Spaces' free tier disk is **ephemeral** — every redeploy/restart starts from
a blank filesystem. This project's `docker-entrypoint.sh` handles that by
rebuilding the Chroma index and reseeding staff accounts (`alice`/`bob`/`carol`/
`dave`/`admin`) on every boot, from the small source files already in `data/`
(a few MB, committed to git) — not from the local `chroma_db/`/`sessions.db`
files, which are gitignored and never shipped.

**What this means in practice:** every container restart gives every visitor
(including you) a genuinely fresh demo — no leftover sessions, no stale custom
eval cases from a prior boot. The golden set version history
(`golden_set_v1.jsonl`, `golden_set_v2.jsonl`, `golden_set_meta.json`) and
`eval_custom_cases.jsonl` in `data/processed/` ARE committed to git, so that
specific demonstration history survives across redeploys even though the
runtime DB/vector-index don't.

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
