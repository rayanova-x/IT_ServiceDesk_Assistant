#!/bin/sh
# Render's (and most free-tier container hosts') disk is ephemeral --
# every redeploy/restart starts from a blank filesystem. sessions.db
# (staff accounts, demo sessions) gets reseeded here on every boot --
# cheap, instant, no real CPU cost.
#
# The Chroma vector index is deliberately NOT built here anymore
# (2026-10-25, real deploy failure): it's built once at Docker BUILD
# time instead (see the Dockerfile's own RUN step + its comment on why
# -- a real deploy showed embedding 299 chunks takes 20+ minutes on
# Render's free-tier 0.1 vCPU, far past the ~5 minute port-scan timeout
# a slow/silent BOOT gets killed by; a BUILD step has no such deadline).
# The ML models themselves are ALSO pre-downloaded at build time, same
# reasoning -- so this boot sequence does no heavy CPU/network work at
# all, just reseeding a tiny SQLite table and starting the server.
#
# Deliberately does NOT run seed_jira_tickets.py: that's a one-off script
# that creates NEW tickets in a real Jira project via the live API --
# re-running it on every container boot would spam duplicate tickets into
# the real Jira instance. Tickets already exist there from when it was
# run once; the app fetches them live via jira_client.py, nothing to
# rebuild locally for that part.
set -e

echo "Seeding staff accounts (idempotent upsert, safe to re-run)..."
python src/seed_staff_accounts.py

echo "Starting server..."
exec python ui/server.py
