#!/bin/sh
# Render's (and most free-tier container hosts') disk is ephemeral --
# every redeploy/restart starts from a blank filesystem, so the Chroma
# index and sessions.db (staff accounts, demo sessions) get rebuilt here
# on boot rather than being treated as persistent state. This is the SAME
# rebuild-from-source-of-truth pattern embed_and_store.py already does
# locally (delete-then-recreate the Chroma collection every run, see its
# own comment on why) -- boot time just makes it happen automatically
# instead of requiring a manual re-run.
#
# The ML models themselves are NOT downloaded here -- see the Dockerfile's
# pre-download RUN step, baked into the image at build time instead, so a
# container restart/wake-up doesn't re-download ~300MB every time.
#
# Deliberately does NOT run seed_jira_tickets.py: that's a one-off script
# that creates NEW tickets in a real Jira project via the live API --
# re-running it on every container boot would spam duplicate tickets into
# the real Jira instance. Tickets already exist there from when it was
# run once; the app fetches them live via jira_client.py, nothing to
# rebuild locally for that part.
set -e

echo "Building Chroma vector index from data/chunks/ (ephemeral disk, rebuilt every boot; models already cached from build)..."
python src/embed_and_store.py

echo "Seeding staff accounts (idempotent upsert, safe to re-run)..."
python src/seed_staff_accounts.py

echo "Starting server..."
exec python ui/server.py
