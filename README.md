# Meeting Notes & Transcription Platform

This is a meeting-assistant app where you can browse a
library of past meetings, open one to read an **interactive transcript** that's synced to a media player,
get **AI-generated** summaries / action items / topics, search across everything, and manage meetings
end-to-end.

Real speech-to-text was out of scope for this assignment, so I focused on recreating the *experience* and
the post-meeting workflows. Transcripts are either seeded, pasted, or uploaded, and the summaries are
generated with **Groq**. I also added a deterministic mock fallback, so the app still works even if there's
no API key set.

> **Live demo**
> - **App (Vercel):** https://frontend-nine-sage-spasdcz85x.vercel.app
> - **API + Swagger docs (Render):** https://fireflies-clone-2g5d.onrender.com/docs
>
> ⏱️ One heads-up: the backend is on Render's free tier, which sleeps after ~15 min of inactivity. The very
> first request after it wakes up can take ~50 seconds, so give it a moment on first load.

---

## What you can do

**The core stuff:**
- **Browse meetings** — a dashboard of cards with the title, date, duration, participants, and tags. You can
  search, filter by date, and sort by recency or title.
- **Read the transcript interactively** — every line has a speaker and a timestamp. Click a line and the
  player jumps to that moment; hit play and the active line highlights and auto-scrolls along with it. There's
  also a search box to find and jump between matches inside the transcript.
- **Play it back** — a media player with a seek bar and play/pause. Since audio was out of scope, this is a
  simulated timeline driven by the transcript timestamps (the spec allows a placeholder here).
- **AI notes** — each meeting gets a summary, extracted action items, and key topics generated from the
  transcript. The topics are clickable and seek the player to that point.
- **Full CRUD** — create a meeting by pasting or uploading a transcript (`.txt` / `.vtt` / `.json`), edit its
  details, delete it, and add / edit / complete / delete action items. Everything persists.

**Extras I added on top:**
- **Ask about this meeting** — a chat widget on each meeting page that answers questions grounded in that
  meeting's transcript, and says "I don't know" when the answer isn't in it. It reuses the Groq key
  server-side only (the browser never sees it) and is ephemeral — nothing is stored.
- A global search page across every meeting (title, participants, summary, and transcript).
- Export a meeting's transcript + summary to **Markdown** or **plain text**.
- **Dark mode** (remembered between visits), a **collapsible sidebar** (a focus-mode toggle on desktop and a
  slide-in drawer on mobile), toasts on every action, and an account menu.
- "Coming soon" placeholders for the live bot, integrations, team/sharing, and real auth (I assume a single
  default logged-in user).

---

## What I used (and why)

I set this up as two apps in one repo:

- **Frontend** — Next.js 16 (App Router) with TypeScript and Tailwind CSS v4. I used TanStack Query for data
  fetching, lucide-react for icons, and sonner for toasts. The app is very interactive, so I built it as a
  client-rendered SPA.
- **Backend** — FastAPI with SQLModel (on SQLite) and Pydantic v2. I picked FastAPI because it gave me clean,
  typed endpoints and auto-generated Swagger docs with basically no extra work.
- **AI** — Groq (`llama-3.3-70b-versatile`). I ask it to return structured JSON so it maps straight into my
  database, and I wrapped the call so that a missing key or a failed request falls back to a local mock. That
  way, creating a meeting never breaks because of the AI step.

---

## How it fits together

The frontend is a client-rendered SPA that talks to the FastAPI backend over a plain REST API. The backend
owns the database and makes the Groq calls.

```
┌──────────────────────┐        HTTPS / REST        ┌──────────────────────────┐
│  frontend/ (Next.js) │  ───────────────────────▶  │  backend/ (FastAPI)      │
│  • App Router SPA    │   NEXT_PUBLIC_API_URL      │  • REST API (/api/*)     │
│  • React Query       │  ◀───────────────────────  │  • SQLModel ↔ SQLite     │
│  • Tailwind v4       │        JSON                │  • Groq service + mock   │
└──────────────────────┘                            └──────────┬───────────────┘
                                                               │
                                                        ┌──────▼──────┐   ┌──────────┐
                                                        │  SQLite DB  │   │ Groq API │
                                                        └─────────────┘   └──────────┘
```

Repo layout:
```
├── frontend/            # Next.js app (app/, components/, lib/)
├── backend/             # FastAPI app (app/: models, schemas, crud, routers/, services/, seed)
├── render.yaml          # Render blueprint for the backend
└── README.md
```

---

## How I modeled the data

I used SQLite through SQLModel. Everything hangs off the `meetings` table, and all the child rows
cascade-delete with their meeting so I never leave orphans behind.

```
meetings ──1:*── participants
   │      ──1:*── transcript_segments   (speaker, start_time, end_time, text, order_index)
   │      ──1:1── summaries             (overview, generated_by: seeded | groq | mock)
   │      ──1:*── action_items          (text, assignee?, completed, due_date?, order_index)
   │      ──1:*── topics                (title, start_time?  → clickable chapter)
   └──────*:*── tags                    (through a meeting_tags link table)
```

| Table | Key columns |
|---|---|
| `meetings` | id, title, description, meeting_date, duration_seconds, media_url, created_at, updated_at |
| `participants` | id, meeting_id →, name, email?, role? |
| `transcript_segments` | id, meeting_id →, speaker, start_time, end_time, text, order_index |
| `summaries` | id, meeting_id → (unique), overview, generated_by, created_at, updated_at |
| `action_items` | id, meeting_id →, text, assignee?, completed, due_date?, order_index, created_at |
| `topics` | id, meeting_id →, title, start_time?, order_index |
| `tags` / `meeting_tags` | id, name (unique) / (meeting_id, tag_id) |

I keep a `generated_by` flag on each summary (`seeded`, `groq`, or `mock`) so the UI can show where it came
from, and every transcript segment carries `start_time` / `end_time` — that's what powers the click-to-seek
and the highlight-follows-playback behavior.

---

## The API

Everything lives under `/api`, and FastAPI serves interactive docs at **`/docs`** if you want to try the
endpoints directly.

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/meetings` | List meetings; supports `search`, `participant`, `meeting_date`, `sort` |
| POST | `/meetings` | Create from `transcript_text` or `segments`; kicks off the AI summary |
| POST | `/meetings/upload` | Create from an uploaded transcript file (multipart) |
| GET | `/meetings/{id}` | Full meeting detail |
| PATCH | `/meetings/{id}` | Edit title / description / participants / tags |
| DELETE | `/meetings/{id}` | Delete a meeting |
| POST | `/meetings/{id}/generate-summary` | Regenerate the summary / action items / topics |
| POST | `/meetings/{id}/chat` | Ask a transcript-grounded question about the meeting |
| POST | `/meetings/{id}/action-items` | Add an action item |
| PATCH | `/action-items/{id}` | Edit / complete an action item |
| DELETE | `/action-items/{id}` | Delete an action item |
| GET | `/search?q=` | Global search across all meetings |
| GET | `/meetings/{id}/export?format=md\|txt` | Export the transcript + summary |
| GET | `/api/health` | Health check |

---

## Running it locally

You'll need **Node.js 20.9+** and **Python 3.11+**.

### 1) Backend (`backend/`)
```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional: drop in your GROQ_API_KEY for real AI summaries
python -m app.seed                 # seeds 5 sample meetings
uvicorn app.main:app --reload --port 8000
```
The API runs at http://localhost:8000, and the docs are at http://localhost:8000/docs. The database also
auto-seeds itself if it's empty on startup, so the app is never blank.

### 2) Frontend (`frontend/`)
```bash
cd frontend
npm install
npm run dev                        # http://localhost:3000
```
By default the frontend points at `http://localhost:8000/api`. If your backend is somewhere else, set
`NEXT_PUBLIC_API_URL` in `frontend/.env.local`.

---

## Environment variables

**Backend** (`backend/.env`)
| Variable | What it's for | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq key for real summaries. Leave blank to use the mock fallback. | *(empty)* |
| `GROQ_MODEL` | Which Groq model to use | `llama-3.3-70b-versatile` |
| `FRONTEND_URL` | Allowed CORS origin (my Vercel URL in production) | `http://localhost:3000` |
| `DATABASE_URL` | SQLAlchemy database URL | `sqlite:///./fireflies.db` |

**Frontend** (`frontend/.env.local`)
| Variable | What it's for | Default |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Backend base URL, including `/api` | `http://localhost:8000/api` |

I keep all secrets in local `.env` files (which are gitignored) or in the hosting dashboard — nothing
sensitive is committed.

---

## How I deployed it

I put the **backend on Render** and the **frontend on Vercel**.

**Backend → Render**
1. New Web Service from this repo, with **Root Directory `backend`** (there's a `render.yaml` blueprint too).
2. Build command `pip install -r requirements.txt`, start command
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3. Set `GROQ_API_KEY` (and `FRONTEND_URL` = the Vercel URL) in the dashboard.

**Frontend → Vercel**
1. Import the repo with **Root Directory `frontend`**.
2. Add `NEXT_PUBLIC_API_URL = https://<render-app>.onrender.com/api`.

A couple of things worth knowing about the deployment:
- CORS already allows any `*.vercel.app` URL (plus whatever `FRONTEND_URL` is set to).
- Render's free disk is ephemeral, so SQLite resets on a redeploy or cold start. To handle that, the app
  **re-seeds the 5 sample meetings automatically** whenever the database is empty, so the demo is never blank.
  For durable user data you'd point `DATABASE_URL` at Postgres instead.

---

## Notes & assumptions

- **Single default user** — I didn't build real auth; there's a placeholder account menu instead.
- **No real audio** — the player is a simulated timeline driven by the transcript timestamps.
- **Groq for the AI** — summaries fall back to a local mock if the key is missing or the call fails, so
  creating a meeting never breaks.
- If you paste a transcript **without timestamps**, I synthesize reasonable ones so the player still works.
