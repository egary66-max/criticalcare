# Deploying the Quiz App for the Intern Cohort

This is the one-time setup to put the ER Quiz on **Streamlit Community Cloud** so your cohort can study 24/7.

## What gets shared

- **Only Quiz Mode.** No Q&A, no Specialist Consultation, no record upload.
- Behind a single shared cohort password.
- Same RAG library as your local app — generated cases stay grounded in the textbooks.

## What stays local on your Mac

- The full app (`3_clinical_agent.py` via `./start_app.sh`) including Q&A, Consultation, Case Builder.
- The 6.4 GB `Books/` folder, the `Obsidian/` vault, the importer scripts.

## Cost ballpark

- Each quiz case ≈ **$0.05–0.15** on Claude Sonnet 4.6.
- 15 interns × 10 cases/week ≈ **$8–25/week** off your Anthropic card.
- **Hard cap it** in the Anthropic console (Settings → Limits → Monthly budget). Start at **$30/month** and adjust.

---

## One-time setup

### 1. Initialize the git repo

```bash
cd "/Users/erikgary/Documents/Agentic Workflows/Critical Care Flow"
git init
git add .
git status   # sanity-check — no Books/, no .env, no Obsidian/
git commit -m "Initial commit"
```

The `.gitignore` already excludes the big stuff. Total repo size should be **~760 MB** (mostly `chroma_db/`).

### 2. Create a PRIVATE GitHub repo

In your browser: github.com → New repository → **set to Private** → don't add a README/license (we already have files). Then:

```bash
git remote add origin git@github.com:<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

> If the push fails with "file too large", check `git ls-files | xargs ls -la | sort -k5 -n | tail -10` — any 100 MB+ files break GitHub's hard limit. The Chroma sqlite files should each be under that, but if they aren't, use Git LFS.

### 3. Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** → sign in with GitHub.
2. **New app** → pick your private repo → set:
   - **Main file path:** `4_quiz_app.py`
   - **Branch:** `main`
3. Click **Advanced settings** → **Secrets**, paste:

   ```toml
   APP_PASSWORD = "pick-a-password-here"
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

4. Click **Deploy**. First build takes ~5 minutes (ChromaDB is bulky).

### 4. Set the spend cap (DO THIS)

Anthropic console → **Settings → Limits** → set a monthly budget. This is your real cost backstop.

### 5. Test before sharing

Open the Streamlit URL in an incognito window. Enter the password. Run one full case through MC + FITB + FR. Make sure the **Q&A** and **Consultation** tabs don't appear (they shouldn't — `4_quiz_app.py` doesn't load them).

### 6. Share with the cohort

Post the URL + password in your intern Slack/group chat.

---

## Day-to-day

### Update the library or the quiz prompts

```bash
# edit code, then
git add -A
git commit -m "Tweak quiz prompts"
git push
```

Streamlit Cloud auto-rebuilds on push.

### Rotate the password

Streamlit Cloud → your app → **Settings → Secrets** → edit `APP_PASSWORD` → save. The app restarts automatically.

### Check usage / cost

- Anthropic console → **Usage**. If it's tracking hot, drop the monthly cap.
- If anyone is being abusive, ask them to chill or rotate the password to a fresh cohort.

### Add a new textbook to the cohort library

Locally only:

```bash
python3 add_book.py "path/to/book.epub" "Short Book Name"
# then push the updated chroma_db
git add chroma_db
git commit -m "Add: Short Book Name"
git push
```

---

## Local development

Run the cohort app locally without breaking your main app (which runs on 8501):

```bash
python3 -m streamlit run 4_quiz_app.py --server.port 8502
```

It will read `APP_PASSWORD` and `ANTHROPIC_API_KEY` from your `.env` for local testing.

## Troubleshooting

- **"Server misconfigured: ANTHROPIC_API_KEY is not set"** — paste it in Streamlit Cloud's Secrets editor.
- **App boots but Chroma errors** — check the deploy logs for the chroma_db path. The folder must be in the repo root.
- **Free tier app sleeps** — Streamlit Community Cloud puts inactive apps to sleep. First wake-up after a quiet stretch takes ~30 seconds.
- **Repo > 1 GB** — Streamlit Cloud's free tier has a soft limit. Trim with Git LFS or reduce the Chroma collection size by re-importing fewer books.
