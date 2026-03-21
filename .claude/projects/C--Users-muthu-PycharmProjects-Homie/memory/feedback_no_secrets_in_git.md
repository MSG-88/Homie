---
name: no-secrets-in-git
description: Never commit secrets to git — keep them in local .env files only
type: feedback
---

Never commit secrets (API keys, tokens, passwords) to git or push them to GitHub.

**Why:** GitHub push protection blocked a push due to a HuggingFace token in homie.config.yaml that was committed in earlier history. Secrets in git history are permanent and cause push failures.

**How to apply:**
- Store all secrets in `.env` files (already in `.gitignore`)
- Use `os.environ.get()` or `python-dotenv` to load them
- Before committing, check for secrets in config files
- Never commit `homie.config.yaml` with real API keys — use placeholder values or env var references
