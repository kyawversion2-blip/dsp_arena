# DSP Arena multi-provider AI setup

DSP Arena's AI actions run through `live_server.py` endpoints `/api/dsp-ai/status` and `/api/dsp-ai/run`. Open the UI at `http://127.0.0.1:8765/arena` while the FastAPI server is running; opening `index.html` directly as a `file://` URL cannot reach the backend.

## API key configuration

Option A — use a private project `.env` file:

```sh
cp .env.example .env
chmod 600 .env
# Edit .env locally and add only the keys you are allowed to use.
uvicorn live_server:app --host 127.0.0.1 --port 8765
```

The gateway reads only the four known provider variables from `.env`, does not override variables already exported by the server process, and refuses a `.env` file that is readable by group/other users. `.env` is ignored by git.

Option B — export variables before starting the server:

```sh
export DEEPSEEK_API_KEY='...'
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
export GEMINI_API_KEY='...'
uvicorn live_server:app --host 127.0.0.1 --port 8765
```

Only set the providers you intend to use. Do not paste keys into the browser, CSV, source code, or chat. `.env.example` is a blank template; real `.env` files are ignored by git. If using a local `.env` loader in the future, ensure the file remains private and is never committed.

## Data privacy and request behavior

No dataset is sent to a provider at page load. A request is made only when the user clicks **Run AI**. At most 200 rows are accepted; the prompt uses bounded samples plus per-group summaries. Feature analysis and classification requests send DSP feature values and group labels where appropriate, not WAV/audio files. Do not send sensitive datasets to a provider unless permitted by the data owner and provider terms.

Classification is AI-assisted inference, not model training. For labeled datasets, one row per group (when enough examples exist) is held out with its group label removed from the prompt. The backend compares predictions to the held-out labels after the response and labels this as a small in-dataset check, not generalization accuracy. Results are suggestions and must be validated against DSP/statistical analysis.

## Supported provider IDs

- `deepseek`: `deepseek-chat`, `deepseek-reasoner`
- `openai`: `gpt-4.1-mini`, `gpt-4.1`, `o4-mini`
- `claude`: `claude-sonnet-4-5`, `claude-3-7-sonnet-latest`, `claude-3-5-haiku-latest`
- `gemini`: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`

Provider access, model availability, rate limits, and billing depend on each account. Errors are returned without exposing API keys or raw upstream response bodies.

## Gemini CLI with Google account login (no Gemini API key)

DSP Arena prefers the official Gemini CLI's cached Google login when it is available. Install the CLI on the same Mac that runs `live_server.py`, then sign in once:

```bash
npm install -g @google/gemini-cli
gemini
```

Choose **Sign in with Google** in the CLI browser flow and finish login. The dashboard uses the CLI's documented non-interactive mode for Feature Analysis and AI group suggestions; it does not ask for a Gemini API key. The server never reads or returns token contents to the browser. Keep the server local and do not copy `~/.gemini/oauth_creds.json` into the project or share it.

If the CLI is not installed or Google login is not cached, the dashboard reports Gemini as disconnected. A Gemini API key remains an optional fallback only; it is not required for the CLI login route.
