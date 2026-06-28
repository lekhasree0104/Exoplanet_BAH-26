# Exoplanet Analysis Agent

A real agentic AI app that fetches live TESS/Kepler/K2 data from NASA MAST and analyses it for exoplanet transits.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run
uvicorn main:app --reload --port 8000

# 4. Open http://localhost:8000
```

## What it does

The agent has 4 real tools:

| Tool | What it does |
|------|--------------|
| `search_tess_target` | Queries NASA MAST for available datasets |
| `fetch_light_curve` | Downloads & normalises the light curve, returns a plot |
| `detect_transits` | Runs BLS periodogram, returns best period + phase-fold plot |
| `get_star_info` | Fetches stellar parameters from TIC catalogue |

Claude autonomously decides which tools to call and in what order, then explains the results.

## Example queries

- "Analyse TRAPPIST-1 for transits"
- "Fetch the Kepler light curve for Kepler-452"
- "What's the transit period of TOI-700d?"
- "Search for any TESS data on Proxima Centauri"

## Architecture

```
Browser (HTML/JS)
     │  POST /chat  {messages: [...]}
     ▼
FastAPI  (main.py)
     │
     ▼
Agentic loop (up to 8 turns):
  Claude ──tool_use──► dispatch_tool()
    ▲                       │
    └──── tool_result ◄──── ▼
                      lightkurve → MAST
                      matplotlib → base64 PNG
     │
     ▼  stop_reason=end_turn
  {reply, plots[], tool_calls[]}
     │
     ▼
Browser renders chat + plots
```