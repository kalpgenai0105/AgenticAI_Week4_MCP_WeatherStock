"""
weather_stock_server.py
=======================
A Model Context Protocol (MCP) server built with the official Python MCP SDK
(FastMCP style). It exposes exactly two tools to any MCP-compatible client:

  1. get_current_weather  – live conditions from the OpenWeather "Current Weather" API
  2. get_stock_data       – current price + OHLCV history from Yahoo Finance via yfinance

HOW IT FITS INTO THE SYSTEM
─────────────────────────────
                    ┌─────────────────────────────┐
                    │         agent.py             │
                    │  (LangChain / LangGraph)     │
                    │                              │
                    │  MultiServerMCPClient        │
                    │   spawns this file as a      │
                    │   subprocess over stdio      │
                    └───────────┬─────────────────┘
                  stdin/stdout  │  (JSON-RPC messages)
                    ┌───────────▼─────────────────┐
                    │   weather_stock_server.py    │  ← YOU ARE HERE
                    │   FastMCP server             │
                    │                              │
                    │  get_current_weather()  ─────┼──► OpenWeather REST API
                    │  get_stock_data()       ─────┼──► Yahoo Finance (yfinance)
                    └─────────────────────────────┘

TRANSPORT
─────────
FastMCP uses "stdio" transport by default: the client writes JSON-RPC request
messages to the server's stdin and reads JSON-RPC response messages from its
stdout. This means the server MUST NOT write anything other than MCP protocol
messages to stdout — use stderr for any debugging output you add.

STANDALONE TESTING (without agent.py)
──────────────────────────────────────
    python weather_stock_server.py
    # Then in a second terminal:
    npx @modelcontextprotocol/inspector python weather_stock_server.py
Open the printed URL (http://localhost:5173) and call tools from the browser UI.
This lets you confirm each tool works before wiring it to the agent.
"""

import os
import sys
from typing import Any

import requests                       # HTTP calls to OpenWeather REST API
import yfinance as yf                 # Yahoo Finance wrapper for stock data
from mcp.server.fastmcp import FastMCP  # High-level MCP server builder


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════
#
# FastMCP is a decorator-based abstraction over the lower-level mcp.Server.
# Providing a name and description helps MCP clients (and LLM agents) understand
# what this server is for before they inspect individual tools.
#
mcp = FastMCP("WeatherStock")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Keeping URLs and env-var names as constants means there is exactly one place
# to update them if the OpenWeather API version changes or you rename the secret.
#
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
# The "2.5/weather" endpoint returns the *current* conditions for one location.
# It is part of the free tier — no paid subscription required.

OPENWEATHER_KEY_VAR = "OPENWEATHER_API_KEY"
# The API key is NEVER hardcoded. It is read from the environment at call time
# so the same server binary can be used in different environments (dev / CI /
# prod) without code changes, and so the key is not accidentally committed to
# version control.


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1: WEATHER
# ═══════════════════════════════════════════════════════════════════════════════
#
# The @mcp.tool() decorator does three things:
#   1. Registers get_current_weather as an MCP tool the server advertises.
#   2. Derives the tool's JSON Schema from the Python type annotations.
#   3. Uses the docstring as the tool's human-readable description — this is
#      exactly what the LLM agent reads to decide *when* to call the tool, so
#      the docstring must be precise and written for the model, not just humans.
#
@mcp.tool()
def get_current_weather(city: str) -> dict[str, Any]:
    """Fetch current weather conditions for a given city.

    Returns temperature (°C and °F), weather description, humidity (%),
    wind speed (m/s), and the city/country the API matched against.

    Use this tool whenever the user asks about weather, temperature,
    humidity, wind, or current conditions for any location.

    Args:
        city: City name, e.g. "London" or "New York" or "Tokyo, JP".
              Appending a country code (ISO 3166-1 alpha-2) after a comma
              disambiguates cities that share a name.
    """

    # ── 1. Validate the API key is present before making any network call ──────
    #
    # We check here (at call time) rather than at startup so that the server
    # can start successfully even without the key, and only fail the specific
    # tool call that needs it.  The error dict is returned to the agent as a
    # normal tool result — not as a Python exception — so the agent can relay
    # a human-readable error message to the user.
    #
    api_key = os.environ.get(OPENWEATHER_KEY_VAR)
    if not api_key:
        return {
            "error": (
                f"Missing API key. Set the {OPENWEATHER_KEY_VAR} environment "
                "variable to a valid OpenWeather API key."
            )
        }

    # ── 2. Make the HTTP request ───────────────────────────────────────────────
    #
    # params breakdown:
    #   q      – city query string; OpenWeather accepts "City", "City,CountryCode"
    #   appid  – the API key
    #   units  – "metric" returns temperature in °C and wind in m/s.
    #             We also compute °F ourselves so callers get both.
    #
    # We wrap the call in a try/except for the two most common infrastructure
    # failures (no network, slow response) and return informative error dicts
    # rather than letting an unhandled exception crash the server process.
    #
    try:
        response = requests.get(
            OPENWEATHER_BASE_URL,
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=10,  # seconds; prevents the tool from hanging indefinitely
        )
    except requests.exceptions.ConnectionError:
        # DNS failure, refused connection, no internet, etc.
        return {"error": "Network error: could not reach the OpenWeather API."}
    except requests.exceptions.Timeout:
        # Server is reachable but did not respond within 10 seconds.
        return {"error": "Request timed out while contacting the OpenWeather API."}

    # ── 3. Handle API-level error codes ───────────────────────────────────────
    #
    # OpenWeather returns HTTP status codes that map to specific failure modes.
    # We handle the most important ones before falling through to a generic error.
    #
    if response.status_code == 401:
        # The API key was present but rejected — wrong key or not yet activated.
        # New free-tier keys can take up to 2 hours to become active.
        return {
            "error": (
                "Invalid API key. Check the value of "
                f"{OPENWEATHER_KEY_VAR} and ensure it is activated "
                "(new keys can take up to 2 hours to activate)."
            )
        }
    if response.status_code == 404:
        # City name not recognised by the API.
        return {
            "error": (
                f"City '{city}' not found. "
                "Try a different spelling or add a country code, e.g. 'Paris, FR'."
            )
        }
    if not response.ok:
        # Catch-all for any other 4xx / 5xx status (rate limits, server errors…).
        return {
            "error": (
                f"OpenWeather API error {response.status_code}: {response.text}"
            )
        }

    # ── 4. Parse and reshape the response ─────────────────────────────────────
    #
    # The raw OpenWeather JSON is deeply nested. We flatten only the fields that
    # are useful, give them clear names, and include both °C and °F so the agent
    # can answer regardless of which unit the user asked for.
    #
    data = response.json()
    temp_c = data["main"]["temp"]  # Already in °C because we passed units=metric

    return {
        # The API may canonicalise the name ("new york" → "New York") or return
        # the closest-matched city, so we echo back what it matched.
        "city": data["name"],
        "country": data["sys"]["country"],

        # Temperature in both scales
        "temperature_c": round(temp_c, 1),
        "temperature_f": round(temp_c * 9 / 5 + 32, 1),   # Standard conversion formula

        # "Feels like" accounts for wind chill / humidity perception
        "feels_like_c": round(data["main"]["feels_like"], 1),

        # Human-readable sky conditions: "clear sky", "light rain", etc.
        # The API returns lowercase; we capitalise the first letter.
        "conditions": data["weather"][0]["description"].capitalize(),

        # Humidity as a percentage (0–100)
        "humidity_pct": data["main"]["humidity"],

        # Wind speed in metres per second; direction in degrees (0=N, 90=E, …)
        "wind_speed_mps": data["wind"]["speed"],
        "wind_direction_deg": data["wind"].get("deg"),  # .get() because "deg" is optional
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2: STOCK DATA
# ═══════════════════════════════════════════════════════════════════════════════
#
# yfinance is an unofficial Yahoo Finance wrapper.  It scrapes the Yahoo Finance
# website/API and wraps the data in pandas DataFrames.  It does NOT require an
# API key, but it is subject to Yahoo's rate limits and occasional breakages when
# Yahoo changes their site structure.
#
@mcp.tool()
def get_stock_data(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
) -> dict[str, Any]:
    """Return the current price and recent historical prices for a stock ticker.

    Use this tool when the user asks about stock prices, market performance,
    price history, or trends for a publicly traded company.

    Args:
        ticker: The stock ticker symbol, e.g. "AAPL", "MSFT", "TSLA", "^GSPC".
                Case-insensitive; will be uppercased automatically.
        period:   How far back history should go. Valid values:
                  "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max".
                  Defaults to "1mo".
        interval: Granularity of each data point. Valid values:
                  "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
                  "1d", "5d", "1wk", "1mo", "3mo".
                  Defaults to "1d". Short intervals (< "1h") are only available
                  for the last 60 days.
    """

    # ── 1. Normalise ticker symbol ─────────────────────────────────────────────
    #
    # Yahoo Finance symbols are always uppercase (AAPL, not aapl).
    # We normalise here so the agent can pass whatever case it receives from the user.
    #
    ticker = ticker.upper().strip()

    # ── 2. Fetch ticker metadata (info dict) ───────────────────────────────────
    #
    # yf.Ticker() itself never raises — it just creates a lazy object.
    # The actual network call happens when we access .info, which fetches the
    # ticker's metadata JSON from Yahoo Finance.
    #
    try:
        stock = yf.Ticker(ticker)
        info = stock.info   # Network call happens here; may raise on bad tickers
    except Exception as exc:
        # yfinance can raise for completely unrecognised symbols or on network issues.
        return {"error": f"Could not fetch data for ticker '{ticker}': {exc}"}

    # ── 3. Detect invalid tickers that didn't raise ────────────────────────────
    #
    # yfinance sometimes returns a minimal stub dict (e.g. {"symbol": "BADINPUT"})
    # for unknown tickers instead of raising.  We detect this by checking for the
    # absence of any recognised price field.
    #
    if not info or (
        info.get("regularMarketPrice") is None
        and info.get("currentPrice") is None
    ):
        return {
            "error": (
                f"Ticker '{ticker}' returned no price data. "
                "Verify the symbol is correct (e.g. 'AAPL', not 'Apple')."
            )
        }

    # ── 4. Extract key metadata fields ────────────────────────────────────────
    #
    # yfinance field names are camelCase and sometimes differ between market
    # hours ("regularMarketPrice" during the session, "currentPrice" at close).
    # We prefer "currentPrice" and fall back to "regularMarketPrice".
    #
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    currency      = info.get("currency", "USD")
    company_name  = info.get("longName") or info.get("shortName") or ticker

    # ── 5. Fetch OHLCV history ─────────────────────────────────────────────────
    #
    # stock.history() returns a pandas DataFrame with columns:
    #   Open, High, Low, Close, Volume, Dividends, Stock Splits
    # indexed by a DatetimeTZDtype (timezone-aware) timestamp.
    #
    # We wrap this in a separate try/except so that a history failure does not
    # also lose the current price we already retrieved — a partial result is
    # more useful than no result.
    #
    try:
        hist = stock.history(period=period, interval=interval)
    except Exception as exc:
        # Return current price even when history fetch fails so the user gets
        # at least the price they asked for.
        return {
            "error": f"Fetched current price but failed to retrieve history: {exc}",
            "current_price": current_price,
            "currency": currency,
        }

    # ── 6. Convert DataFrame rows to plain dicts ───────────────────────────────
    #
    # MCP tool results must be JSON-serialisable.  pandas Timestamps and numpy
    # float64/int64 values are NOT directly JSON-serialisable, so we convert:
    #   ts.date()    → "YYYY-MM-DD" string
    #   row["Open"]  → Python float (via round())
    #   row["Volume"]→ Python int (via int())
    #
    if hist.empty:
        # Empty DataFrame is not an error — it just means no data for that
        # combination of period and interval (e.g. very short period with a
        # fine interval that Yahoo doesn't carry).
        history_data = []
        history_note = (
            f"No history returned for period='{period}', interval='{interval}'. "
            "Try a longer period or coarser interval."
        )
    else:
        history_note = None
        history_data = [
            {
                "date":   str(ts.date()),           # ISO 8601 date string
                "open":   round(row["Open"],   4),  # 4 decimal places for penny stocks
                "high":   round(row["High"],   4),
                "low":    round(row["Low"],    4),
                "close":  round(row["Close"],  4),
                "volume": int(row["Volume"]),        # always a whole number
            }
            for ts, row in hist.iterrows()
        ]

    # ── 7. Assemble the result dict ────────────────────────────────────────────
    #
    # We include summary stats (market cap, P/E, 52-week range) because they are
    # often what users really want when they ask "how is the stock doing?", and
    # they come for free from the .info dict we already fetched.
    #
    result: dict[str, Any] = {
        "ticker":            ticker,
        "company":           company_name,
        "current_price":     current_price,
        "currency":          currency,

        # Fundamental summary stats (may be None for ETFs, indices, etc.)
        "market_cap":        info.get("marketCap"),
        "pe_ratio":          info.get("trailingPE"),     # trailing twelve months P/E
        "52_week_high":      info.get("fiftyTwoWeekHigh"),
        "52_week_low":       info.get("fiftyTwoWeekLow"),

        # History metadata so the caller knows the scope of the data
        "history_period":    period,
        "history_interval":  interval,
        "history":           history_data,              # list of OHLCV dicts
    }

    # Only add the note key when there is actually a note to avoid cluttering
    # clean responses with a null field.
    if history_note:
        result["history_note"] = history_note

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
#
# When run as a script (`python weather_stock_server.py`), FastMCP starts the
# server loop and communicates over stdio using the MCP JSON-RPC protocol.
#
# When imported as a module (e.g. in tests), this block is skipped, which means
# you can import and call get_current_weather() / get_stock_data() directly
# without starting a server.
#
if __name__ == "__main__":
    # mcp.run() with no arguments defaults to stdio transport.
    # stdio is the correct transport for subprocess-based clients like agent.py,
    # which spawns this script and communicates over its stdin/stdout pipes.
    #
    # IMPORTANT: After this call, stdout belongs to the MCP protocol.
    # Do NOT add any print() calls below (or anywhere in this file when running
    # in server mode) — any stray bytes on stdout will corrupt the JSON-RPC
    # framing and break the connection to the client.
    mcp.run()
