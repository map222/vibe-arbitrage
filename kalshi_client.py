"""Kalshi market data client using the official kalshi_python_sync SDK.

Wraps the SDK with pandas output, .env-based configuration, and
convenience methods for fetching open markets and orderbooks.

Usage:
    from kalshi_client import KalshiMarketClient
    client = KalshiMarketClient()  # reads from .env
    df = client.get_open_markets(limit=50)
    ob = client.get_orderbook("KXBTC-25MAR15-B100000")
"""

import os
from pathlib import Path

import pandas as pd
from kalshi_python_sync import Configuration, KalshiClient
from dotenv import load_dotenv

load_dotenv()

BASE_URLS = {
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
    "production": "https://api.elections.kalshi.com/trade-api/v2",
}


class KalshiMarketClient:
    """Thin wrapper around kalshi_python_sync.KalshiClient with pandas output."""

    def __init__(self, env=None, api_key_id=None, private_key_path=None):
        """Initialize the client.

        Parameters
        ----------
        env : str, optional
            "demo" or "production". Falls back to KALSHI_ENV env var.
        api_key_id : str, optional
            API key ID. Falls back to KALSHI_API_KEY_ID env var.
        private_key_path : str, optional
            Path to RSA private key PEM file. Falls back to
            KALSHI_PRIVATE_KEY_PATH env var.
        """
        env = env or os.getenv("KALSHI_ENV", "demo")
        if env not in BASE_URLS:
            raise ValueError(f"KALSHI_ENV must be 'demo' or 'production', got '{env}'")

        api_key_id = api_key_id or os.getenv("KALSHI_API_KEY_ID")
        private_key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")

        if not api_key_id:
            raise ValueError("KALSHI_API_KEY_ID is required (set in .env or pass api_key_id)")
        if not private_key_path:
            raise ValueError("KALSHI_PRIVATE_KEY_PATH is required (set in .env or pass private_key_path)")

        private_key = Path(private_key_path).read_text()

        config = Configuration(host=BASE_URLS[env])
        config.api_key_id = api_key_id
        config.private_key_pem = private_key

        self.client = KalshiClient(config)
        self.env = env

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def get_open_markets(
        self,
        limit=1000,
        event_ticker=None,
        series_ticker=None,
        tickers=None,
        min_close_ts=None,
        max_close_ts=None,
        max_pages=None,
    ) -> pd.DataFrame:
        """Fetch open markets and return a DataFrame.

        Parameters
        ----------
        limit : int
            Results per API page (max 1000, default 100).
        event_ticker : str, optional
            Filter by event ticker (comma-separated, max 10).
        series_ticker : str, optional
            Filter by series ticker.
        tickers : str, optional
            Comma-separated list of specific market tickers.
        min_close_ts : int, optional
            Only markets closing after this Unix timestamp.
        max_close_ts : int, optional
            Only markets closing before this Unix timestamp.
        max_pages : int, optional
            Cap on number of pages to fetch. None = all pages.

        Returns
        -------
        pd.DataFrame
        """
        # Use a page size of min(limit, 1000) so we never request more than needed
        page_size = min(limit, 1000)
        kwargs = {"limit": page_size, "status": "open"}
        if event_ticker:
            kwargs["event_ticker"] = event_ticker
        if series_ticker:
            kwargs["series_ticker"] = series_ticker
        if tickers:
            kwargs["tickers"] = tickers
        if min_close_ts is not None:
            kwargs["min_close_ts"] = min_close_ts
        if max_close_ts is not None:
            kwargs["max_close_ts"] = max_close_ts

        all_markets = []
        cursor = None
        pages = 0

        while True:
            if cursor:
                kwargs["cursor"] = cursor

            response = self.client.get_markets(**kwargs)

            # Convert markets to dicts
            all_markets.extend(
                m.to_dict() if hasattr(m, "to_dict") else m.__dict__
                for m in response.markets
            )

            cursor = response.cursor
            pages += 1

            if not cursor or (max_pages and pages >= max_pages) or len(all_markets) >= limit:
                break

        if not all_markets:
            return pd.DataFrame()

        return pd.DataFrame(all_markets[:limit])

    def get_market(self, ticker: str) -> pd.Series:
        """Fetch a single market by ticker, returned as a pandas Series."""
        m = self.client.get_market(ticker)
        data = m.to_dict() if hasattr(m, "to_dict") else m.__dict__
        return pd.Series(data)

    # ------------------------------------------------------------------
    # Orderbook
    # ------------------------------------------------------------------

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Fetch the orderbook for a market.

        Parameters
        ----------
        ticker : str
            Market ticker.
        depth : int
            Orderbook depth (default 10).

        Returns
        -------
        dict with keys:
            "ticker", "yes" (DataFrame of price/quantity),
            "no" (DataFrame of price/quantity), "raw".
        """
        response = self.client.get_market_orderbook(ticker=ticker, depth=depth)
        ob = response.orderbook if hasattr(response, "orderbook") else response

        raw = ob.to_dict() if hasattr(ob, "to_dict") else ob.__dict__

        yes_data = raw.get("yes") or []
        no_data = raw.get("no") or []

        yes_df = pd.DataFrame(yes_data, columns=["price", "quantity"]) if yes_data else pd.DataFrame(columns=["price", "quantity"])
        no_df = pd.DataFrame(no_data, columns=["price", "quantity"]) if no_data else pd.DataFrame(columns=["price", "quantity"])

        return {
            "ticker": ticker,
            "yes": yes_df,
            "no": no_df,
            "raw": raw,
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def search_markets(self, query: str, limit: int = 100) -> pd.DataFrame:
        """Fetch open markets and filter titles by a search string.

        Performs a case-insensitive substring match on the title column.
        """
        df = self.get_open_markets(limit=limit)
        if df.empty:
            return df
        mask = df["title"].str.contains(query, case=False, na=False)
        return df[mask].reset_index(drop=True)
