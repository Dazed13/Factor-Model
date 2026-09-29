"""Symbol normalization and ticker alias mapping for NSE equities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import polars as pl

# Known NSE ticker renames / aliases (bare NSE symbol -> current bare symbol).
# Extend this table as corporate identity changes are discovered.
DEFAULT_ALIASES: dict[str, str] = {
    "HDFC": "HDFCBANK",  # post-merger continuity for historical joins
    "BHARTIARTL": "BHARTIARTL",
}


@dataclass
class SymbolMap:
    """Bidirectional helper for NSE <-> yfinance ticker forms."""

    aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ALIASES))

    def normalize_nse(self, symbol: str) -> str:
        """Strip exchange suffixes and whitespace; apply alias map."""
        bare = strip_exchange_suffix(symbol)
        return self.aliases.get(bare, bare)

    def to_yfinance(self, symbol: str) -> str:
        """Return Yahoo Finance ticker (NSE cash = ``.NS``)."""
        return f"{self.normalize_nse(symbol)}.NS"

    def from_yfinance(self, symbol: str) -> str:
        """Convert a Yahoo ticker back to bare NSE symbol."""
        return self.normalize_nse(symbol)

    def normalize_frame(
        self,
        df: pl.DataFrame,
        column: str = "symbol",
    ) -> pl.DataFrame:
        """Vectorized bare-NSE normalization for a Polars column."""
        aliases = self.aliases
        expr = (
            pl.col(column)
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.to_uppercase()
            .str.replace(r"\.(NS|BO|NSE|BSE)$", "", literal=False)
        )
        if aliases:
            expr = expr.replace(aliases)
        return df.with_columns(expr.alias(column))

    def extend(self, mapping: Mapping[str, str]) -> None:
        """Merge additional aliases in-place."""
        cleaned = {
            strip_exchange_suffix(k): strip_exchange_suffix(v) for k, v in mapping.items()
        }
        self.aliases.update(cleaned)


def strip_exchange_suffix(symbol: str) -> str:
    """Uppercase bare ticker: ``RELIANCE.NS`` -> ``RELIANCE``."""
    s = symbol.strip().upper()
    for suffix in (".NS", ".BO", ".NSE", ".BSE"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def to_yfinance_symbols(symbols: Iterable[str], symbol_map: SymbolMap | None = None) -> list[str]:
    """Batch-convert bare NSE symbols to yfinance tickers."""
    sm = symbol_map or SymbolMap()
    return [sm.to_yfinance(s) for s in symbols]


def from_yfinance_symbols(symbols: Iterable[str], symbol_map: SymbolMap | None = None) -> list[str]:
    """Batch-convert yfinance tickers to bare NSE symbols."""
    sm = symbol_map or SymbolMap()
    return [sm.from_yfinance(s) for s in symbols]
