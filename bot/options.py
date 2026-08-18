"""Options & crypto order dataclasses.

Re-exports the dataclasses defined in :mod:`bot.broker` so that downstream
strategies can import from a stable path (``bot.options``) without creating a
hard dependency on the broker module (which has heavier import cost — MCP, etc.).
"""

from bot.broker import CryptoOrder, CryptoQuote, OptionChain, OptionOrder

__all__ = ["CryptoOrder", "CryptoQuote", "OptionChain", "OptionOrder"]