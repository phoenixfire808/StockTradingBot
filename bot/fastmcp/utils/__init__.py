"""Utility modules for the FastMCP server."""

from bot.fastmcp.utils.git_ops import git_repo_path
from bot.fastmcp.utils.code_search import (
    search_files,
    extract_symbols,
    find_references,
    find_definitions,
    dependency_graph,
    class_file_type,
)
from bot.fastmcp.utils.strategies import (
    list_strategies,
    scaffold_strategy,
    validate_strategy,
    optimize_strategy_params,
)
from bot.fastmcp.utils.analytics import (
    query_trades,
    portfolio_summary,
    equity_curve_stats,
    daily_pnl_breakdown,
    risk_metrics,
    kill_switch_stats,
)

__all__ = [
    "git_repo_path",
    "search_files",
    "extract_symbols",
    "find_references",
    "find_definitions",
    "dependency_graph",
    "class_file_type",
    "list_strategies",
    "scaffold_strategy",
    "validate_strategy",
    "optimize_strategy_params",
    "query_trades",
    "portfolio_summary",
    "equity_curve_stats",
    "daily_pnl_breakdown",
    "risk_metrics",
    "kill_switch_stats",
]
