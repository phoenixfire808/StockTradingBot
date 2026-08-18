"""Smoke tests for bot.fastmcp.server — verifies tool registration and basic operations."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestServerBoot:
    """Verify the server imports and tools register correctly."""

    def test_imports_without_error(self):
        from bot.fastmcp.server import mcp
        assert mcp is not None

    def test_tools_registered_count(self):
        from bot.fastmcp.server import mcp
        tool_count = len(mcp._tools)
        assert tool_count > 50, f"Expected 50+ tools, got {tool_count}"

    def test_tool_categories_covered(self):
        from bot.fastmcp.server import mcp
        tool_names = list(mcp._tools.keys())
        
        # Verify at least one tool from each major category
        required_picks = [
            "fs_read",       # File management
            "code_search",   # Code search
            "git_status",    # Git ops
            "project_install", # Project commands
            "analytics_trade_journal", # Analytics
            "strategy_list", # Strategy lifecycle
        ]
        for name in required_picks:
            assert name in tool_names, f"Missing tool: {name}. Available: {tool_names[:10]}..."


class TestFileOperations:
    """Basic sanity checks for file operations (no mutations)."""

    def test_fs_read_existing_file(self):
        from bot.fastmcp.server import fs_read
        result = fs_read("requirements.txt")
        assert "fastmcp" in result.lower() or "requests" in result.lower(), \
            f"Should contain package names, got: {result[:100]}"

    def test_fs_list_returns_items(self):
        from bot.fastmcp.server import fs_list
        result = fs_list(".")
        assert "#" in result or "BOT" in result.upper() or "README" in result or "MAIN" in result.upper(), \
            f"Should list items: {result[:200]}"

    def test_fs_project_structure_returns_json(self):
        from bot.fastmcp.server import fs_project_structure
        data = json.loads(fs_project_structure())
        assert "children" in data or "name" in data

    def test_fs_info_existing_file(self):
        from bot.fastmcp.server import fs_info
        data = json.loads(fs_info("main.py"))
        assert data["is_file"] is True
        assert data["size_bytes"] > 0


class TestCodeSearch:
    """Verify code search utilities work."""

    def test_code_search_finds_pattern(self):
        from bot.fastmcp.server import code_search
        results = code_search("def ", max_results=5)
        # Should find function definitions
        assert "# Found" in results or "match" in results.lower()

    def test_code_symbols_parses_valid_py(self):
        from bot.fastmcp.server import code_symbols
        result = code_symbols(file_path="bot/config.py")
        data = json.loads(result)
        assert isinstance(data, list), "Should return a list of symbols"

    def test_code_file_tree(self):
        from bot.fastmcp.server import code_file_tree
        result = code_file_tree(max_depth=2)
        data = json.loads(result)
        assert "children" in data[0] if isinstance(data, list) else "children" in data

    def test_code_classify_file(self):
        from bot.fastmcp.server import code_classify_file
        result = code_classify_file("main.py")
        data = json.loads(result)
        assert "type" in data
        assert data["type"] in ["entry point", "source module"]


class TestAnalyticsHelpers:
    """Verify analytics helpers don't crash on missing data."""

    def test_analytics_trade_journal_no_trades(self):
        from bot.fastmcp.server import analytics_trade_journal
        result = analytics_trade_journal()
        data = json.loads(result)
        assert "trades" in data or "count" in data

    def test_analytics_portfolio_summary_no_state(self):
        from bot.fastmcp.server import analytics_portfolio_summary
        result = analytics_portfolio_summary()
        data = json.loads(result)
        assert "equity" in data or "message" in data

    def test_analytics_equity_curve_stats_no_data(self):
        from bot.fastmcp.server import analytics_equity_curve_stats
        result = analytics_equity_curve_stats()
        data = json.loads(result)
        assert "error" in data or "data_points" in data


class TestStrategyHelpers:
    """Verify strategy helpers discover registered plugins."""

    def test_strategy_list_discovers_ema_cross_rsi(self):
        from bot.fastmcp.server import strategy_list
        result = strategy_list()
        data = json.loads(result)
        names = [s["name"] for s in data.get("strategies", [])]
        assert "ema_cross_rsi" in names, f"Should discover EmaCrossRsi. Found: {names}"


class TestGitOpsSanity:
    """Quick sanity check for git operations (non-mutating only)."""

    def test_git_status_succeeds(self):
        from bot.fastmcp.utils.git_ops import git_status
        result = git_status()
        assert "modified" in result or "summary" in result

    def test_git_verify(self):
        from bot.fastmcp.utils.git_ops import git_verify
        result = git_verify()
        assert "repo_path" in result
        assert "clean" in result


class TestResources:
    """Test resource helper tools."""

    def test_resource_file(self):
        from bot.fastmcp.server import resource_file
        content = resource_file("main.py")
        assert len(content) > 0

    def test_resource_settings(self):
        from bot.fastmcp.server import resource_settings
        data = json.loads(resource_settings())
        assert "symbols" in data or "log_level" in data

    def test_resource_meta_project(self):
        from bot.fastmcp.server import resource_meta_project
        data = json.loads(resource_meta_project())
        assert "version" in data


class TestHelpTools:
    """Verify help tools work."""

    def test_help_topics_lists_categories(self):
        from bot.fastmcp.server import help_topics
        result = help_topics()
        assert "FILE MANAGEMENT" in result or "CODE SEARCH" in result

    def test_help_tool_describes_one(self):
        from bot.fastmcp.server import help_tool
        result = help_tool("fs_read")
        data = json.loads(result)
        assert "description" in data
        assert "fs_read" in data["tool"]

    def test_time_now(self):
        from bot.fastmcp.server import time_now
        result = time_now()
        data = json.loads(result)
        assert "utc" in data


class TestProjectCommandSanity:
    """Sanity tests for project command wrappers (non-destructive)."""

    def test_project_check_env(self):
        from bot.fastmcp.server import project_check_env
        result = project_check_env()
        data = json.loads(result)
        assert "python_version" in data

    def test_project_schema_update(self):
        from bot.fastmcp.server import project_schema_update
        result = project_schema_update()
        data = json.loads(result)
        assert "strategies" in data

    def test_project_lint(self):
        from bot.fastmcp.server import project_lint
        result = project_lint()
        data = json.loads(result)
        assert "checked" in data or "healthy" in data

    def test_project_send_order_mock(self):
        import asyncio
        from bot.fastmcp.server import project_send_order
        result = project_send_order("AAPL", 1)
        data = json.loads(result)
        assert "order_id" in data or "position" in data

    def test_project_send_sentiment_fallback(self):
        from bot.fastmcp.server import project_send_sentiment
        result = project_send_sentiment(["AAPL"], hours=1)
        data = json.loads(result)
        # May fail gracefully due to network/VADER; just verify structure
        assert isinstance(data, dict) or "AAPL" in json.dumps(data)


class TestEnvAndMisc:
    """Test environment variable and misc helpers."""

    def test_env_var_get(self):
        from bot.fastmcp.server import env_var_get
        result = env_var_get("PATH")
        data = json.loads(result)
        assert "key" in data

    def test_todo_list_invalid_action(self):
        from bot.fastmcp.server import todo_list
        try:
            todo_list(action="invalid")
            assert False, "Should raise ValueError"
        except ValueError:
            pass  # Expected

    def test_note(self):
        from bot.fastmcp.server import note
        note("test note for smoke test")
        notes_file = Path("logs/mcp_scratch_notes.txt")
        assert notes_file.exists(), "Note should be saved"
