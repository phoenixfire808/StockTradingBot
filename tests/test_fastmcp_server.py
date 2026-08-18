"""Smoke tests for bot.fastmcp.server — verifies tool registration and basic operations."""

import asyncio
import json
from pathlib import Path

sys_path = str(Path(__file__).resolve().parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)


class TestServerBoot:
    """Verify the server imports and tools register correctly."""

    def test_imports_without_error(self):
        from bot.fastmcp.server import mcp
        assert mcp is not None

    def test_tools_registered_count(self):
        from bot.fastmcp.server import mcp
        tools = asyncio.run(mcp.list_tools())
        assert len(tools) > 50, f"Expected 50+ tools, got {len(tools)}"

    def test_tool_categories_covered(self):
        from bot.fastmcp.server import mcp
        tools = asyncio.run(mcp.list_tools())
        tool_names = [t.name for t in tools]
        required_picks = [
            "fs_read", "code_search", "git_status",
            "project_install", "analytics_trade_journal", "strategy_list",
        ]
        for name in required_picks:
            assert name in tool_names, f"Missing tool: {name}"


class TestFileOperations:
    def test_fs_read_existing_file(self):
        from bot.fastmcp.server import fs_read
        result = fs_read("requirements.txt")
        assert "fastmcp" in result.lower() or "requests" in result.lower()

    def test_fs_list_returns_items(self):
        from bot.fastmcp.server import fs_list
        result = fs_list(".")
        assert "#" in result

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
    def test_code_search_finds_pattern(self):
        from bot.fastmcp.server import code_search
        results = code_search("def ", max_results=5)
        assert "# Found" in results or "match" in results.lower()

    def test_code_symbols_parses_valid_py(self):
        from bot.fastmcp.server import code_symbols
        data = json.loads(code_symbols(file_path="bot/config.py"))
        assert isinstance(data, list)

    def test_code_file_tree(self):
        from bot.fastmcp.server import code_file_tree
        data = json.loads(code_file_tree(max_depth=2))
        assert "children" in data[0] if isinstance(data, list) else "children" in data

    def test_code_classify_file(self):
        from bot.fastmcp.server import code_classify_file
        data = json.loads(code_classify_file("main.py"))
        assert "type" in data


class TestAnalyticsHelpers:
    def test_analytics_trade_journal_no_trades(self):
        from bot.fastmcp.server import analytics_trade_journal
        data = json.loads(analytics_trade_journal())
        assert "trades" in data or "count" in data

    def test_analytics_portfolio_summary_no_state(self):
        from bot.fastmcp.server import analytics_portfolio_summary
        data = json.loads(analytics_portfolio_summary())
        assert "equity" in data or "message" in data

    def test_analytics_equity_curve_stats_no_data(self):
        from bot.fastmcp.server import analytics_equity_curve_stats
        data = json.loads(analytics_equity_curve_stats())
        assert "error" in data or "data_points" in data


class TestStrategyHelpers:
    def test_strategy_list_discovers_ema_cross_rsi(self):
        from bot.fastmcp.server import strategy_list
        data = json.loads(strategy_list())
        names = [s["name"] for s in data.get("strategies", [])]
        assert "ema_cross_rsi" in names, f"Found: {names}"


class TestGitOpsSanity:
    def test_git_status_succeeds(self):
        from bot.fastmcp.utils.git_ops import git_status
        result = git_status()
        assert "modified" in result or "summary" in result

    def test_git_verify(self):
        from bot.fastmcp.utils.git_ops import git_verify
        result = git_verify()
        assert "repo_path" in result and "clean" in result


class TestResources:
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
    def test_help_topics_lists_categories(self):
        from bot.fastmcp.server import help_topics
        result = help_topics()
        assert "FILE MANAGEMENT" in result or "CODE SEARCH" in result

    def test_help_tool_describes_one(self):
        from bot.fastmcp.server import help_tool
        data = json.loads(help_tool("fs_read"))
        assert "description" in data and "fs_read" in data["tool"]

    def test_time_now(self):
        from bot.fastmcp.server import time_now
        data = json.loads(time_now())
        assert "utc" in data


class TestProjectCommandSanity:
    def test_project_check_env(self):
        from bot.fastmcp.server import project_check_env
        data = json.loads(project_check_env())
        assert "python_version" in data

    def test_project_schema_update(self):
        from bot.fastmcp.server import project_schema_update
        data = json.loads(project_schema_update())
        assert "strategies" in data

    def test_project_lint(self):
        from bot.fastmcp.server import project_lint
        data = json.loads(project_lint())
        assert "checked" in data or "healthy" in data

    def test_project_send_order_mock(self):
        from bot.fastmcp.server import project_send_order
        data = json.loads(project_send_order("AAPL", 1))
        assert "order_id" in data or "position" in data


class TestEnvAndMisc:
    def test_env_var_get(self):
        from bot.fastmcp.server import env_var_get
        data = json.loads(env_var_get("PATH"))
        assert "key" in data

    def test_todo_list_invalid_action(self):
        from bot.fastmcp.server import todo_list
        try:
            todo_list(action="invalid")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_note(self):
        from bot.fastmcp.server import note
        note("test smoke note")
        notes_file = PROJECT_ROOT = Path("logs/mcp_scratch_notes.txt")
        assert notes_file.exists()
