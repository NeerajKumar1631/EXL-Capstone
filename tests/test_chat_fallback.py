"""The rule-based fallback must route prompts to the right tool with NO LLM."""
from unittest.mock import patch

from chat.agent import ChatAgent, _extract_tickers


def _agent_no_llm() -> ChatAgent:
    a = ChatAgent()
    a._ready = False          # force the fallback path
    return a


def test_extract_tickers():
    got = _extract_tickers("Compare AAPL and MSFT please")
    assert "AAPL" in got and "MSFT" in got
    assert "AND" not in got   # stop-word filtered


def test_fallback_routes_compare():
    a = _agent_no_llm()
    with patch("chat.tools._compare_text", return_value="CMP") as m:
        answer, tools = a.ask("compare AAPL and MSFT")
    assert tools == ["compare_stocks"] and "CMP" in answer
    m.assert_called_once()


def test_fallback_routes_risk():
    a = _agent_no_llm()
    with patch("chat.tools._risk_text", return_value="RISK"):
        answer, tools = a.ask("how risky is TSLA?")
    assert tools == ["risk_history"] and "RISK" in answer


def test_fallback_routes_screen():
    a = _agent_no_llm()
    with patch("chat.tools._screen_text", return_value="SCREEN"):
        answer, tools = a.ask("show me top nifty50 names")
    assert tools == ["screen_index"] and "SCREEN" in answer


def test_fallback_routes_analyze():
    a = _agent_no_llm()
    with patch("chat.tools._analyze_text", return_value="ANALYZE"):
        answer, tools = a.ask("what's the outlook on AAPL")
    assert tools == ["analyze_stock"] and "ANALYZE" in answer


def test_fallback_no_ticker_is_helpful():
    a = _agent_no_llm()
    answer, tools = a.ask("hello there")
    assert tools == [] and "ticker" in answer.lower()
    assert "not financial advice" in answer.lower()


def test_fallback_resolves_company_names_when_intent_is_clear():
    """"tesla" in lowercase should reach the risk tool as TSLA via symbol search."""
    from orchestration.schemas import SymbolHit

    a = _agent_no_llm()
    hit = SymbolHit(symbol="TSLA", name="Tesla, Inc.", exchange="NASDAQ", in_region=True)
    with patch("data_ingestion.markets.search_symbols", return_value=[hit]) as search, \
         patch("chat.tools._risk_text", return_value="RISK"):
        answer, tools = a.ask("how risky is tesla?")
    assert tools == ["risk_history"] and "RISK" in answer
    search.assert_called()


def test_fallback_never_searches_on_small_talk():
    """No stock intent -> no symbol search -> no arbitrary match for "hello"."""
    a = _agent_no_llm()
    with patch("data_ingestion.markets.search_symbols") as search:
        answer, tools = a.ask("hello there")
    search.assert_not_called()
    assert tools == [] and "ticker" in answer.lower()
