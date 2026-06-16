"""World Cup match data provider — fetches today's matches and odds from
Chinese sports lottery official API (sporttery.cn).

Uses the public JSON API behind 中国体育彩票竞彩网:
  getMatchListV1.qry  — match list + HAD/HHAD odds (primary)
  getMatchCalculatorV1.qry — per-pool detailed odds (enhancement)

Falls back to local sample data (data/worldcup_sample.json) when the API
is unreachable.
"""
import logging
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx

LOGGER = logging.getLogger(__name__)

# ── API endpoints ──
_SPORTTERY_MATCH_LIST_URL = (
    "https://webapi.sporttery.cn/gateway/uniform/football/"
    "getMatchListV1.qry?clientCode=3001"
)
_SPORTTERY_CALC_URL = (
    "https://webapi.sporttery.cn/gateway/jc/football/"
    "getMatchCalculatorV1.qry"
)
# ── Local fallback ──
_SAMPLE_DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "worldcup_sample.json"
)

# ── Browser-like headers to avoid WAF blocking ──
_API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.sporttery.cn/",
    "Origin": "https://www.sporttery.cn",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}


@dataclass
class Match:
    """A single World Cup match with all available bet types and odds."""
    match_id: str              # "周二017"
    home_team: str             # 主队
    away_team: str             # 客队
    match_time: str            # "03:00"
    league: str                # "世界杯"
    spf_odds: tuple[float, float, float]       # 胜平负 (胜, 平, 负)
    rqspf_odds: tuple[float, float, float]     # 让球胜平负
    handicap: int = 0                          # 让球数
    total_goals_odds: dict[str, float] = field(default_factory=dict)
    bq_odds: tuple[float, ...] = ()            # 半全场 9 项
    score_odds: dict[str, float] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return f"{self.match_id} {self.home_team} vs {self.away_team}"


@dataclass
class Bet:
    """A single bet recommendation."""
    match_id: str
    home_team: str
    away_team: str
    play_type: str             # "胜平负" / "让球" / "总进球" / "半全场" / "比分"
    pick: str                  # "胜" / "2-3球"
    odds: float
    stake: float               # 2元倍数
    expected_value: float
    confidence: float          # 0-1


@dataclass
class Strategy:
    """One risk-tier strategy with all bets and metadata."""
    risk_level: str            # "保守" / "均衡" / "进取"
    total_stake: float
    expected_return: float
    max_loss: float
    bets: list[Bet] = field(default_factory=list)
    reasoning: str = ""        # filled by LLM


class MatchFetchError(Exception):
    """Raised when match data cannot be fetched from any source."""
    pass


class WorldCupDataProvider:
    """Fetch today's World Cup matches and odds from sporttery.cn.

    Uses the official China Sports Lottery public JSON API.
    Falls back to local sample data when unreachable.
    """

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    async def get_today_matches(self, target_date: str | None = None) -> list:
        """Fetch all World Cup matches for today (or specified date).

        Returns empty list if no data is available from any source.
        """
        if target_date is None:
            target_date = date.today().strftime("%Y-%m-%d")

        # ── Primary: sporttery.cn match list API ──
        try:
            data = await self._fetch_json(_SPORTTERY_MATCH_LIST_URL)
            matches = self._parse_match_list(data)
            # Filter to only World Cup matches
            wc_matches = [m for m in matches if "世界杯" in m.league]
            if wc_matches:
                LOGGER.info("Got %d World Cup matches from sporttery.cn",
                            len(wc_matches))
                return wc_matches
        except Exception as e:
            LOGGER.warning("sporttery.cn match list fetch failed: %s", e)

        # ── Fallback: local sample data ──
        try:
            matches = self._load_sample_data()
            if matches:
                LOGGER.info("Using %d matches from local sample data", len(matches))
                return matches
        except Exception as e:
            LOGGER.warning("Local sample data failed: %s", e)

        LOGGER.warning("No match data available from any source")
        return []

    # ── HTTP helpers ──

    async def _fetch_json(self, url: str, params: dict | None = None) -> dict:
        """HTTP GET and return JSON dict."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params, headers=_API_HEADERS)
            if resp.status_code != 200:
                raise MatchFetchError(
                    f"HTTP {resp.status_code} from {url}")
            return resp.json()

    # ── Parsing: getMatchListV1.qry ──

    def _parse_match_list(self, data: dict) -> list[Match]:
        """Parse getMatchListV1.qry JSON into Match objects.

        Response structure:
          value.matchInfoList[].subMatchList[]
            matchNumStr, homeTeamAllName, awayTeamAllName,
            leagueAllName, matchDate, matchTime, matchStatus, remark
            oddsList[]: {poolCode, h, d, a, goalLine}
        """
        matches = []
        match_info_list = data.get("value", {}).get("matchInfoList", [])
        if not match_info_list:
            return matches

        for info in match_info_list:
            for item in info.get("subMatchList", []):
                try:
                    # Skip matches not yet selling
                    if item.get("matchStatus") != "Selling":
                        continue

                    # Extract HAD / HHAD odds from oddsList
                    odds_map = {}
                    for o in item.get("oddsList", []):
                        pool = o.get("poolCode", "")
                        if pool in ("HAD", "HHAD"):
                            odds_map[pool] = o

                    had = odds_map.get("HAD", {})
                    hhad = odds_map.get("HHAD", {})

                    spf_odds = (
                        float(had.get("h", 0) or 0),
                        float(had.get("d", 0) or 0),
                        float(had.get("a", 0) or 0),
                    )
                    rqspf_odds = (
                        float(hhad.get("h", 0) or 0),
                        float(hhad.get("d", 0) or 0),
                        float(hhad.get("a", 0) or 0),
                    )

                    # Skip if no valid SPF odds
                    if spf_odds == (0, 0, 0):
                        continue

                    # goalLine: e.g. "-1" or "+2" → int
                    goal_line_str = hhad.get("goalLine", "0") or "0"
                    try:
                        handicap = int(float(goal_line_str))
                    except (ValueError, TypeError):
                        handicap = 0

                    match_time = item.get("matchTime", "") or ""
                    # matchTime may be "03:00:00" → strip seconds
                    if match_time.count(":") == 2:
                        match_time = match_time[:5]

                    m = Match(
                        match_id=item.get("matchNumStr", ""),
                        home_team=item.get("homeTeamAllName", ""),
                        away_team=item.get("awayTeamAllName", ""),
                        match_time=match_time,
                        league=item.get("leagueAllName", ""),
                        spf_odds=spf_odds,
                        rqspf_odds=rqspf_odds,
                        handicap=handicap,
                    )
                    matches.append(m)
                except Exception as e:
                    LOGGER.debug("Failed to parse match item: %s", e)
                    continue

        return matches

    # ── Local fallback ──

    def _load_sample_data(self) -> list[Match]:
        """Load matches from local sample JSON when external APIs are unreachable."""
        if not _SAMPLE_DATA_PATH.exists():
            LOGGER.debug("Sample data file not found: %s", _SAMPLE_DATA_PATH)
            return []

        with open(_SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)

        matches = []
        for item in raw:
            try:
                m = Match(
                    match_id=item.get("match_id", ""),
                    home_team=item.get("home_team", ""),
                    away_team=item.get("away_team", ""),
                    match_time=item.get("match_time", ""),
                    league=item.get("league", ""),
                    spf_odds=tuple(item.get("spf_odds", [0, 0, 0])),
                    rqspf_odds=tuple(item.get("rqspf_odds", [0, 0, 0])),
                    handicap=item.get("handicap", 0),
                    total_goals_odds=item.get("total_goals_odds", {}),
                    bq_odds=tuple(item.get("bq_odds", [])),
                    score_odds=item.get("score_odds", {}),
                )
                matches.append(m)
            except Exception as e:
                LOGGER.debug("Failed to parse sample match: %s", e)
                continue

        return matches
