"""World Cup match data provider — fetches today's matches and odds from
Chinese sports lottery APIs (sporttery.cn → 500.com fallback).

When both external APIs are unreachable (e.g. cloud IP blocked by anti-scraping),
falls back to local sample data (data/worldcup_sample.json) for demo purposes.
"""
import logging
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx

LOGGER = logging.getLogger(__name__)

# ── API endpoints ──
_SPORTTERY_URL = "https://webapi.sporttery.cn/gateway/lottery/getFootBallMatchList.qry"
_FIVEHUNDRED_URL = "https://odds.500.com/fenxi/shuju-{date}.shtml"
# ── Local fallback ──
_SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "worldcup_sample.json"


@dataclass
class Match:
    """A single World Cup match with all available bet types and odds."""
    match_id: str              # "周一001"
    home_team: str             # 主队
    away_team: str             # 客队
    match_time: str            # "09:00"
    league: str                # "世界杯A组"
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
    """Fetch today's World Cup matches and odds from Chinese lottery APIs.

    Priority: sporttery.cn (JSON API) → 500.com (HTML parse).
    """

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    async def get_today_matches(self, target_date: str | None = None) -> list:
        """Fetch all World Cup matches for today (or specified date).

        Returns empty list if no data is available from any source.
        """
        if target_date is None:
            target_date = date.today().strftime("%Y-%m-%d")

        # ── Try sporttery.cn first ──
        try:
            data = await self._fetch_sporttery_games()
            matches = self._parse_sporttery_data(data)
            # Filter to only World Cup matches
            wc_matches = [m for m in matches if "世界杯" in m.league]
            if wc_matches:
                LOGGER.info("Got %d World Cup matches from sporttery.cn",
                            len(wc_matches))
                return wc_matches
        except Exception as e:
            LOGGER.warning("sporttery.cn fetch failed: %s", e)

        # ── Fallback: 500.com ──
        try:
            html = await self._fetch_500com(target_date)
            matches = self._parse_500_html(html)
            if matches:
                LOGGER.info("Got %d matches from 500.com fallback", len(matches))
                return matches
        except Exception as e:
            LOGGER.warning("500.com fetch failed: %s", e)

        # ── Last resort: local sample data (for demo / anti-scraping fallback) ──
        try:
            matches = self._load_sample_data()
            if matches:
                LOGGER.info("Using %d matches from local sample data", len(matches))
                return matches
        except Exception as e:
            LOGGER.warning("Local sample data failed: %s", e)

        LOGGER.warning("No match data available from any source")
        return []

    async def _fetch_sporttery_games(self) -> dict:
        """Fetch match list from sporttery.cn JSON API."""
        params = {
            "matchPage": 1,
            "matchPageSize": 50,
        }
        return await self._fetch_json(_SPORTTERY_URL, params=params)

    async def _fetch_500com(self, target_date: str) -> str:
        """Fetch match page from 500.com."""
        date_str = target_date.replace("-", "")
        url = _FIVEHUNDRED_URL.format(date=date_str)
        return await self._fetch_html(url)

    async def _fetch_json(self, url: str, params: dict | None = None) -> dict:
        """HTTP GET and return JSON dict."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/125.0.0.0 Safari/537.36",
            }
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                raise MatchFetchError(
                    f"HTTP {resp.status_code} from {url}")
            return resp.json()

    async def _fetch_html(self, url: str) -> str:
        """HTTP GET and return HTML text."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36",
            }
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                raise MatchFetchError(
                    f"HTTP {resp.status_code} from {url}")
            return resp.text

    # ── Parsing ──

    def _parse_sporttery_data(self, data: dict) -> list:
        """Parse sporttery.cn JSON response into Match objects."""
        matches = []
        match_list = data.get("data", {}).get("matchList", [])
        if not match_list:
            # Some versions nest differently
            match_list = data.get("value", {}).get("matchList", [])

        for item in match_list:
            try:
                odds = item.get("odds", {}) or {}
                spf = odds.get("spf", {}) or {}
                rqspf = odds.get("rqspf", {}) or {}

                spf_odds = (
                    float(spf.get("w", 0)),
                    float(spf.get("d", 0)),
                    float(spf.get("l", 0)),
                )
                rqspf_odds = (
                    float(rqspf.get("w", 0)),
                    float(rqspf.get("d", 0)),
                    float(rqspf.get("l", 0)),
                )
                handicap = int(rqspf.get("handicap", 0))

                # Skip if no valid odds
                if spf_odds == (0, 0, 0):
                    continue

                match_time = item.get("matchTime", "")
                if " " in match_time:
                    match_time = match_time.split(" ")[-1][:5]

                m = Match(
                    match_id=item.get("matchId", ""),
                    home_team=item.get("homeTeam", ""),
                    away_team=item.get("awayTeam", ""),
                    match_time=match_time,
                    league=item.get("leagueName", ""),
                    spf_odds=spf_odds,
                    rqspf_odds=rqspf_odds,
                    handicap=handicap,
                )
                matches.append(m)
            except Exception as e:
                LOGGER.debug("Failed to parse match item: %s", e)
                continue

        return matches

    def _parse_500_html(self, html: str) -> list:
        """Parse 500.com HTML into Match objects (best-effort regex)."""
        if not html:
            return []
        matches = []
        # Extract match rows from the data table
        rows = re.findall(
            r'<tr[^>]*data-match[^>]*>.*?</tr>', html, re.DOTALL
        )
        for row in rows:
            try:
                # Extract match ID
                mid_match = re.search(r'(\w+\d+)', row)
                if not mid_match:
                    continue
                match_id = mid_match.group(1)

                # Extract teams
                teams = re.findall(r'<a[^>]*>([^<]+)</a>', row)
                if len(teams) >= 2:
                    home, away = teams[0], teams[1]
                else:
                    continue

                # Extract SPF odds (first 3 odds after teams)
                odds = re.findall(r'>(\d+\.\d+)<', row)
                spf_odds = tuple(float(o) for o in odds[:3]) if len(odds) >= 3 else (0, 0, 0)

                if spf_odds == (0, 0, 0):
                    continue

                m = Match(
                    match_id=match_id,
                    home_team=home.strip(),
                    away_team=away.strip(),
                    match_time="",
                    league="",
                    spf_odds=spf_odds,
                    rqspf_odds=(0, 0, 0),
                )
                matches.append(m)
            except Exception:
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
