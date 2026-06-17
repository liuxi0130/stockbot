"""Strategy engine — multi-factor scoring, Kelly criterion, risk-tier allocation.
Includes parlay (串关) bets for maximum return."""
import math
import itertools
from stockbot.worldcup.data_provider import Match, Bet, Strategy

# Approximate FIFA rankings for 2026 World Cup teams (June 2026 estimates)
_FIFA_RANKING: dict[str, int] = {
    "阿根廷": 1, "法国": 2, "西班牙": 3, "英格兰": 4,
    "巴西": 5, "葡萄牙": 6, "荷兰": 7, "比利时": 8,
    "意大利": 9, "德国": 10, "乌拉圭": 11, "哥伦比亚": 12,
    "墨西哥": 13, "摩洛哥": 14, "日本": 15, "美国": 16,
    "塞内加尔": 17, "伊朗": 18, "丹麦": 19, "韩国": 20,
    "澳大利亚": 23, "加拿大": 27, "沙特": 30, "卡塔尔": 35,
    "新西兰": 40,
}

# 2026 host nations get home-advantage boost
_HOST_NATIONS = {"美国", "加拿大", "墨西哥"}

# Play types available for each risk tier
_SPF_TYPES = ["胜平负"]
_MEDIUM_TYPES = ["胜平负", "让球"]
_ALL_TYPES = ["胜平负", "让球", "总进球", "半全场", "比分"]

# Risk tier configs
# parlay_alloc: {parlay_size: budget_ratio} — fraction of tier budget for each parlay size
# parlay_kelly_discount: extra discount on kelly for parlays (less certain due to independence assumption)
_TIER_CONFIG = {
    "保守": {"kelly_factor": 0.25, "max_budget_ratio": 0.50, "max_single": 0.10,
              "min_prob": 0.45, "play_types": _SPF_TYPES,
              "parlay_sizes": [2],  # only 2串1
              "parlay_budget_ratio": 0.30,  # 30% of budget to parlays
              "parlay_kelly_discount": 0.5,
              "max_parlays": 5},
    "均衡": {"kelly_factor": 0.50, "max_budget_ratio": 0.75, "max_single": 0.15,
              "min_prob": 0.30, "play_types": _MEDIUM_TYPES,
              "parlay_sizes": [2, 3, 4],  # 2串1, 3串1, 4串1
              "parlay_budget_ratio": 0.60,  # 60% to parlays
              "parlay_kelly_discount": 0.65,
              "max_parlays": 8},
    "进取": {"kelly_factor": 1.00, "max_budget_ratio": 0.95, "max_single": 0.20,
              "min_prob": 0.15, "play_types": _ALL_TYPES,
              "parlay_sizes": [2, 3, 4, 5, 6],  # full range
              "parlay_budget_ratio": 0.80,  # 80% to parlays
              "parlay_kelly_discount": 0.80,
              "max_parlays": 12},
}


class StrategyEngine:
    """Generate betting strategies from match data and investment amount."""

    # ── Public API ──

    def generate(self, matches: list[Match], amount: float) -> list[Strategy]:
        """Produce 3 risk-tier strategies from today's matches and budget.

        Each tier now includes both single bets and parlay (串关) bets,
        with budget allocated to maximize expected return.
        """
        if not matches or amount <= 0:
            return [
                Strategy(risk_level=tier, total_stake=0,
                         expected_return=0, max_loss=0)
                for tier in ["保守", "均衡", "进取"]
            ]

        strategies = []
        for tier in ["保守", "均衡", "进取"]:
            cfg = _TIER_CONFIG[tier]

            # Step 1: Generate single bets
            single_bets = self._generate_bets(matches, amount, cfg)

            # Step 2: Generate parlay bets from best picks
            parlay_ratio = cfg.get("parlay_budget_ratio", 0)
            parlay_bets = self._generate_parlay_bets(
                matches, amount, cfg, single_bets)

            # Step 3: Scale stakes by budget split ratio
            all_bets = single_bets + parlay_bets
            if parlay_ratio > 0 and single_bets and parlay_bets:
                single_total = sum(b.stake for b in single_bets)
                parlay_total = sum(b.stake for b in parlay_bets)

                if single_total > 0:
                    single_scale = (
                        amount * (1 - parlay_ratio) / single_total)
                    for b in single_bets:
                        new_stake = self._round_to_2(
                            b.stake * single_scale)
                        b.stake = new_stake

                if parlay_total > 0:
                    parlay_scale = (
                        amount * parlay_ratio / parlay_total)
                    for b in parlay_bets:
                        new_stake = self._round_to_2(
                            b.stake * parlay_scale)
                        b.stake = new_stake

                # Filter bets below min stake
                all_bets = [b for b in all_bets if b.stake >= 2]

            total_stake = sum(b.stake for b in all_bets)
            expected_return = sum(
                b.stake * b.odds * b.confidence for b in all_bets)
            strategies.append(Strategy(
                risk_level=tier,
                total_stake=total_stake,
                expected_return=round(expected_return - total_stake, 2),
                max_loss=total_stake,
                bets=all_bets,
            ))
        return strategies

    def _generate_bets(self, matches: list[Match], amount: float,
                       cfg: dict) -> list[Bet]:
        """Generate all candidate bets for a tier, then filter and allocate."""
        candidates: list[tuple[float, Bet]] = []

        for m in matches:
            # ── 胜平负 ──
            for i, pick in enumerate(["胜", "平", "负"]):
                odds = m.spf_odds[i]
                prob = self._match_probability(m, pick, odds)
                kelly = self._kelly_fraction(odds, prob) * cfg["kelly_factor"]
                if kelly > 0 and prob >= cfg["min_prob"]:
                    candidates.append((kelly, Bet(
                        match_id=m.match_id, home_team=m.home_team,
                        away_team=m.away_team,
                        play_type="胜平负", pick=pick, odds=odds,
                        stake=0.0,
                        expected_value=self._expected_value(odds, prob),
                        confidence=prob,
                    )))

            # ── 让球 ──
            if "让球" in cfg["play_types"] and m.rqspf_odds != (0, 0, 0):
                for i, pick in enumerate(["胜", "平", "负"]):
                    odds = m.rqspf_odds[i]
                    if odds <= 0:
                        continue
                    prob = self._match_probability(m, pick, odds,
                                                   handicap=m.handicap)
                    kelly = self._kelly_fraction(odds, prob) * cfg["kelly_factor"]
                    if kelly > 0 and prob >= cfg["min_prob"]:
                        candidates.append((kelly, Bet(
                            match_id=m.match_id, home_team=m.home_team,
                            away_team=m.away_team,
                            play_type="让球",
                            pick=f"{pick}({'+' if m.handicap > 0 else ''}{m.handicap})",
                            odds=odds, stake=0.0,
                            expected_value=self._expected_value(odds, prob),
                            confidence=prob,
                        )))

            # ── 总进球 ──
            if "总进球" in cfg["play_types"] and m.total_goals_odds:
                best_goals = max(
                    m.total_goals_odds.items(),
                    key=lambda kv: self._implied_probability(float(kv[1]))
                )
                odds = best_goals[1]
                prob = self._implied_probability(odds) * 0.9  # discount for uncertainty
                kelly = self._kelly_fraction(odds, prob) * cfg["kelly_factor"]
                if kelly > 0 and prob >= cfg["min_prob"]:
                    candidates.append((kelly, Bet(
                        match_id=m.match_id, home_team=m.home_team,
                        away_team=m.away_team,
                        play_type="总进球", pick=f"{best_goals[0]}球",
                        odds=odds, stake=0.0,
                        expected_value=self._expected_value(odds, prob),
                        confidence=prob,
                    )))

            # ── 半全场 ──
            if "半全场" in cfg["play_types"] and m.bq_odds:
                bq_labels = ["胜胜", "胜平", "胜负", "平胜", "平平", "平负",
                             "负胜", "负平", "负负"]
                if len(m.bq_odds) >= 9:
                    best_idx = max(
                        range(9),
                        key=lambda i: self._implied_probability(m.bq_odds[i])
                    )
                    odds = m.bq_odds[best_idx]
                    prob = self._implied_probability(odds) * 0.85
                    kelly = self._kelly_fraction(odds, prob) * cfg["kelly_factor"]
                    if kelly > 0 and prob >= cfg["min_prob"]:
                        candidates.append((kelly, Bet(
                            match_id=m.match_id, home_team=m.home_team,
                            away_team=m.away_team,
                            play_type="半全场", pick=bq_labels[best_idx],
                            odds=odds, stake=0.0,
                            expected_value=self._expected_value(odds, prob),
                            confidence=prob,
                        )))

            # ── 比分 ──
            if "比分" in cfg["play_types"] and m.score_odds:
                best_score = max(
                    m.score_odds.items(),
                    key=lambda kv: self._implied_probability(float(kv[1]))
                )
                odds = best_score[1]
                prob = self._implied_probability(odds) * 0.8
                kelly = self._kelly_fraction(odds, prob) * cfg["kelly_factor"]
                if kelly > 0 and prob >= cfg["min_prob"]:
                    candidates.append((kelly, Bet(
                        match_id=m.match_id, home_team=m.home_team,
                        away_team=m.away_team,
                        play_type="比分", pick=best_score[0],
                        odds=odds, stake=0.0,
                        expected_value=self._expected_value(odds, prob),
                        confidence=prob,
                    )))

        # Sort by EV descending, allocate proportionally to Kelly fractions
        candidates.sort(key=lambda x: x[1].expected_value, reverse=True)
        total_kelly = sum(k for k, _ in candidates)
        if total_kelly <= 0 or not candidates:
            return []

        max_single = self._round_to_2(amount * cfg["max_single"])
        budget = amount * cfg["max_budget_ratio"]
        result = []
        for k, bet in candidates:
            # Proportional stake allocation + round to nearest 2
            raw_stake = (k / total_kelly) * budget
            bet.stake = min(self._round_to_2(raw_stake), max_single)
            if bet.stake >= 2:
                result.append(bet)
        return result

    # ── Internal scoring methods ──

    def _generate_parlay_bets(self, matches: list[Match], amount: float,
                               cfg: dict, single_bets: list[Bet]) -> list[Bet]:
        """Generate parlay (串关) bets from best single picks.

        For each parlay size in cfg["parlay_sizes"], combines the best
        EV picks across matches. Combined odds = product of leg odds;
        combined probability = product of leg probabilities.
        """
        parlay_sizes = cfg.get("parlay_sizes", [])
        if not parlay_sizes:
            return []

        # Pick the best single bet per match (highest EV)
        best_by_match: dict[str, Bet] = {}
        for b in single_bets:
            key = b.match_id
            if (key not in best_by_match
                    or b.expected_value > best_by_match[key].expected_value):
                best_by_match[key] = b

        picks = list(best_by_match.values())
        if len(picks) < 2:
            return []

        kelly_discount = cfg.get("parlay_kelly_discount", 0.5)
        max_parlays = cfg.get("max_parlays", 8)
        max_stake = self._round_to_2(amount * cfg["max_single"] * 0.5)
        budget = amount * cfg["max_budget_ratio"]

        all_parlays: list[tuple[float, Bet]] = []

        for size in parlay_sizes:
            if size > len(picks):
                continue

            size_parlays: list[tuple[float, Bet]] = []
            for combo in itertools.combinations(picks, size):
                # Check no duplicate matches in combo
                match_ids = {b.match_id for b in combo}
                if len(match_ids) < size:
                    continue

                # Only combine matches from the same day
                # match_id format: "周一001", "周二017" — first 2 chars = day
                day_prefixes = {m[:2] for m in match_ids}
                if len(day_prefixes) > 1:
                    continue

                # Combined odds and probability
                combined_odds = 1.0
                combined_prob = 1.0
                for b in combo:
                    combined_odds *= b.odds
                    combined_prob *= b.confidence

                ev = combined_prob * combined_odds - 1
                kelly = (self._kelly_fraction(combined_odds, combined_prob)
                         * kelly_discount * cfg["kelly_factor"])

                if kelly > 0:
                    pick_desc = "+".join(
                        f"{b.home_team}{b.pick}" for b in combo)
                    leg_info = [
                        {"match_id": b.match_id, "home": b.home_team,
                         "away": b.away_team, "pick": b.pick,
                         "odds": b.odds, "play_type": b.play_type}
                        for b in combo
                    ]
                    size_parlays.append((kelly, Bet(
                        match_id=f"{size}串1",
                        home_team="",
                        away_team="",
                        play_type=f"{size}串1",
                        pick=pick_desc,
                        odds=round(combined_odds, 2),
                        stake=0.0,
                        expected_value=round(ev, 4),
                        confidence=round(combined_prob, 4),
                        parlay_legs=leg_info,
                    )))

            # Sort by EV descending, take top N per size
            size_parlays.sort(key=lambda x: x[1].expected_value, reverse=True)
            per_size = max(1, max_parlays // len(parlay_sizes))
            all_parlays.extend(size_parlays[:per_size])

        # Allocate stakes proportionally by Kelly fraction
        total_kelly = sum(k for k, _ in all_parlays)
        if total_kelly <= 0 or not all_parlays:
            return []

        result = []
        for k, bet in all_parlays:
            raw_stake = (k / total_kelly) * budget
            bet.stake = min(self._round_to_2(raw_stake), max_stake)
            if bet.stake >= 2:
                result.append(bet)
        return result

    def _match_probability(self, match: Match, pick: str, odds: float,
                           handicap: int = 0) -> float:
        """Compute multi-factor win/draw/loss probability for a pick."""
        implied = self._implied_probability(odds)
        rank_score = self._rank_score(match.home_team, match.away_team)
        home_bonus = self._home_advantage(match.home_team)

        # Adjust base probability by rank difference and home advantage
        if pick in ("胜", "胜胜", "平胜", "负胜") or (
                pick.startswith("胜(") and handicap <= 0):
            adjusted = implied + rank_score * 0.05 + home_bonus * 0.03
        elif pick in ("负", "负负", "胜负", "平负") or (
                pick.startswith("负(") and handicap <= 0):
            adjusted = implied - rank_score * 0.05 - home_bonus * 0.03
        else:
            # Draw — less sensitive to rank diff
            adjusted = implied * (1 - abs(rank_score) * 0.03)

        # Bookmaker margin is embedded in odds; multi-factor adjustment
        # provides our edge estimate vs. the market. Removing margin
        # entirely would make all bets negative Kelly (no value exists
        # in pure odds). Instead, keep implied as-is and let factor
        # adjustments (rank, home, etc.) surface relative value.
        return max(0.05, min(0.95, adjusted))

    def _implied_probability(self, odds: float) -> float:
        """Convert decimal odds to implied probability."""
        if odds <= 0:
            return 0.0
        return 1.0 / odds

    def _kelly_fraction(self, odds: float, probability: float) -> float:
        """Kelly criterion: f* = (bp - q) / b."""
        b = odds - 1
        if b <= 0:
            return 0.0
        q = 1 - probability
        f = (b * probability - q) / b
        return max(0.0, f)

    def _expected_value(self, odds: float, probability: float) -> float:
        """EV = probability * odds - 1."""
        return probability * odds - 1

    def _rank_score(self, home: str, away: str) -> float:
        """Normalized rank difference score (-1 to 1, positive = home better)."""
        home_rank = _FIFA_RANKING.get(home, 40)
        away_rank = _FIFA_RANKING.get(away, 40)
        diff = away_rank - home_rank
        return max(-1.0, min(1.0, diff / 30.0))

    def _home_advantage(self, team: str) -> float:
        """Host nation bonus (0.0 to 1.0)."""
        return 0.6 if team in _HOST_NATIONS else 0.0

    def _round_to_2(self, value: float) -> float:
        """Round stake down to nearest multiple of 2 (min bet unit)."""
        return max(0, math.floor(value / 2) * 2)
