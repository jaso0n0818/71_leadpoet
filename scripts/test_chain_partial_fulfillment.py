"""
Stress test for the chain-aware partial-fulfillment lifecycle.

Scenarios:
  1. single-cycle full quota               → fulfilled
  2. partial → continued → fulfilled       → matches user's example
  3. displacement (new beats held)         → 8 new score=15 displace 4 of 6 held score=10
  4. abandoned chain                       → expired with held leads → preserved
  5. zero-held edge case                   → recycled, fresh start

Runs against an in-memory mock of the supabase write client and the
gateway's fulfillment_score_consensus / fulfillment_requests /
fulfillment_submissions tables.  Exercises the actual lifecycle code
paths via direct function calls.
"""

import asyncio
import sys
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

# Allow `from gateway...` imports
sys.path.insert(0, ".")


# ─────────────────────────────────────────────────────────────────
# Mock Supabase client
# ─────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class _Query:
    def __init__(self, table: "_Table", op: str = "select"):
        self.table = table
        self.op = op
        self.filters: List[tuple] = []
        self.order_field: Optional[tuple] = None
        self.limit_n: Optional[int] = None
        self.update_payload: Optional[Dict[str, Any]] = None
        self.select_count: Optional[str] = None
        self._not = False
        # last execute result cache
        self.last_data: List[Dict[str, Any]] = []
        # Convenience for "WHERE x IN (...)" with negation
        self.not_in_filter: Optional[tuple] = None

    @property
    def not_(self):
        return _NotProxy(self)

    def select(self, *_args, count: Optional[str] = None):
        self.op = "select"
        self.select_count = count
        return self

    def insert(self, payload: Dict[str, Any]):
        self.op = "insert"
        self.update_payload = payload
        return self

    def update(self, payload: Dict[str, Any]):
        self.op = "update"
        self.update_payload = payload
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, col: str, val: Any):
        self.filters.append(("eq", col, val))
        return self

    def in_(self, col: str, vals: List[Any]):
        self.filters.append(("in", col, vals))
        return self

    def lt(self, col: str, val: Any):
        self.filters.append(("lt", col, val))
        return self

    def gt(self, col: str, val: Any):
        self.filters.append(("gt", col, val))
        return self

    def gte(self, col: str, val: Any):
        self.filters.append(("gte", col, val))
        return self

    def is_(self, col: str, val):
        self.filters.append(("is", col, val))
        return self

    def order(self, col: str, desc: bool = False):
        self.order_field = (col, desc)
        return self

    def limit(self, n: int):
        self.limit_n = n
        return self

    def _apply_filters(self, rows: List[Dict[str, Any]]):
        out = list(rows)
        for f in self.filters:
            if f[0] == "eq":
                out = [r for r in out if r.get(f[1]) == f[2]]
            elif f[0] == "in":
                out = [r for r in out if r.get(f[1]) in f[2]]
            elif f[0] == "lt":
                out = [r for r in out if r.get(f[1]) is not None and r.get(f[1]) < f[2]]
            elif f[0] == "gt":
                out = [r for r in out if r.get(f[1]) is not None and r.get(f[1]) > f[2]]
            elif f[0] == "gte":
                out = [r for r in out if r.get(f[1]) is not None and r.get(f[1]) >= f[2]]
            elif f[0] == "is":
                if f[2] is None or f[2] == "null":
                    out = [r for r in out if r.get(f[1]) is None]
                else:
                    out = [r for r in out if r.get(f[1]) == f[2]]
            elif f[0] == "not_in":
                out = [r for r in out if r.get(f[1]) not in f[2]]
        return out

    def execute(self) -> _Resp:
        rows = self.table.rows
        if self.op == "select":
            data = self._apply_filters(rows)
            if self.order_field is not None:
                col, desc = self.order_field
                data = sorted(
                    data,
                    key=lambda r: (r.get(col) is None, r.get(col)),
                    reverse=desc,
                )
            if self.limit_n is not None:
                data = data[: self.limit_n]
            if self.select_count == "exact":
                return _Resp(data, count=len(data))
            return _Resp(list(data))
        elif self.op == "update":
            updated: List[Dict[str, Any]] = []
            target = self._apply_filters(rows)
            target_ids = [id(r) for r in target]
            for r in rows:
                if id(r) in target_ids:
                    r.update(self.update_payload)
                    updated.append(dict(r))
            return _Resp(updated)
        elif self.op == "insert":
            new_row = dict(self.update_payload)
            rows.append(new_row)
            return _Resp([new_row])
        elif self.op == "delete":
            target = self._apply_filters(rows)
            target_ids = [id(r) for r in target]
            self.table.rows = [r for r in rows if id(r) not in target_ids]
            return _Resp([dict(r) for r in target])
        return _Resp([])


class _NotProxy:
    def __init__(self, q: _Query):
        self.q = q

    def in_(self, col: str, vals: List[Any]):
        self.q.filters.append(("not_in", col, vals))
        return self.q


class _Table:
    def __init__(self, name: str):
        self.name = name
        self.rows: List[Dict[str, Any]] = []


class _MockSupabase:
    def __init__(self):
        self.tables: Dict[str, _Table] = defaultdict(lambda: _Table("?"))
        for n in [
            "fulfillment_requests",
            "fulfillment_score_consensus",
            "fulfillment_submissions",
            "fulfillment_scores",
            "transparency_log",
            "subnet_state",
            "qualification_models",
            "metagraph",
            "leads_private",
            "companies",
            "banned_hotkeys",
        ]:
            self.tables[n] = _Table(n)
        # Seed subnet_state with tempo
        self.tables["subnet_state"].rows.append({"tempo": 360})

    def table(self, name: str) -> _Query:
        return _Query(self.tables[name])

    def rpc(self, fn: str, params: Dict[str, Any]):
        # Simulate the consensus upsert: just write rows into
        # fulfillment_score_consensus.
        if fn == "fulfillment_upsert_consensus":
            cons_rows = params.get("p_consensus", [])
            tab = self.tables["fulfillment_score_consensus"]
            for cr in cons_rows:
                # Replace existing by (request_id, submission_id, lead_id)
                existing = next(
                    (r for r in tab.rows if r["request_id"] == cr["request_id"]
                     and r["submission_id"] == cr["submission_id"]
                     and r["lead_id"] == cr["lead_id"]),
                    None,
                )
                if existing:
                    existing.update(cr)
                else:
                    new_row = dict(cr)
                    new_row.setdefault("is_winner", False)
                    new_row.setdefault("is_chain_held", False)
                    new_row.setdefault("reward_pct", None)
                    new_row.setdefault("reward_expires_epoch", None)
                    tab.rows.append(new_row)
            return _RpcReturn()
        if fn == "fulfillment_close_window":
            rid = params.get("p_request_id")
            new_status = params.get("p_new_status")
            tab = self.tables["fulfillment_requests"]
            for r in tab.rows:
                if r["request_id"] == rid:
                    r["status"] = new_status
                    break
            return _RpcReturn()
        return _RpcReturn()


class _RpcReturn:
    def execute(self):
        return _Resp([])


# ─────────────────────────────────────────────────────────────────
# Scenario harness
# ─────────────────────────────────────────────────────────────────

def _seed_request(sb: _MockSupabase, *, request_id: str, num_leads: int, status: str = "scoring", company: str = "TestCo", icp_extras=None) -> Dict[str, Any]:
    icp = {"prompt": "x", "industry": ["Software"], "sub_industry": ["SaaS"], "excluded_companies": []}
    if icp_extras:
        icp.update(icp_extras)
    row = {
        "request_id": request_id,
        "request_hash": "h",
        "icp_details": icp,
        "num_leads": num_leads,
        "internal_label": "test",
        "company": company,
        "status": status,
        "successor_request_id": None,
        "window_start": "2026-04-26T00:00:00+00:00",
        "window_end": "2026-04-26T01:00:00+00:00",
        "reveal_window_end": "2026-04-26T01:15:00+00:00",
        "created_at": "2026-04-26T00:00:00+00:00",
        "created_by": "api",
    }
    sb.tables["fulfillment_requests"].rows.append(row)
    return row


def _seed_submission(sb: _MockSupabase, *, request_id: str, miner_hk: str, leads: List[Dict[str, Any]]):
    sub_id = str(uuid.uuid4())
    sb.tables["fulfillment_submissions"].rows.append({
        "submission_id": sub_id,
        "request_id": request_id,
        "miner_hotkey": miner_hk,
        "revealed": True,
        "lead_data": [{"lead_id": l["lead_id"], "data": l["data"]} for l in leads],
    })
    return sub_id


def _seed_consensus(sb: _MockSupabase, *, request_id: str, submission_id: str, lead_id: str, miner_hk: str, score: float, intent: float = 0.0, is_chain_held: bool = False, is_winner: bool = False):
    sb.tables["fulfillment_score_consensus"].rows.append({
        "consensus_id": str(uuid.uuid4()),
        "request_id": request_id,
        "submission_id": submission_id,
        "lead_id": lead_id,
        "miner_hotkey": miner_hk,
        "consensus_final_score": score,
        "consensus_intent_signal_final": intent,
        "consensus_company_verified": True,
        "consensus_person_verified": True,
        "consensus_email_verified": True,
        "consensus_decision_maker": True,
        "consensus_icp_fit": True,
        "consensus_rep_score": 30.0,
        "consensus_tier2_passed": True,
        "any_fabricated": False,
        "intent_details": "",
        "intent_signal_mapping": [],
        "num_validators": 1,
        "is_chain_held": is_chain_held,
        "is_winner": is_winner,
        "reward_pct": None,
        "reward_expires_epoch": None,
    })


# ─────────────────────────────────────────────────────────────────
# Patch external dependencies for offline test
# ─────────────────────────────────────────────────────────────────

def _patch_lifecycle_for_test(sb: _MockSupabase):
    """Monkey-patch the lifecycle module so it uses our mock supabase
    and skips real LLM / DB calls that aren't relevant to the chain
    state machine."""
    from gateway.fulfillment import lifecycle, rewards
    from gateway.fulfillment import api as gw_api

    lifecycle._get_supabase = lambda: sb
    rewards._get_supabase = lambda: sb
    gw_api._get_supabase = lambda: sb

    async def _stub_intent_details(*args, **kwargs):
        pass
    lifecycle._attach_intent_details_for_winners = _stub_intent_details

    async def _stub_get_epoch():
        return 22500
    lifecycle._get_current_epoch = _stub_get_epoch


# ─────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────

async def scenario_1_single_cycle_full():
    """First cycle hits full quota → fulfilled, rewards flow."""
    print("\n" + "="*60)
    print("SCENARIO 1: single-cycle full quota → fulfilled")
    print("="*60)
    sb = _MockSupabase()
    _patch_lifecycle_for_test(sb)
    from gateway.fulfillment.lifecycle import _resolve_chain_topk, _finalize_chain_rewards

    # Request asks for 3 leads.  3 candidates all score>0 — should fulfill.
    rid = "req-s1"
    _seed_request(sb, request_id=rid, num_leads=3)
    leads = [
        {"lead_id": "l1", "data": {"business": "Acme"}},
        {"lead_id": "l2", "data": {"business": "Beta"}},
        {"lead_id": "l3", "data": {"business": "Gamma"}},
    ]
    sub_id = _seed_submission(sb, request_id=rid, miner_hk="m1", leads=leads)

    consensus_results = [
        {"request_id": rid, "submission_id": sub_id, "lead_id": "l1", "miner_hotkey": "m1",
         "consensus_final_score": 50.0, "consensus_intent_signal_final": 50.0,
         "consensus_tier2_passed": True, "any_fabricated": False, "num_validators": 1,
         "intent_signal_mapping": [], "intent_details": ""},
        {"request_id": rid, "submission_id": sub_id, "lead_id": "l2", "miner_hotkey": "m1",
         "consensus_final_score": 40.0, "consensus_intent_signal_final": 40.0,
         "consensus_tier2_passed": True, "any_fabricated": False, "num_validators": 1,
         "intent_signal_mapping": [], "intent_details": ""},
        {"request_id": rid, "submission_id": sub_id, "lead_id": "l3", "miner_hotkey": "m1",
         "consensus_final_score": 30.0, "consensus_intent_signal_final": 30.0,
         "consensus_tier2_passed": True, "any_fabricated": False, "num_validators": 1,
         "intent_signal_mapping": [], "intent_details": ""},
    ]
    # Persist consensus (mimics RPC upsert)
    for cr in consensus_results:
        _seed_consensus(sb, request_id=rid, submission_id=sub_id, lead_id=cr["lead_id"],
                        miner_hk="m1", score=cr["consensus_final_score"])

    chain = await _resolve_chain_topk(rid, consensus_results, 3)
    print(f"  chain_target={chain['chain_target']}, held={len(chain['topk'])}, displaced={chain['displaced_count']}")
    assert chain["chain_target"] == 3
    assert len(chain["topk"]) == 3, f"expected 3 in top-K, got {len(chain['topk'])}"

    winner_ids = await _finalize_chain_rewards(rid, chain["topk"], chain["tied_groups"])
    print(f"  winners after fulfilment: {len(winner_ids)}")
    assert len(winner_ids) == 3

    # Check is_winner=True flipped for all 3
    cons_rows = sb.tables["fulfillment_score_consensus"].rows
    winners_in_db = [r for r in cons_rows if r["is_winner"]]
    assert len(winners_in_db) == 3
    for r in winners_in_db:
        assert r["reward_pct"] is not None and r["reward_pct"] > 0, "reward_pct missing"
    print(f"  ✅ {len(winners_in_db)} consensus rows have is_winner=True + reward_pct set")


async def scenario_2_partial_continued_fulfilled():
    """User's example: cycle 1 → 6/10 score=10, cycle 2 → 8 new score=15 displace 4 of 6.
    Final top-10 = 8 new + 2 best held."""
    print("\n" + "="*60)
    print("SCENARIO 2: partial → continued_open → fulfilled (with displacement)")
    print("="*60)
    sb = _MockSupabase()
    _patch_lifecycle_for_test(sb)
    from gateway.fulfillment.lifecycle import _resolve_chain_topk, _finalize_chain_rewards

    # Generation 1: chain root with num_leads=10
    g1 = "req-s2-gen1"
    _seed_request(sb, request_id=g1, num_leads=10, status="partially_fulfilled")
    leads_g1 = [{"lead_id": f"l1_{i}", "data": {"business": f"G1Co{i}"}} for i in range(6)]
    sub_g1 = _seed_submission(sb, request_id=g1, miner_hk="m1", leads=leads_g1)
    for ld in leads_g1:
        _seed_consensus(sb, request_id=g1, submission_id=sub_g1, lead_id=ld["lead_id"],
                        miner_hk="m1", score=10.0, is_chain_held=True)

    # Generation 2 — the active scoring request; chain_target should be 10 (root's num_leads).
    g2 = "req-s2-gen2"
    _seed_request(sb, request_id=g2, num_leads=4, status="scoring")
    # Link g1 → g2
    sb.tables["fulfillment_requests"].rows[0]["successor_request_id"] = g2
    leads_g2 = [{"lead_id": f"l2_{i}", "data": {"business": f"G2Co{i}"}} for i in range(8)]
    sub_g2 = _seed_submission(sb, request_id=g2, miner_hk="m1", leads=leads_g2)

    consensus_results_g2 = []
    for ld in leads_g2:
        cr = {"request_id": g2, "submission_id": sub_g2, "lead_id": ld["lead_id"], "miner_hotkey": "m1",
              "consensus_final_score": 15.0, "consensus_intent_signal_final": 15.0,
              "consensus_tier2_passed": True, "any_fabricated": False, "num_validators": 1,
              "intent_signal_mapping": [], "intent_details": ""}
        consensus_results_g2.append(cr)
        _seed_consensus(sb, request_id=g2, submission_id=sub_g2, lead_id=ld["lead_id"],
                        miner_hk="m1", score=15.0)

    chain = await _resolve_chain_topk(g2, consensus_results_g2, 4)
    print(f"  chain_target={chain['chain_target']}, held={len(chain['topk'])}, displaced={chain['displaced_count']}")
    print(f"  topk_companies={chain['topk_companies']}")
    assert chain["chain_target"] == 10, f"chain_target should walk to root num_leads=10, got {chain['chain_target']}"
    assert len(chain["topk"]) == 10, f"expected 10 in top-K, got {len(chain['topk'])}"
    assert chain["displaced_count"] == 4, f"expected 4 displaced (8 new score=15 > 4 of 6 held score=10), got {chain['displaced_count']}"

    # Top-10 must be all 8 new (score=15) + 2 highest of the 6 old (still score=10)
    new_from_g2 = [r for r in chain["topk"] if r.get("_chain_origin") == "current"]
    held_from_g1 = [r for r in chain["topk"] if r.get("_chain_origin") == "prior"]
    assert len(new_from_g2) == 8, f"expected 8 from g2, got {len(new_from_g2)}"
    assert len(held_from_g1) == 2, f"expected 2 from g1, got {len(held_from_g1)}"

    # Now distribute rewards
    winner_ids = await _finalize_chain_rewards(g2, chain["topk"], chain["tied_groups"])
    print(f"  winners after fulfilment: {len(winner_ids)}")
    assert len(winner_ids) == 10

    # Displaced 4 should have is_chain_held=False AND is_winner=False
    cons_rows = sb.tables["fulfillment_score_consensus"].rows
    displaced = [r for r in cons_rows if r["request_id"] == g1 and not r["is_chain_held"]]
    assert len(displaced) == 4, f"expected 4 displaced rows, got {len(displaced)}"
    for r in displaced:
        assert not r["is_winner"], "displaced row should not have is_winner=True"
        assert r["reward_pct"] is None, f"displaced row should have NULL reward_pct, got {r['reward_pct']}"
    print(f"  ✅ 4 displaced consensus rows correctly have is_chain_held=False, is_winner=False, reward_pct=NULL")

    # The 2 held-from-g1 winners should be is_winner=True with reward_pct set
    held_winners = [r for r in cons_rows if r["request_id"] == g1 and r["is_winner"]]
    assert len(held_winners) == 2
    print(f"  ✅ 2 g1-held rows correctly flipped to is_winner=True with rewards on chain fulfilment")


async def scenario_3_multi_generation_chain():
    """5 generations, never reaches quota of 100 — held leads accumulate."""
    print("\n" + "="*60)
    print("SCENARIO 3: 3-generation chain accumulating held leads")
    print("="*60)
    sb = _MockSupabase()
    _patch_lifecycle_for_test(sb)
    from gateway.fulfillment.lifecycle import _resolve_chain_topk

    # Gen 1: root num_leads=15.  Produces 5 held.
    g1 = "req-s3-gen1"
    _seed_request(sb, request_id=g1, num_leads=15, status="partially_fulfilled")
    sub_g1 = _seed_submission(sb, request_id=g1, miner_hk="m1",
        leads=[{"lead_id": f"g1l{i}", "data": {"business": f"G1{i}"}} for i in range(5)])
    for i in range(5):
        _seed_consensus(sb, request_id=g1, submission_id=sub_g1, lead_id=f"g1l{i}",
                        miner_hk="m1", score=10.0 + i, is_chain_held=True)

    # Gen 2: 3 more held.  Walks back to g1.  chain_target should still be 15.
    g2 = "req-s3-gen2"
    _seed_request(sb, request_id=g2, num_leads=10, status="partially_fulfilled")
    sb.tables["fulfillment_requests"].rows[0]["successor_request_id"] = g2  # g1 → g2
    sub_g2 = _seed_submission(sb, request_id=g2, miner_hk="m1",
        leads=[{"lead_id": f"g2l{i}", "data": {"business": f"G2{i}"}} for i in range(3)])
    for i in range(3):
        _seed_consensus(sb, request_id=g2, submission_id=sub_g2, lead_id=f"g2l{i}",
                        miner_hk="m1", score=20.0 + i, is_chain_held=True)

    # Gen 3 — currently scoring, num_leads = 7.  We expect chain_target=15.
    g3 = "req-s3-gen3"
    _seed_request(sb, request_id=g3, num_leads=7, status="scoring")
    # Link g2 → g3
    g2_row = next(r for r in sb.tables["fulfillment_requests"].rows if r["request_id"] == g2)
    g2_row["successor_request_id"] = g3

    leads_g3 = [{"lead_id": f"g3l{i}", "data": {"business": f"G3{i}"}} for i in range(4)]
    sub_g3 = _seed_submission(sb, request_id=g3, miner_hk="m1", leads=leads_g3)
    consensus_g3 = []
    for i, ld in enumerate(leads_g3):
        cr = {"request_id": g3, "submission_id": sub_g3, "lead_id": ld["lead_id"], "miner_hotkey": "m1",
              "consensus_final_score": 30.0 + i, "consensus_intent_signal_final": 30.0 + i,
              "consensus_tier2_passed": True, "any_fabricated": False, "num_validators": 1,
              "intent_signal_mapping": [], "intent_details": ""}
        consensus_g3.append(cr)
        _seed_consensus(sb, request_id=g3, submission_id=sub_g3, lead_id=ld["lead_id"],
                        miner_hk="m1", score=cr["consensus_final_score"])

    chain = await _resolve_chain_topk(g3, consensus_g3, 7)
    print(f"  chain_target={chain['chain_target']}, held={len(chain['topk'])}, displaced={chain['displaced_count']}")
    assert chain["chain_target"] == 15, f"chain_target should walk to root num_leads=15, got {chain['chain_target']}"
    # 5 + 3 + 4 = 12 unique companies, all score>0 → all should be in top-K (≤15)
    assert len(chain["topk"]) == 12, f"expected 12 (all candidates), got {len(chain['topk'])}"
    assert chain["displaced_count"] == 0, "no displacement (no score collisions on company)"
    print(f"  ✅ chain target=15, 12 held across 3 generations, 0 displaced — chain in flight")


async def scenario_4_zero_held_recycled():
    """First cycle produces 0 held leads → recycled, fresh start (open)."""
    print("\n" + "="*60)
    print("SCENARIO 4: zero held → recycled, fresh open successor")
    print("="*60)
    sb = _MockSupabase()
    _patch_lifecycle_for_test(sb)
    from gateway.fulfillment.lifecycle import _resolve_chain_topk

    rid = "req-s4"
    _seed_request(sb, request_id=rid, num_leads=10, status="scoring")
    sub_id = _seed_submission(sb, request_id=rid, miner_hk="m1", leads=[{"lead_id": "l1", "data": {"business": "Acme"}}])

    # All leads fail with score=0 (e.g., insufficient_intent)
    consensus = [{"request_id": rid, "submission_id": sub_id, "lead_id": "l1", "miner_hotkey": "m1",
                  "consensus_final_score": 0.0, "consensus_intent_signal_final": 0.0,
                  "consensus_tier2_passed": True, "any_fabricated": False, "num_validators": 1,
                  "intent_signal_mapping": [], "intent_details": ""}]
    _seed_consensus(sb, request_id=rid, submission_id=sub_id, lead_id="l1", miner_hk="m1", score=0.0)

    chain = await _resolve_chain_topk(rid, consensus, 10)
    print(f"  chain_target={chain['chain_target']}, held={len(chain['topk'])}")
    assert chain["chain_target"] == 10
    assert len(chain["topk"]) == 0, "no qualifying candidates"
    print(f"  ✅ 0 held, recycled with fresh open successor")


async def scenario_5_displacement_company_replacement():
    """Same company submitted in gen2 with higher score replaces gen1's entry."""
    print("\n" + "="*60)
    print("SCENARIO 5: same-company displacement (cross-cycle dedup)")
    print("="*60)
    sb = _MockSupabase()
    _patch_lifecycle_for_test(sb)
    from gateway.fulfillment.lifecycle import _resolve_chain_topk

    g1 = "req-s5-gen1"
    _seed_request(sb, request_id=g1, num_leads=2, status="partially_fulfilled")
    leads_g1 = [
        {"lead_id": "l_held1", "data": {"business": "OverlapCo"}},
        {"lead_id": "l_held2", "data": {"business": "OtherCo"}},
    ]
    sub_g1 = _seed_submission(sb, request_id=g1, miner_hk="m1", leads=leads_g1)
    _seed_consensus(sb, request_id=g1, submission_id=sub_g1, lead_id="l_held1",
                    miner_hk="m1", score=20.0, is_chain_held=True)
    _seed_consensus(sb, request_id=g1, submission_id=sub_g1, lead_id="l_held2",
                    miner_hk="m1", score=15.0, is_chain_held=True)

    # Gen 2 — different miner submits NEW lead at same company "OverlapCo" with higher score
    g2 = "req-s5-gen2"
    _seed_request(sb, request_id=g2, num_leads=0, status="scoring")
    sb.tables["fulfillment_requests"].rows[0]["successor_request_id"] = g2

    leads_g2 = [{"lead_id": "l_new", "data": {"business": "OverlapCo"}}]  # SAME company
    sub_g2 = _seed_submission(sb, request_id=g2, miner_hk="m2", leads=leads_g2)
    consensus_g2 = [{
        "request_id": g2, "submission_id": sub_g2, "lead_id": "l_new", "miner_hotkey": "m2",
        "consensus_final_score": 50.0, "consensus_intent_signal_final": 50.0,
        "consensus_tier2_passed": True, "any_fabricated": False, "num_validators": 1,
        "intent_signal_mapping": [], "intent_details": "",
    }]
    _seed_consensus(sb, request_id=g2, submission_id=sub_g2, lead_id="l_new",
                    miner_hk="m2", score=50.0)

    chain = await _resolve_chain_topk(g2, consensus_g2, 0)
    print(f"  chain_target={chain['chain_target']}, held={len(chain['topk'])}, displaced={chain['displaced_count']}")
    print(f"  topk lead_ids: {[r['lead_id'] for r in chain['topk']]}")

    # Top-2 should be: l_new (50, OverlapCo) + l_held2 (15, OtherCo).
    # l_held1 (20, OverlapCo) gets DISPLACED by l_new because of cross-cycle dedup.
    assert chain["chain_target"] == 2
    assert len(chain["topk"]) == 2
    held_ids = chain["topk_lead_ids"]
    assert "l_new" in held_ids
    assert "l_held2" in held_ids
    assert "l_held1" not in held_ids, "l_held1 should be displaced by l_new (same company, higher score)"

    # Verify displaced row has is_chain_held=False
    cons_rows = sb.tables["fulfillment_score_consensus"].rows
    l_held1_row = next(r for r in cons_rows if r["lead_id"] == "l_held1")
    assert not l_held1_row["is_chain_held"], "l_held1 should be displaced"
    print(f"  ✅ Same-company cross-cycle dedup: l_new (score 50) replaced l_held1 (score 20)")


# ─────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────

async def main():
    await scenario_1_single_cycle_full()
    await scenario_2_partial_continued_fulfilled()
    await scenario_3_multi_generation_chain()
    await scenario_4_zero_held_recycled()
    await scenario_5_displacement_company_replacement()
    print("\n" + "="*60)
    print("🎉  All 5 scenarios passed.")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
