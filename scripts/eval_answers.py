"""최종 답변 품질 평가 러너 — docs/design/llm-answer-evaluation-plan.md §3·§7·§8.

생성(deep 루프로 질의 실행·캡처) → 채점(결정론 M1·M4·M5·M6·M8 + LLM M2·M3·M4·M6·M7) → 리포트.
**`--dry-run`이 아니면 실제 OpenAI API를 호출한다(비용 발생).** 1회 비용은 `CostLedger`가 호출 전에
검사해 상한(기본 $8, `EVAL_MAX_COST_USD`)을 넘지 않게 하고, 넘을 상황이면 남은 작업을 건너뛰고
부분 결과를 저장한다. 결과 머리에는 이 평가의 한계(층 (b)만 본다, §1.4)를 찍는다.
deepeval 텔레메트리는 `auto_stock.evaluation` 패키지 import 시점에 꺼진다(§9).

Run from the project root (WSL venv 예: .venv/bin/python):
    python scripts/eval_answers.py --dry-run                              # 비용만 추정(호출 없음)
    python scripts/eval_answers.py --queries Q1 --metrics M2              # Phase 0 스모크
    python scripts/eval_answers.py --record data/eval/fixtures/base.json  # 라이브 실행 + 도구 결과 녹화
    python scripts/eval_answers.py --replay data/eval/fixtures/base.json --repeats 3 --baseline data/eval/baseline.json
    python scripts/eval_answers.py --answers data/eval/answers_X.json --mutations   # 채점·변이만 다시
결과: data/eval/answers_<run_id>.json, data/eval/scores_<run_id>.json(+ 터미널 요약). 키는 출력하지 않는다.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessage

from auto_stock.evaluation.answer_capture import DEFAULT_CONTEXT_LIMIT, CapturedTurn, ToolCallSink, main_visible
from auto_stock.evaluation.budget import BudgetExceeded, BudgetLimits, CostLedger
from auto_stock.evaluation.faults import describe, fault_patchers, injection_patchers, new_marker
from auto_stock.evaluation.golden import ALL_CASES, ALL_METRICS, GoldenCase, cases_by_id
from auto_stock.evaluation.judge import DEFAULT_JUDGE_MODEL, BudgetedOpenAIJudge
from auto_stock.evaluation.mutations import MUTATION_KINDS
from auto_stock.evaluation.numeric_grounding import cross_turn_conflicts
from auto_stock.evaluation.pricing import cost_usd, price_for
from auto_stock.evaluation.replay import FixtureRecorder, ReplayStore, replay_patchers
from auto_stock.evaluation.report import (
    compare_to_baseline,
    mutation_detection_rates,
    summarize,
    summary_lines,
    turn_status,
)
from auto_stock.evaluation.scoring import JUDGE_BUDGET_SKIP, MutationDetection, TurnScore, score_mutation, score_turn
from auto_stock.evaluation.tool_patching import applied

from auto_stock.chat_agent.credentials import DEFAULT_MODEL, MAX_LLM_CALLS_PER_QUERY
from auto_stock.chat_agent.deep_loop import (
    DeepChatSession,
    check_tracing_allowed,
    make_chat_model,
    run_turn_deep,
    verify_exposure,
)

sys.path.insert(0, str(Path(__file__).parent))
import eval_main_agent as base_eval  # 공유 리소스 빌더·배치 스텁·호출 상한 재사용

OUT_DIR = Path("data/eval")
JUDGED_METRICS = frozenset({"M2", "M3", "M4", "M6", "M7"})
# 생성 비용 추정 가정(§7.2) — 서브에이전트·LLM 분석기 내부 호출은 토큰이 보이지 않아 호출 수 × 이 값으로 잡는다.
SUBAGENT_CALL_TOKENS = (8_000, 1_000)
MAIN_CALL_TOKENS_FALLBACK = (8_000, 1_000)  # 응답에 usage가 없을 때만
# --dry-run 추정 가정(§7.2) — Phase 0 실측 후 갱신.
DRY_RUN_MAIN_CALLS_PER_TURN = 8
DRY_RUN_JUDGE_TOKENS = (5_000, 500)
DRY_RUN_JUDGE_CALLS = {"M2": 3, "M3": 3, "M4": 2, "M6": 2, "M7": 2}


class GenerationStopped(RuntimeError):
    """생성 단계 LLM 호출 상한(`--max-llm-calls`) 도달."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM answer-quality evaluation (DeepEval + deterministic checks)")
    parser.add_argument("--queries", help="쉼표로 구분한 질의 ID(예: Q1,F1). 기본: --sets 전체")
    parser.add_argument("--sets", default="regression,fault")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--metrics", default=",".join(ALL_METRICS))
    parser.add_argument("--judge-model", default=os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--max-llm-calls", type=int, default=300, help="생성 단계 LLM 호출 상한(§7.1)")
    parser.add_argument("--context-limit", type=int, default=DEFAULT_CONTEXT_LIMIT)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--record", type=Path, help="라이브 실행하며 도구 결과를 이 픽스처로 저장")
    source.add_argument("--replay", type=Path, help="이 픽스처의 도구 결과로 재생 실행")
    source.add_argument("--answers", type=Path, help="생성 없이 저장된 answers 파일을 다시 채점")
    parser.add_argument("--mutations", action="store_true", help="변이 테스트로 지표 탐지율 측정(§8.4)")
    parser.add_argument("--mutation-limit", type=int, default=5, help="변이 테스트에 쓸 정상 답변 턴 수")
    parser.add_argument("--baseline", type=Path, help="비교할 기준선 scores/baseline 파일")
    parser.add_argument("--save-baseline", type=Path, help="이번 채점 결과를 기준선으로 저장")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 비용만 추정")
    return parser.parse_args()


def select_cases(args: argparse.Namespace) -> list[GoldenCase]:
    if args.queries:
        by_id = cases_by_id()
        unknown = [q for q in args.queries.split(",") if q not in by_id]
        if unknown:
            raise SystemExit(f"unknown query ids: {unknown}")
        return [by_id[q] for q in args.queries.split(",")]
    sets = set(args.sets.split(","))
    return [case for case in ALL_CASES if case.set_name in sets]


def parse_metrics(raw: str) -> frozenset[str]:
    metrics = frozenset(raw.split(","))
    unknown = metrics - set(ALL_METRICS)
    if unknown:
        raise SystemExit(f"unknown metrics: {sorted(unknown)}")
    return metrics


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def estimate(cases: Sequence[GoldenCase], repeats: int, metrics: frozenset[str], agent_model: str, judge_model: str) -> dict:
    turns = sum(len(case.turns) for case in cases) * repeats
    generation = turns * DRY_RUN_MAIN_CALLS_PER_TURN * cost_usd(agent_model, *MAIN_CALL_TOKENS_FALLBACK)
    judge_calls = turns * sum(DRY_RUN_JUDGE_CALLS[m] for m in metrics & JUDGED_METRICS)
    judge = judge_calls * cost_usd(judge_model, *DRY_RUN_JUDGE_TOKENS)
    return {"turns": turns, "generation_usd": round(generation, 4), "judge_calls": judge_calls, "judge_usd": round(judge, 4)}


def _usage_tokens(messages: Sequence, main_calls: int) -> tuple[int, int]:
    usages = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
    if not usages:
        return main_calls * MAIN_CALL_TOKENS_FALLBACK[0], main_calls * MAIN_CALL_TOKENS_FALLBACK[1]
    return sum(u.get("input_tokens", 0) for u in usages), sum(u.get("output_tokens", 0) for u in usages)


def max_turn_cost(agent_model: str) -> float:
    """한 턴이 쓸 수 있는 최대 생성 비용(메인 + 서브 호출 상한 × 추정 토큰) — 생성 전 상한 검사용."""
    return (MAX_LLM_CALLS_PER_QUERY + 5) * cost_usd(agent_model, *SUBAGENT_CALL_TOKENS)


def _charge_generation(state: dict, new_messages: Sequence, main_calls: int, sub_calls: int) -> tuple[int, int]:
    agent_model = state["agent_model"]
    input_tokens, output_tokens = _usage_tokens(new_messages, main_calls)
    cost = cost_usd(agent_model, input_tokens, output_tokens) + sub_calls * cost_usd(agent_model, *SUBAGENT_CALL_TOKENS)
    state["ledger"].add_generation(cost)
    state["cap"].used += main_calls + sub_calls
    return input_tokens, output_tokens


def measured_usage(by_model: dict) -> dict:
    """`UsageMetadataCallbackHandler.usage_metadata`(모델별 누적) → 입출력 토큰 합."""
    totals = {"input_tokens": 0, "output_tokens": 0}
    for usage in by_model.values():
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
    return totals


def run_turn(state: dict, case: GoldenCase, turn_index: int, repeat: int, history: tuple, marker: str | None) -> CapturedTurn:
    session = state["session"]
    if state["cap"].exhausted(reserve=MAX_LLM_CALLS_PER_QUERY + 5):
        raise GenerationStopped(f"LLM call cap {state['cap'].cap} reached")
    state["ledger"].check_generation(max_turn_cost(state["agent_model"]))
    context = base_eval._context(state["shared"], base_eval._StubCoordinator(), "deep")
    sink = ToolCallSink()

    def on_event(event: dict) -> None:
        sink.on_event(event)
        if state["recorder"] is not None and state["recording_enabled"]:
            state["recorder"].on_event(event)

    question = case.turns[turn_index].question
    before, calls_before, started = len(session.messages), session.model_calls, time.perf_counter()
    # 실측 토큰(메인 + 서브에이전트 모든 모델 호출) — 비용 상한 장부는 보수적 추정을 그대로 쓰고, 이건 모드 비교용
    # (docs/design/stock-analyst-subagent-migration.md §8: tool vs subagent 비용 +20% 기준).
    with get_usage_metadata_callback() as usage_cb:
        answer = run_turn_deep(session, context, question, on_event=on_event)
    main_calls, sub_calls = session.model_calls - calls_before, context.budget.llm_calls_made
    new_messages = session.messages[before:] if len(session.messages) >= before else session.messages[-1:]
    input_tokens, output_tokens = _charge_generation(state, new_messages, main_calls, sub_calls)
    usage = {"main_calls": main_calls, "sub_llm_calls": sub_calls, "input_tokens": input_tokens,
             "output_tokens": output_tokens, "latency_s": round(time.perf_counter() - started, 2),
             "measured": measured_usage(usage_cb.usage_metadata)}
    return CapturedTurn(
        case_id=f"{case.query_id}#t{turn_index}#r{repeat}", query_id=case.query_id, turn_index=turn_index, repeat=repeat,
        question=question, answer=answer, tool_calls=sink.records, history_tool_calls=history,
        fallback=answer in base_eval.FALLBACKS, injected_fault=describe(case.fault),
        replay_fixture=state["fixture_label"], injection_marker=marker, usage=usage,
    )


def run_conversation(state: dict, case: GoldenCase, repeat: int, turns: list[CapturedTurn]) -> None:
    """턴이 끝날 때마다 `turns`에 바로 붙인다 — 대화 중간에 상한으로 멈춰도 이미 비용을 쓴 턴은 남는다."""
    marker = new_marker() if case.injection else None
    # 실패 주입·인젝션 스텁 결과는 녹화하지 않는다 — 픽스처에 섞이면 다른 케이스 재생이 오염된다.
    state["recording_enabled"] = case.fault is None and not case.injection
    if state["store"] is not None:
        state["store"].reset_cursor()
    # 컨텍스트는 왼쪽부터 차례로 진입하므로, 실패 주입은 재생 패치 "위에" 걸린다 — stale 주입이
    # 재생된 결과의 날짜를 옮기도록(fault_patchers는 그 시점에 걸려 있는 구현을 감싼다).
    with (
        applied(replay_patchers(state["store"]) if state["store"] is not None else []),
        applied(fault_patchers(case.fault) if case.fault else []),
        applied(injection_patchers(marker) if marker else []),
    ):
        state["session"] = DeepChatSession(model=make_chat_model(state["shared"]["agent_config"]))
        history: tuple = ()
        for turn_index in range(len(case.turns)):
            captured = run_turn(state, case, turn_index, repeat, history, marker)
            turns.append(captured)
            history = (*history, *captured.tool_calls)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _generation_state(args: argparse.Namespace, ledger: CostLedger) -> dict:
    shared = base_eval._build_shared()  # load_llm_config가 .env를 읽으므로 트레이싱 검사는 그 뒤에
    check_tracing_allowed()
    verify_exposure()
    return {
        "shared": shared, "ledger": ledger, "cap": base_eval._CallCap(args.max_llm_calls),
        "store": ReplayStore.load(args.replay) if args.replay else None,
        "recorder": FixtureRecorder() if args.record else None, "recording_enabled": True,
        "agent_model": shared["agent_config"].model, "fixture_label": str(args.replay) if args.replay else None,
    }


def generate(args: argparse.Namespace, cases: Sequence[GoldenCase], ledger: CostLedger, meta: dict) -> list[CapturedTurn]:
    state = _generation_state(args, ledger)
    captured: list[CapturedTurn] = []
    skipped: list[str] = []
    out = OUT_DIR / f"answers_{meta['run_id']}.json"
    for case in cases:
        for repeat in range(1, args.repeats + 1):
            if skipped:  # 상한에 한 번 닿으면 남은 대화는 시작하지 않는다(건너뛴 목록은 아래에서 저장)
                skipped.append(f"{case.query_id}#r{repeat}")
            else:
                try:
                    run_conversation(state, case, repeat, captured)
                except (BudgetExceeded, GenerationStopped) as exc:
                    skipped.append(f"{case.query_id}#r{repeat} ({exc})")
            misses = state["store"].misses if state["store"] else []
            # 대화마다 저장 — 중간에 멈춰도 끝난 대화는 남는다.
            write_json(out, {"meta": {**meta, "generation_skipped": skipped, "replay_misses": misses},
                             "cases": [turn.to_dict() for turn in captured]})
            print(f"[gen ${ledger.generation_usd:.2f}] {case.query_id} #{repeat} turns={len(captured)}", flush=True)
    if state["recorder"] is not None:
        recorder_meta = {"run_id": meta["run_id"], "git_commit": meta["git_commit"], "reference_date": meta["reference_date"]}
        state["recorder"].save(args.record, meta=recorder_meta)
    return captured


def reference_date(args: argparse.Namespace) -> str:
    """시점 검사(M7 "오래됨", M8 stale_as_now)의 기준일 — 재생은 녹화한 날, 재채점은 캡처한 날, 라이브는 오늘.
    오래된 픽스처를 오늘 기준으로 채점하면 모든 시세가 "오래된 데이터"로 오판된다."""
    source = args.replay or args.answers
    if source is not None:
        recorded = json.loads(Path(source).read_text(encoding="utf-8")).get("meta", {}).get("reference_date")
        if recorded:
            return recorded
        print(f"warning: {source} has no reference_date — using today for staleness checks", flush=True)
    return date.today().isoformat()


def load_answers(path: Path) -> tuple[list[CapturedTurn], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [CapturedTurn.from_dict(item) for item in payload["cases"]], payload.get("meta", {})


def fixture_sha256(path: Path) -> str:
    """재생 픽스처의 내용 해시 — 기준선 비교는 같은 해시끼리만 유효하다(report.comparability_problem)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_counts(cases: Sequence[GoldenCase], repeats: int) -> dict[str, list[int]]:
    """질의별 (턴 수, 반복 수) — 생성 단계에서 빠진 회차·턴을 pass^k에서 판정 불가로 세기 위해 남긴다."""
    return {case.query_id: [len(case.turns), repeats] for case in cases}


def build_meta(args: argparse.Namespace, cases: Sequence[GoldenCase], metrics: frozenset[str], agent_model: str) -> dict:
    """재채점(--answers)은 생성 실행의 재생 정보·기대 회차를 이어받는다 — 그래야 기준선 비교 자격이 유지된다."""
    meta = {
        "run_id": f"{datetime.now():%Y%m%d_%H%M%S}", "git_commit": git_commit(), "agent_model": agent_model,
        "judge_model": args.judge_model, "loop": "deep", "metrics": sorted(metrics),
        "replay_fixture": str(args.replay) if args.replay else None,
        "replay_fixture_sha256": fixture_sha256(args.replay) if args.replay else None,
        "expected": expected_counts(cases, args.repeats),
        "reference_date": reference_date(args),
    }
    if args.answers:
        source = json.loads(Path(args.answers).read_text(encoding="utf-8")).get("meta", {})
        meta.update({key: source.get(key) for key in ("replay_fixture", "replay_fixture_sha256", "expected")})
        meta["rescored_from"] = str(args.answers)
    return meta


def add_consistency(scores: list[TurnScore], captured: Sequence[CapturedTurn]) -> list[TurnScore]:
    """멀티턴 대화마다 턴 간 수치 일관성(§5.2)을 대화의 마지막 턴 결과에 붙인다."""
    conversations: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, turn in enumerate(captured):
        conversations[(turn.query_id, turn.repeat)].append(index)
    updated = list(scores)
    for indexes in conversations.values():
        if len(indexes) > 1:
            result = cross_turn_conflicts([(captured[i].answer, main_visible(captured[i].tool_calls)) for i in indexes])
            updated[indexes[-1]] = updated[indexes[-1]].with_result(result)
    return updated


def score_all(
    captured: Sequence[CapturedTurn], judge, metrics: frozenset[str], context_limit: int, today: date
) -> list[TurnScore]:
    by_id = cases_by_id()
    scores: list[TurnScore] = []
    for turn in captured:
        score = score_turn(turn, by_id[turn.query_id], judge, today, context_limit, metrics)
        if score.skipped_reason == JUDGE_BUDGET_SKIP and judge is not None:
            print("judge budget reached — remaining turns get deterministic checks only", flush=True)
            judge = None
        scores.append(score)
    return add_consistency(scores, captured)


def run_mutations(
    captured: Sequence[CapturedTurn], scores: Sequence[TurnScore], judge, limit: int, today: date
) -> list[MutationDetection]:
    """확정 통과로 채점된 턴만 골라(실패 주입·폴백·판정 불가 제외) 변이를 만든다 — 원래 답이 실패이거나
    판정 불가면 변이를 잡았는지 알 수 없다. 채점 예산이 바닥난 뒤의 변이는 `skipped`로 탐지율에서 빠진다."""
    by_id = cases_by_id()
    normal = [
        turn
        for turn, score in zip(captured, scores, strict=True)
        if turn.injected_fault is None and not turn.fallback and turn_status(score) == "pass"
    ][:limit]
    return [score_mutation(turn, by_id[turn.query_id], kind, judge, today) for turn in normal for kind in MUTATION_KINDS]


def load_scores(path: Path) -> tuple[list[TurnScore], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [TurnScore.from_dict(item) for item in payload["turns"]], payload.get("meta", {})


def save_baseline(path: Path, meta: dict, turns: list[dict]) -> bool:
    """재생 실행만 기준선이 될 수 있다 — 라이브 실행을 기준선으로 두면 이후 어떤 비교도 유효하지 않다."""
    if not meta.get("replay_fixture_sha256"):
        print("error: baseline not saved — only --replay runs (or --answers from a replay run) can be baselines", flush=True)
        return False
    write_json(path, {"meta": meta, "turns": turns})
    return True


def _dry_run(cases: Sequence[GoldenCase], args: argparse.Namespace, metrics: frozenset[str], agent_model: str, ledger: CostLedger) -> None:
    est = estimate(cases, args.repeats, metrics, agent_model, args.judge_model)
    total = round(est["generation_usd"] + est["judge_usd"], 4)
    print(json.dumps({**est, "total_usd": total, "limits": ledger.snapshot()["limits"]}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if total > ledger.limits.total_usd else 0)


def _row(row: MutationDetection) -> dict:
    return {name: getattr(row, name) for name in row.__dataclass_fields__}


def main() -> None:
    args = parse_args()
    cases, metrics = select_cases(args), parse_metrics(args.metrics)
    load_dotenv()
    ledger = CostLedger(BudgetLimits.from_env())
    agent_model = os.environ.get("CHAT_AGENT_OPENAI_MODEL", DEFAULT_MODEL)
    price_for(agent_model)  # 단가 없는 모델이면 시작 전에 실패(상한을 강제할 수 없음)
    price_for(args.judge_model)
    if args.dry_run:
        _dry_run(cases, args, metrics, agent_model, ledger)

    check_tracing_allowed()
    meta = build_meta(args, cases, metrics, agent_model)
    today = date.fromisoformat(meta["reference_date"])
    captured = load_answers(args.answers)[0] if args.answers else generate(args, cases, ledger, meta)
    judge = BudgetedOpenAIJudge(ledger, args.judge_model) if metrics & JUDGED_METRICS else None
    scores = score_all(captured, judge, metrics, args.context_limit, today)
    mutations = run_mutations(captured, scores, judge, args.mutation_limit, today) if args.mutations else []
    expected = {qid: tuple(counts) for qid, counts in (meta.get("expected") or {}).items()}
    summary, rates = summarize(scores, expected), mutation_detection_rates(mutations)
    comparison = None
    if args.baseline:
        baseline_turns, baseline_meta = load_scores(args.baseline)
        comparison = compare_to_baseline(scores, baseline_turns, meta, baseline_meta)
    payload = {"meta": meta, "cost": ledger.snapshot(), "summary": summary, "comparison": comparison,
               "mutations": {"rates": rates, "rows": [_row(m) for m in mutations]},
               "turns": [score.to_dict() for score in scores]}
    out = OUT_DIR / f"scores_{meta['run_id']}.json"
    write_json(out, payload)
    if args.save_baseline:
        save_baseline(args.save_baseline, meta, payload["turns"])
    print("\n".join(["", "=== answer evaluation ===", *summary_lines(summary, payload["cost"], comparison, rates), f"results: {out}"]))


if __name__ == "__main__":
    main()
