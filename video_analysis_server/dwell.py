"""Match vehicles across two analysis jobs and report how long they stayed.

Two vehicle jobs - an entrance camera and an exit camera, or the same camera on
two recordings - are joined on the licence plate. The first job's ENTRY is when
the car arrived, the second job's ENTRY is when it left, and the difference is
the dwell time.

    python dwell.py <job_dir_A> <job_dir_B> \
        --start-a 2026-09-17T09:00:00 --start-b 2026-09-17T09:00:00

THE TIME BASE MATTERS. Every time inside a result file is seconds from the start
of THAT video, so "entry_time 73.2" in two different files are not comparable on
their own. Pass each recording's wall-clock start with --start-a / --start-b and
the dwell times come out in real time. Leave them off and the tool still runs,
but it says so and reports the raw video-time difference, which is only
meaningful if both recordings started at the same instant.

Plate matching reuses the project's own rules: a reading whose Hangul syllable
could not be resolved is stored as e.g. "85?0527", and detect_car._completes()
decides whether that is the same car as "85아0527". Nothing is matched on a
guessed character beyond that unless --fuzzy is given, and anything matched that
way is flagged in the output.

Standalone utility - the server never imports it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import detect_car as alpr

UNKNOWN = alpr.UNKNOWN


def load_vehicles(path):
    """Accept a job directory, a vehicles.json or a summary.json."""
    if os.path.isdir(path):
        for name in ("vehicles.json", "summary.json"):
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(f"no vehicles.json in {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "vehicles" not in data:
        raise ValueError(f"{path} is not a vehicle result (no 'vehicles' key)")
    return data.get("job_id"), data["vehicles"], path


def same_plate(a, b, fuzzy=False):
    """(matched, how). Exact, then unresolved-syllable, then optional fuzzy."""
    if a == b:
        return True, "exact"
    # "85?0527" vs "85아0527" - the project's own rule for a syllable the OCR
    # could not resolve on either side
    if UNKNOWN in a and alpr._completes(a, b):
        return True, "syllable"
    if UNKNOWN in b and alpr._completes(b, a):
        return True, "syllable"
    if fuzzy and len(a) == len(b):
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        if len(diff) == 1 and all(c.isdigit() for c in diff[0]):
            return True, "fuzzy-1digit"
    return False, ""


def parse_start(value, label):
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise SystemExit(f"--start-{label} is not an ISO timestamp: {value!r}\n"
                         f"  e.g. 2026-09-17T09:00:00")


def fmt_hms(seconds):
    if seconds is None:
        return "-"
    sign = "-" if seconds < 0 else ""
    s = int(abs(seconds))
    return f"{sign}{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_a", help="arrival job: directory or vehicles.json")
    ap.add_argument("job_b", help="departure job: directory or vehicles.json")
    ap.add_argument("--start-a", help="wall-clock start of recording A (ISO)")
    ap.add_argument("--start-b", help="wall-clock start of recording B (ISO)")
    ap.add_argument("--fuzzy", action="store_true",
                    help="also match plates differing by a single digit (flagged)")
    ap.add_argument("--min-vote", type=float, default=0.0,
                    help="ignore plates whose winning reading is below this vote share")
    ap.add_argument("--json", help="write the joined result here")
    args = ap.parse_args(argv)

    job_a, veh_a, path_a = load_vehicles(args.job_a)
    job_b, veh_b, path_b = load_vehicles(args.job_b)
    start_a = parse_start(args.start_a, "a")
    start_b = parse_start(args.start_b, "b")
    absolute = start_a is not None and start_b is not None

    def keep(v):
        return (v.get("entry_time") is not None
                and (v.get("vote_share") or 0) >= args.min_vote)

    arrivals = [v for v in veh_a if keep(v)]
    departures = [v for v in veh_b if keep(v)]

    print(f"A (arrival)   {job_a}  {len(arrivals)} plates   {path_a}")
    print(f"B (departure) {job_b}  {len(departures)} plates   {path_b}")
    if absolute:
        print(f"time base     A={start_a.isoformat()}  B={start_b.isoformat()}")
    else:
        print("time base     NOT GIVEN - dwell is the raw video-time difference,")
        print("              which only means something if both recordings started together.")
    print()

    used_b = set()
    matched, ambiguous = [], []
    for a in arrivals:
        hits = []
        for i, b in enumerate(departures):
            if i in used_b:
                continue
            ok, how = same_plate(a["plate_number"], b["plate_number"], args.fuzzy)
            if ok:
                hits.append((i, b, how))
        if not hits:
            continue
        if len(hits) > 1:
            # the same plate seen twice in B - take the first departure after
            # the arrival, and say that it was ambiguous
            ambiguous.append(a["plate_number"])
        i, b, how = sorted(hits, key=lambda h: h[1]["entry_time"])[0]
        used_b.add(i)

        if absolute:
            t_in = start_a + timedelta(seconds=a["entry_time"])
            t_out = start_b + timedelta(seconds=b["entry_time"])
            dwell = (t_out - t_in).total_seconds()
        else:
            t_in = t_out = None
            dwell = b["entry_time"] - a["entry_time"]

        matched.append({
            "plate_number": b["plate_number"] if UNKNOWN in a["plate_number"] else a["plate_number"],
            "plate_a": a["plate_number"],
            "plate_b": b["plate_number"],
            "match": how,
            "arrival_video_time": a["entry_time"],
            "departure_video_time": b["entry_time"],
            "arrival_at": t_in.isoformat() if t_in else None,
            "departure_at": t_out.isoformat() if t_out else None,
            "dwell_seconds": round(dwell, 1),
            "vehicle_class": a.get("vehicle_class") or b.get("vehicle_class"),
            "reads_a": a.get("reads"), "reads_b": b.get("reads"),
        })

    matched.sort(key=lambda m: m["arrival_video_time"])
    only_a = [v for v in arrivals
              if not any(m["plate_a"] == v["plate_number"] for m in matched)]
    only_b = [b for i, b in enumerate(departures) if i not in used_b]

    print(f"{'번호판':<14}{'도착':>10}{'출발':>10}{'체류':>10}  매칭")
    for m in matched:
        print(f"{m['plate_number']:<14}"
              f"{fmt_hms(m['arrival_video_time']):>10}"
              f"{fmt_hms(m['departure_video_time']):>10}"
              f"{fmt_hms(m['dwell_seconds']):>10}  {m['match']}")
    if not matched:
        print("  (양쪽에서 함께 발견된 번호판 없음)")

    negative = [m for m in matched if m["dwell_seconds"] < 0]
    print()
    print(f"매칭 {len(matched)}대 | A에만 {len(only_a)}대 | B에만 {len(only_b)}대")
    if matched:
        d = sorted(m["dwell_seconds"] for m in matched)
        mid = d[len(d) // 2]
        print(f"체류시간  최소 {fmt_hms(d[0])}  중앙 {fmt_hms(mid)}  최대 {fmt_hms(d[-1])}")
    if negative:
        print(f"!! 체류시간이 음수인 차량 {len(negative)}대 - 시간 기준(--start-a/-b)을 "
              f"확인하세요")
    if ambiguous:
        print(f"!! B에서 같은 번호판이 여러 번 나타난 차량 {len(ambiguous)}대: "
              f"{', '.join(ambiguous[:5])}")
    fuzzy_used = [m for m in matched if m["match"] == "fuzzy-1digit"]
    if fuzzy_used:
        print(f"!! 한 자리 차이로 추정 매칭한 차량 {len(fuzzy_used)}대 - 확인 필요")

    if args.json:
        payload = {
            "job_a": job_a, "job_b": job_b,
            "start_a": args.start_a, "start_b": args.start_b,
            "absolute_time": absolute,
            "matched": matched,
            "only_in_a": [v["plate_number"] for v in only_a],
            "only_in_b": [v["plate_number"] for v in only_b],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
