#!/usr/bin/env python3
# truth_planner.py — Extract, score, dedupe, and operationalize "truths" (axioms).
#
# Examples:
#   python truth_planner.py input.txt
#   python truth_planner.py input.json --focus "internet,micro platforms" --relevance-threshold 0.12
#   python truth_planner.py input.txt --format md --min-sim 0.6 --emit-only-truths
#   python truth_planner.py input.txt --format json --max-truths 120
#   python truth_planner.py input.txt --contradict "sensor_outputs_data|sensor_does_not_output_data"
#
# Input formats:
#   1) INI-like with sections [purpose], [axioms], [something_else]...
#   2) JSON: {"purpose": "...", "axioms": ["...","..."], "sections": {"name":[...]}}
#
# Output:
#   - By default: a markdown "truth plan": truths (ranked), duplicates, contradictions, and actions.
#   - With --emit-only-truths: prints normalized truths only (txt or json).
#
# Key ideas:
#   - Truths are slugified (lowercase, spaces→_, punctuation stripped but . and - kept).
#   - Focus scoring boosts truths that contain given keywords (length-normalized).
#   - Near-duplicates found via token Jaccard similarity (threshold --min-sim).
#   - Contradictions: automatic (x vs not_x & "does_not_") + optional custom pairs via --contradict.
#
from __future__ import annotations
import argparse, json, os, re, sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Set
from collections import defaultdict, Counter
from math import log1p

SECTION_RE = re.compile(r'^\s*\[(?P<name>[^\]]+)\]\s*$')

# ---------------- Models ----------------

@dataclass
class Doc:
    purpose: str = ""
    axioms: List[str] = field(default_factory=list)           # raw lines
    sections: Dict[str, List[str]] = field(default_factory=dict)

# ---------------- Utils ----------------

def eprint(*a, **k): print(*a, file=sys.stderr, **k)

def read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    if not os.path.exists(path):
        sys.exit(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_input_any(path: str) -> Doc:
    raw = read_text(path).strip()

    # JSON first
    if raw.startswith("{") or raw.startswith("["):
        try:
            data = json.loads(raw)
            purpose = str(data.get("purpose", "") or "").strip()
            axioms = [str(x) for x in (data.get("axioms", []) or []) if str(x).strip()]
            sections: Dict[str, List[str]] = {}
            for k, v in (data.get("sections", {}) or {}).items():
                sections[k] = v if isinstance(v, list) else [str(v)]
            return Doc(purpose=purpose, axioms=axioms, sections=sections)
        except Exception as e:
            eprint(f"[WARN] JSON parse failed, falling back to INI-like: {e}")

    # INI-like parser
    lines = raw.splitlines()
    current = None
    buckets: Dict[str, List[str]] = defaultdict(list)
    purpose_buf: List[str] = []
    for rawln in lines:
        ln = rawln.rstrip("\n")
        m = SECTION_RE.match(ln)
        if m:
            current = m.group("name").strip()
            if current not in buckets:
                buckets[current] = []
            continue
        if ln.strip().startswith("#") or ln.strip().startswith(";"):
            continue
        if current is None:
            if ln.strip():
                purpose_buf.append(ln)
            continue
        buckets[current].append(ln)

    doc = Doc()
    for k in list(buckets.keys()):
        if k.lower() == "purpose":
            doc.purpose = "\n".join(buckets.pop(k)).strip()
            break
    if not doc.purpose and purpose_buf:
        doc.purpose = "\n".join(purpose_buf).strip()

    for k in list(buckets.keys()):
        if k.lower() == "axioms":
            ax: List[str] = []
            for ln in buckets.pop(k):
                s = ln.strip()
                if not s or s.startswith("#") or s.startswith(";"):
                    continue
                if s.startswith("- "):
                    s = s[2:].strip()
                if s:
                    ax.append(s)
            doc.axioms = ax
            break

    doc.sections = dict(buckets)
    return doc

def slugify(s: str) -> str:
    s = s.strip().lower().replace("→", "->")
    s = re.sub(r"[^\w.\- ]+", " ", s)   # keep word chars, dot, dash, space
    s = re.sub(r"\s+", "_", s.strip())
    return s

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

def dedup_stable(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def derive_truths(axioms: List[str]) -> List[str]:
    return dedup_stable(slugify(a) for a in axioms if str(a).strip())

def score_truths(truths: List[str], focus_terms: List[str]) -> Dict[str, float]:
    if not focus_terms:
        return {t: 1.0 for t in truths}
    vocab = {w for ft in focus_terms for w in tokenize(ft)}
    scores: Dict[str, float] = {}
    for t in truths:
        toks = tokenize(t)
        hits = sum(1 for tok in toks if tok in vocab)
        # length-normalized + mild log gain
        scores[t] = (hits / max(1, len(toks))) * (1.0 + 0.1 * log1p(len(toks)))
    return scores

def filter_truths(truths: List[str], focus: Optional[str], relevance_threshold: float, max_truths: Optional[int]) -> List[str]:
    if not truths:
        return []
    if not focus:
        ranked = truths[:]
    else:
        terms = [x.strip() for x in (focus or "").split(",") if x.strip()]
        scores = score_truths(truths, terms)
        ranked = sorted(truths, key=lambda t: scores.get(t, 0.0), reverse=True)
        ranked = [t for t in ranked if scores.get(t, 0.0) >= relevance_threshold]
    if max_truths is not None:
        ranked = ranked[:max_truths]
    return ranked

def jaccard(a: str, b: str) -> float:
    A, B = set(tokenize(a)), set(tokenize(b))
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def cluster_duplicates(truths: List[str], min_sim: float) -> List[List[str]]:
    # simple greedy clustering by similarity threshold
    unvisited: Set[str] = set(truths)
    clusters: List[List[str]] = []
    while unvisited:
        seed = min(unvisited)  # deterministic
        unvisited.remove(seed)
        cluster = [seed]
        to_add = []
        for t in list(unvisited):
            if jaccard(seed, t) >= min_sim:
                cluster.append(t)
                to_add.append(t)
        for t in to_add:
            unvisited.remove(t)
        clusters.append(sorted(cluster))
    # only return clusters that have >1 members as duplicates
    dups = [c for c in clusters if len(c) > 1]
    # also return singleton clusters? not needed for duplicates section
    return dups

def detect_contradictions(truths: List[str], custom_pairs: List[Tuple[str,str]]) -> List[Tuple[str, str, str]]:
    """
    Returns list of tuples: (type, a, b)
      type: 'auto_negation' | 'custom'
    Auto rules:
      - x vs not_x (prefix "not_" or "no_" or "does_not_" mid-phrases)
      - "_does_not_" phrasing vs positive counterpart if easy to spot
    """
    out: List[Tuple[str, str, str]] = []
    tset = set(truths)

    # auto: not_ / no_ prefix
    # normalized: try to strip leading not_/no_
    base_map: Dict[str, str] = {}
    for t in truths:
        if t.startswith("not_"):
            base_map[t] = t[len("not_"):]
        elif t.startswith("no_"):
            base_map[t] = t[len("no_"):]
        else:
            # try "does_not_" in the middle
            # e.g., sensor_does_not_output_data -> sensor_outputs_data (heuristic)
            if "does_not_" in t:
                base_map[t] = t.replace("does_not_", "")
    # check for base vs negated coexistence
    for neg, base in base_map.items():
        if base and base in tset:
            out.append(("auto_negation", base, neg))

    # custom explicit pairs (regex | regex)
    for a_pat, b_pat in custom_pairs:
        ra, rb = re.compile(a_pat, re.IGNORECASE), re.compile(b_pat, re.IGNORECASE)
        As = [t for t in truths if ra.search(t)]
        Bs = [t for t in truths if rb.search(t)]
        for A in As:
            for B in Bs:
                out.append(("custom", A, B))
    # dedup
    seen = set()
    uniq = []
    for k in out:
        key = tuple(sorted((k[1], k[2]))) + (k[0],)
        if key not in seen:
            seen.add(key)
            uniq.append(k)
    # deterministic order
    uniq.sort(key=lambda x: (x[0], x[1], x[2]))
    return uniq

# ---------------- Renderers ----------------

def render_txt_only(truths: List[str], as_json: bool) -> str:
    if as_json:
        return json.dumps(truths, ensure_ascii=False, indent=2)
    return "\n".join(truths)

def render_md_plan(
    truths_ranked: List[str],
    dups: List[List[str]],
    contradictions: List[Tuple[str, str, str]],
    purpose: str,
    kpis: List[str]
) -> str:
    out: List[str] = []
    out.append("# Truth Planner")
    out.append("")
    if purpose:
        out.append("## Purpose (verbatim)")
        out.append(purpose.strip())
        out.append("")
    out.append("## Truths (normalized & ranked)")
    if truths_ranked:
        for t in truths_ranked:
            out.append(f"- `{t}`")
    else:
        out.append("- —")
    out.append("")

    out.append("## Near-duplicates (clusters)")
    if dups:
        for i, cluster in enumerate(dups, 1):
            out.append(f"**Cluster {i}**")
            for t in cluster:
                out.append(f"- `{t}`")
            out.append("")
    else:
        out.append("_None detected at current similarity threshold._")
        out.append("")

    out.append("## Contradictions (auto + custom)")
    if contradictions:
        for typ, a, b in contradictions:
            out.append(f"- **{typ}**: `{a}`  ⇄  `{b}`")
    else:
        out.append("_None detected with current rules._")
    out.append("")

    out.append("## Hardening plan (do-this-next checklist)")
    out.append("Use the same template for each truth you intend to keep:")
    out.append("")
    out.append("1) **Glossary entry**: one-sentence definition; include scope and exclusions.")
    out.append("2) **Examples**: 2–3 positive examples; 1–2 counterexamples.")
    out.append("3) **Tests**: a repeatable check (yes/no) that the truth applies in context.")
    out.append("4) **SOP binding**: link to the SOP/checklist that consumes this truth.")
    out.append("5) **KPI mapping**: where this truth moves a metric (e.g., {})".format(", ".join(kpis) if kpis else "cycle_time, throughput, quality_signal"))
    out.append("6) **Change control**: who can edit, and under what process.")
    out.append("")
    out.append("> Tip: resolve duplicates by merging wording; resolve contradictions by choosing")
    out.append("> one canonical form and explicitly retiring the other (but keep a redirect).")
    out.append("")
    return "\n".join(out)

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Truth Planner — extract, score, dedupe, and operationalize truths.")
    ap.add_argument("input", help="Path to input (INI-like or JSON), or '-' for stdin.")
    ap.add_argument("--focus", type=str, default=None, help="Comma-separated keywords to boost (e.g., 'internet,micro platforms').")
    ap.add_argument("--relevance-threshold", type=float, default=0.0, help="Keep truths with score >= threshold (0..1 sensible).")
    ap.add_argument("--max-truths", type=int, default=None, help="Cap number of truths after filtering.")
    ap.add_argument("--min-sim", type=float, default=0.65, help="Jaccard similarity for near-duplicate clustering (0..1).")
    ap.add_argument("--format", choices=["md","txt","json"], default="md", help="Output format.")
    ap.add_argument("--emit-only-truths", action="store_true", help="Print only truths (txt/json) and exit.")
    ap.add_argument("--kpis", type=str, default="cycle_time,throughput,quality_signal", help="Comma-separated KPI names for planner section.")
    ap.add_argument("--contradict", action="append", default=[],
                    help="Custom contradiction pair as 'regexA|regexB'. Can be repeated.")

    args = ap.parse_args()

    # Parse input
    try:
        doc = parse_input_any(args.input)
    except Exception as e:
        sys.exit(f"[FATAL] Failed to parse input: {e}")

    # Derive truths
    truths_all = derive_truths(doc.axioms)
    truths_ranked = filter_truths(
        truths_all,
        focus=args.focus,
        relevance_threshold=float(args.relevance_threshold),
        max_truths=args.max_truths
    )
    if not truths_ranked and truths_all:
        eprint("[WARN] All truths filtered; emitting original set.")
        truths_ranked = truths_all[:]

    # Only truths?
    if args.emit_only_truths:
        as_json = (args.format == "json")
        print(render_txt_only(truths_ranked, as_json=as_json))
        return

    # Prepare planner outputs
    try:
        sim_threshold = max(0.0, min(1.0, float(args.min_sim)))
    except:
        sim_threshold = 0.65

    dups = cluster_duplicates(truths_ranked, sim_threshold)

    # custom contradictions
    pairs: List[Tuple[str,str]] = []
    for p in args.contradict:
        if "|" in p:
            a, b = p.split("|", 1)
            pairs.append((a.strip(), b.strip()))
        else:
            eprint(f"[WARN] Ignoring --contradict without '|': {p}")

    contradictions = detect_contradictions(truths_ranked, pairs)

    # Render
    if args.format == "json":
        obj = {
            "purpose": doc.purpose,
            "truths": truths_ranked,
            "duplicates": dups,
            "contradictions": [{"type": t, "a": a, "b": b} for (t, a, b) in contradictions],
            "kpis": [k.strip() for k in args.kpis.split(",") if k.strip()]
        }
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    elif args.format == "txt":
        # text with simple sections
        print("\n".join(truths_ranked))
        if dups:
            print("\n# Duplicates")
            for i, c in enumerate(dups, 1):
                print(f"cluster_{i}:" + ",".join(c))
        if contradictions:
            print("\n# Contradictions")
            for typ, a, b in contradictions:
                print(f"{typ}: {a} <> {b}")
    else:
        # markdown planner
        kpis = [k.strip() for k in args.kpis.split(",") if k.strip()]
        md = render_md_plan(truths_ranked, dups, contradictions, doc.purpose, kpis)
        print(md)

if __name__ == "__main__":
    main()
