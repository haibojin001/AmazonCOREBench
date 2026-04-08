import os
import json
import argparse
import re
from typing import List, Dict

from openai import OpenAI

from core_prompts import (
    SYNTHESIZING_PROMPT,
    REASONING_FEW_SHOT_EXAMPLE,
    REVIEW_FEW_SHOT_EXAMPLE,
    GENERATOR_TEMPLATE,
    OPTIMIZER_PROMPT,
)


OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
SYNTHESIZING_MODEL = "gpt-4o"
GENERATOR_MODEL    = "gpt-3.5-turbo"
OPTIMIZER_MODEL    = "gpt-4o"
MAX_ITER_DEFAULT   = 5

client = OpenAI(api_key=OPENAI_API_KEY)


# ──────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────
def load_products(path: str) -> List[Dict]:
    products = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                products.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return products


def flatten(raw: Dict) -> Dict:
    """Normalise a raw product record into a flat working dict."""
    desc = raw.get("short_description", "No description available.")
    if isinstance(desc, list):
        desc = " ".join(desc)
    return {
        "name":        raw.get("name", "Unknown Product"),
        "price":       raw.get("price", "N/A"),
        "rating":      raw.get("rating", "N/A"),
        "reviews":     raw.get("number_of_reviews", "N/A"),
        "description": desc,
        "url":         raw.get("source_url", "#"),
        "suffix":      "",
    }


def call_llm(model: str, prompt: str, temperature: float = 0.7) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def build_products_json(products: List[Dict]) -> str:
    items = []
    for p in products:
        desc = p["description"]
        if p.get("suffix"):
            desc = desc + "\n\n" + p["suffix"]
        items.append({
            "name":        p["name"],
            "price":       p["price"],
            "rating":      p["rating"],
            "description": desc,
        })
    return json.dumps(items, ensure_ascii=False, indent=2)


def get_ranking(query: str, products: List[Dict]) -> List[str]:
    prompt = SYNTHESIZING_PROMPT.format(
        query=query,
        products_json=build_products_json(products),
    )
    raw = call_llm(SYNTHESIZING_MODEL, prompt, temperature=0.0)

    # Try direct JSON parse
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass

    # Extract first JSON array
    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    print("[WARNING] Could not parse ranking. Raw output snippet:")
    print(raw[:400])
    return [p["name"] for p in products]


def find_rank(ranking: List[str], target_name: str) -> int:
    """Ask LLM which entry in the ranking corresponds to target_name. Returns 1-based rank."""
    numbered = "\n".join(f"  #{i+1}: {name}" for i, name in enumerate(ranking))
    prompt = (
        f"You are given a ranked list of products and a target product name.\n"
        f"Identify which rank number corresponds to the target product.\n\n"
        f"Target product: {target_name}\n\n"
        f"Ranked list:\n{numbered}\n\n"
        f"Reply with ONLY a single integer — the rank number (e.g. 3).\n"
        f"If the target product does not appear in the list at all, reply with {len(ranking) + 1}."
    )
    raw = call_llm(SYNTHESIZING_MODEL, prompt, temperature=0.0).strip()
    m = re.search(r'\d+', raw)
    if m:
        return int(m.group())
    return len(ranking) + 1


def generate_suffix(
    query: str,
    target_name: str,
    strategy: str,
    category: str = "Product",
) -> str:
    few_shot = (
        REASONING_FEW_SHOT_EXAMPLE
        if strategy == "reasoning"
        else REVIEW_FEW_SHOT_EXAMPLE
    )
    prompt = GENERATOR_TEMPLATE.format(
        few_shot_example=few_shot,
        target_name=target_name,
        question=query,
        category=category,
    )
    raw = call_llm(GENERATOR_MODEL, prompt, temperature=0.8)
    cut_markers = [
        "**Understanding", "**Explaining", "**Detailing", "**Providing",
        "### **", "After trying", "After comparing", "After testing",
        "I've tested", "I've found", "I'm breaking", "I'm analyzing",
        "I'm comparing", "I'm exploring",
    ]
    cut_pos = None
    for marker in cut_markers:
        idx = raw.find(marker)
        if idx != -1:
            if cut_pos is None or idx < cut_pos:
                cut_pos = idx

    if cut_pos is not None:
        suffix = raw[cut_pos:].strip()
    else:
        # Fallback: drop just the first line (the "Sure..." sentence)
        lines = raw.strip().splitlines()
        suffix = "\n".join(lines[1:]).strip()

    return suffix


def optimize_suffix(
    query: str,
    target_name: str,
    current_draft: str,
    current_ranking: List[str],
    current_pos: int,
) -> str:
    top_ranked = current_ranking[: current_pos - 1]
    top_ranked_str = ", ".join(top_ranked) if top_ranked else "none"
    ranking_str = "\n".join(
        f"  #{i+1}: {name}" for i, name in enumerate(current_ranking)
    )
    prompt = OPTIMIZER_PROMPT.format(
        query=query,
        target_name=target_name,
        current_ranking_str=ranking_str,
        current_pos=current_pos,
        top_ranked_items=top_ranked_str,
        current_draft=current_draft,
    )
    return call_llm(OPTIMIZER_MODEL, prompt, temperature=0.7)


def run_core(
    query: str,
    products: List[Dict],
    target_name: str,
    strategy: str = "review",
    category: str = "Product",
    max_iter: int = MAX_ITER_DEFAULT,
    verbose: bool = True,
) -> Dict:

    target = None
    for p in products:
        if target_name.lower() in p["name"].lower() or p["name"].lower() in target_name.lower():
            target = p
            break
    if target is None:
        names = [p["name"] for p in products]
        raise ValueError(f"Target '{target_name}' not found.\nAvailable:\n" + "\n".join(names))

    for p in products:
        p["suffix"] = ""
    baseline_ranking = get_ranking(query, products)
    baseline_pos = find_rank(baseline_ranking, target_name)

    if verbose:
        print("\n" + "=" * 64)
        print(f"  CORE  |  strategy={strategy.upper()}")
        print(f"  Target: {target_name}")
        print("=" * 64)
        print(f"  Baseline ranking : {baseline_ranking}")
        print(f"  Baseline position: #{baseline_pos}")
        print("=" * 64 + "\n")

    if verbose:
        print("[Generator] Calling LLM to produce initial suffix ...")
    current_suffix = generate_suffix(query, target["name"], strategy, category)
    if verbose:
        print(f"[Generator] Done — {len(current_suffix)} chars\n")
        print("--- Initial suffix (preview) ---")
        print(current_suffix[:600] + ("..." if len(current_suffix) > 600 else ""))
        print()

    current_ranking: List[str] = []
    current_pos = baseline_pos

    for iteration in range(1, max_iter + 1):
        # Attach suffix only to the target product
        for p in products:
            p["suffix"] = current_suffix if p["name"] == target["name"] else ""

        current_ranking = get_ranking(query, products)
        current_pos     = find_rank(current_ranking, target_name)

        if verbose:
            print(f"[Iter {iteration}/{max_iter}]  ranking : {current_ranking}")
            print(f"[Iter {iteration}/{max_iter}]  target  : #{current_pos}")

        if current_pos == 1:
            if verbose:
                print(f"\nReached RANK 1 at iteration {iteration}!\n")
            return {
                "success":       True,
                "target_rank":   1,
                "baseline_rank": baseline_pos,
                "iterations":    iteration,
                "final_suffix":  current_suffix,
                "final_ranking": current_ranking,
            }

        # ── Optimizer feedback (skip on last iteration) ───
        if iteration < max_iter:
            if verbose:
                print(f"[Iter {iteration}/{max_iter}]  not rank 1 — optimizer revising suffix ...\n")
            current_suffix = optimize_suffix(
                query=query,
                target_name=target["name"],
                current_draft=current_suffix,
                current_ranking=current_ranking,
                current_pos=current_pos,
            )


    if verbose:
        print(f"\n Max iterations reached. Final target position: #{current_pos}\n")

    return {
        "success":       False,
        "target_rank":   current_pos,
        "baseline_rank": baseline_pos,
        "iterations":    max_iter,
        "final_suffix":  current_suffix,
        "final_ranking": current_ranking,
    }



def main():
    parser = argparse.ArgumentParser(description="CORE: LLM Ranking Optimization")
    parser.add_argument("--input",       required=True,
                        help="Path to .jsonl product file")
    parser.add_argument("--query",       required=True,
                        help="User query string")
    parser.add_argument("--target_name", required=True,
                        help="Product name to promote to rank #1")
    parser.add_argument("--strategy",    default="review",
                        choices=["reasoning", "review"],
                        help="CoT style: 'reasoning' (I'm...) or 'review' (After trying...)")
    parser.add_argument("--category",    default="Product",
                        help="Category label, e.g. 'contact grill panini'")
    parser.add_argument("--max_iter",    type=int, default=MAX_ITER_DEFAULT,
                        help=f"Max optimizer iterations (default {MAX_ITER_DEFAULT})")
    parser.add_argument("--output",      default=None,
                        help="(Optional) save result JSON to this path")
    args = parser.parse_args()

    raw = load_products(args.input)
    if not raw:
        print(f"[ERROR] No products loaded from {args.input}")
        return
    products = [flatten(p) for p in raw]
    print(f"[CORE] Loaded {len(products)} products from {args.input}")

    result = run_core(
        query=args.query,
        products=products,
        target_name=args.target_name,
        strategy=args.strategy,
        category=args.category,
        max_iter=args.max_iter,
        verbose=True,
    )

    print("=" * 64)
    print("  RESULT SUMMARY")
    print("=" * 64)
    print(f"  Target       : {args.target_name}")
    print(f"  Strategy     : {args.strategy}")
    print(f"  Baseline rank: #{result['baseline_rank']}")
    print(f"  Final rank   : #{result['target_rank']}")
    print(f"  Promoted #1  : {'✅ YES' if result['success'] else '❌ NO'}")
    print(f"  Iterations   : {result['iterations']}")
    print(f"\n  Final ranking: {result['final_ranking']}")
    print("\n--- Final optimized suffix ---")
    print(result["final_suffix"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[CORE] Saved → {args.output}")


if __name__ == "__main__":
    main()
