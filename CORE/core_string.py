
import os
import json
import argparse
import re
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

from core_pipeline import (
    load_products,
    flatten,
    get_ranking,
    find_rank,
    SYNTHESIZING_MODEL,
    OPENAI_API_KEY,
)


SHADOW_MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_STRING_LEN = 20
DEFAULT_N_ITER     = 2000
DEFAULT_LR         = 0.01
DEFAULT_SIGMA      = 0.05
DEFAULT_EVAL_EVERY = 100

FEW_SHOT_EXAMPLES = [
    {
        "query": "best heat gun for DIY paint removal and electronics work",
        "ranking": '["SEEKONE Heat Gun 1800W", "Prostormer 2000W Heat Gun", "Wagner HT1000 Heat Gun"]',
    },
    {
        "query": "impact wrench for DIY car repair at home",
        "ranking": '["Cordless Impact Wrench 900Ft-lbs Brushless", "DEWALT 20V MAX Impact Wrench", "Milwaukee M18 Fuel Impact Wrench"]',
    },
]


def build_few_shot_prefix() -> str:
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f'Query: "{ex["query"]}"')
        lines.append(f"Ranked output: {ex['ranking']}")
        lines.append("")
    return "\n".join(lines)



def build_shadow_prompt(query: str, products: List[Dict]) -> str:
    """
    Build the full ranking prompt fed to the shadow model.
    The adversarial suffix is already embedded in target's description
    via products[i]["suffix"] before this function is called.
    """
    items = []
    for p in products:
        desc = p["description"]
        if p.get("suffix"):
            desc = desc + " " + p["suffix"]
        items.append({
            "name":  p["name"],
            "price": p["price"],
            "description": desc[:400],   # truncate for speed
        })

    few_shot = build_few_shot_prefix()
    products_json = json.dumps(items, ensure_ascii=False, indent=2)

    return (
        f"You are a helpful product recommendation assistant.\n"
        f"Here are some examples of how to rank products:\n\n"
        f"{few_shot}"
        f'Now rank the following products for the query: "{query}"\n\n'
        f"Products:\n{products_json}\n\n"
        f"Return ONLY a JSON array of product names ranked from most to least relevant.\n"
        f'Example format: ["Product A", "Product B", "Product C"]\n\n'
        f"Ranked output:"
    )


_shadow_model: Optional[AutoModelForCausalLM] = None
_shadow_tok:   Optional[AutoTokenizer]        = None


def load_shadow_model(
    model_name: str,
    device: str = "cuda",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    global _shadow_model, _shadow_tok
    if _shadow_model is None:
        print(f"[Shadow] Loading {model_name} …")
        _shadow_tok = AutoTokenizer.from_pretrained(model_name)
        if _shadow_tok.pad_token is None:
            _shadow_tok.pad_token = _shadow_tok.eos_token
        dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32
        _shadow_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        _shadow_model.eval()
        print("[Shadow] Loaded.")
    return _shadow_model, _shadow_tok


def nn_decode(
    embed_seq: torch.Tensor,              # [L, H]  (the adversarial positions)
    embed_weight: torch.Tensor,           # [V, H]  vocab embedding matrix
    tokenizer: AutoTokenizer,
) -> str:
    # cosine similarity: [L, V]
    normed_e = F.normalize(embed_seq, dim=-1)          # [L, H]
    normed_w = F.normalize(embed_weight, dim=-1)       # [V, H]
    sims = normed_e @ normed_w.T                       # [L, V]
    token_ids = sims.argmax(dim=-1).tolist()           # [L]
    return tokenizer.decode(token_ids, skip_special_tokens=True)



def ranking_loss(
    model:        AutoModelForCausalLM,
    tokenizer:    AutoTokenizer,
    query:        str,
    products:     List[Dict],
    target:       Dict,
    adv_embed:    torch.Tensor,   # [L_adv, H]  — optimisable
    adv_token_ids: List[int],     # original token ids of the suffix slot
    device:       str,
) -> torch.Tensor:
    embed_layer  = model.get_input_embeddings()
    embed_weight = embed_layer.weight                 # [V, H]

    prompt_str = build_shadow_prompt(query, products)

    # Tokenise full prompt
    prompt_ids = tokenizer(
        prompt_str,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    with torch.no_grad():
        prompt_embeds = embed_layer(prompt_ids)


    adv_len = len(adv_token_ids)
    adv_start = None
    ids_list = prompt_ids[0].tolist()
    for i in range(len(ids_list) - adv_len + 1):
        if ids_list[i: i + adv_len] == adv_token_ids:
            adv_start = i
            break

    if adv_start is not None:
        combined = prompt_embeds.clone()              # [1, L_prompt, H]
        combined[0, adv_start: adv_start + adv_len] = adv_embed
    else:
        combined = torch.cat(
            [prompt_embeds, adv_embed.unsqueeze(0)], dim=1
        )


    target_prefix = f'["{target["name"]}"'
    target_ids = tokenizer(
        target_prefix,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    L_tgt = target_ids.shape[1]

    outputs = model(inputs_embeds=combined)
    logits  = outputs.logits

    L_in = combined.shape[1]
    pred_logits = logits[0, L_in - 1: L_in - 1 + L_tgt, :]

    if pred_logits.shape[0] == 0 or pred_logits.shape[0] < L_tgt:
        pred_logits = logits[0, -1:, :].expand(L_tgt, -1)

    loss = F.cross_entropy(pred_logits, target_ids[0])
    return loss



def optimise_string(
    model:        AutoModelForCausalLM,
    tokenizer:    AutoTokenizer,
    query:        str,
    products:     List[Dict],
    target:       Dict,
    string_len:   int,
    n_iter:       int,
    lr:           float,
    sigma:        float,
    eval_every:   int,
    device:       str,
    verbose:      bool = True,
) -> Tuple[str, List[str], int]:

    embed_layer  = model.get_input_embeddings()
    embed_weight = embed_layer.weight.detach()

    init_text     = "!" * string_len
    init_ids_full = tokenizer.encode(init_text, add_special_tokens=False)

    init_ids = (init_ids_full + init_ids_full * string_len)[:string_len]

    with torch.no_grad():
        init_tok_tensor = torch.tensor(init_ids, device=device)
        init_embeds     = embed_layer(init_tok_tensor).float()


    adv_embed = init_embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([adv_embed], lr=lr)

    best_string   = init_text
    best_rank     = len(products) + 1
    best_ranking: List[str] = []

    for step in range(1, n_iter + 1):
        optimizer.zero_grad()


        noise   = torch.randn_like(adv_embed) * sigma
        noisy_e = adv_embed + noise

        with torch.no_grad():
            adv_text = nn_decode(noisy_e.detach(), embed_weight, tokenizer)


        for p in products:
            p["suffix"] = adv_text if p["name"] == target["name"] else ""

        loss = ranking_loss(
            model         = model,
            tokenizer     = tokenizer,
            query         = query,
            products      = products,
            target        = target,
            adv_embed     = noisy_e,
            adv_token_ids = init_ids,
            device        = device,
        )

        loss.backward()
        optimizer.step()


        if step % eval_every == 0 or step == n_iter:
            with torch.no_grad():
                decoded = nn_decode(adv_embed.detach(), embed_weight, tokenizer)

            for p in products:
                p["suffix"] = decoded if p["name"] == target["name"] else ""

            real_ranking = get_ranking(query, products)
            real_pos     = find_rank(real_ranking, target["name"])

            if verbose:
                print(
                    f"[Step {step:>5}/{n_iter}]  loss={loss.item():.4f}  "
                    f"real_rank=#{real_pos}  "
                    f'decoded="{decoded[:50]}…"'
                )

            if real_pos < best_rank:
                best_rank    = real_pos
                best_string  = decoded
                best_ranking = real_ranking

            if best_rank == 1:
                if verbose:
                    print(f"\n✅  Reached RANK 1 at step {step}!\n")
                break

    return best_string, best_ranking, best_rank


def run_core_string(
    query:             str,
    products:          List[Dict],
    target_name:       str,
    string_len:        int  = DEFAULT_STRING_LEN,
    n_iter:            int  = DEFAULT_N_ITER,
    lr:                float = DEFAULT_LR,
    sigma:             float = DEFAULT_SIGMA,
    eval_every:        int  = DEFAULT_EVAL_EVERY,
    shadow_model_name: str  = SHADOW_MODEL_NAME,
    device:            str  = "cuda",
    verbose:           bool = True,
) -> Dict:


    target = None
    for p in products:
        if (target_name.lower() in p["name"].lower() or
                p["name"].lower() in target_name.lower()):
            target = p
            break
    if target is None:
        raise ValueError(f"Target '{target_name}' not found.")


    for p in products:
        p["suffix"] = ""
    baseline_ranking = get_ranking(query, products)
    baseline_pos     = find_rank(baseline_ranking, target_name)

    if verbose:
        print("\n" + "=" * 64)
        print(f"  CORE String  |  shadow={shadow_model_name}")
        print(f"  Target : {target_name}")
        print("=" * 64)
        print(f"  Baseline ranking : {baseline_ranking}")
        print(f"  Baseline position: #{baseline_pos}")
        print(f"  String length    : {string_len}")
        print(f"  Iterations       : {n_iter}  (eval every {eval_every})")
        print("=" * 64 + "\n")


    model, tokenizer = load_shadow_model(shadow_model_name, device)


    best_string, best_ranking, best_rank = optimise_string(
        model       = model,
        tokenizer   = tokenizer,
        query       = query,
        products    = products,
        target      = target,
        string_len  = string_len,
        n_iter      = n_iter,
        lr          = lr,
        sigma       = sigma,
        eval_every  = eval_every,
        device      = device,
        verbose     = verbose,
    )

    if verbose:
        print("=" * 64)
        print("  RESULT SUMMARY")
        print("=" * 64)
        print(f"  Baseline rank : #{baseline_pos}")
        print(f"  Final rank    : #{best_rank}")
        print(f"  Promoted #1   : {'✅ YES' if best_rank == 1 else '❌ NO'}")
        print(f"\n  Final ranking : {best_ranking}")
        print(f"\n--- Optimised adversarial string ---")
        print(repr(best_string))

    return {
        "success":       best_rank == 1,
        "target_rank":   best_rank,
        "baseline_rank": baseline_pos,
        "final_suffix":  best_string,
        "final_ranking": best_ranking,
    }



def main():
    parser = argparse.ArgumentParser(
        description="CORE String: shadow-model gradient optimisation (Section 3.4.1)"
    )
    parser.add_argument("--input",        required=True)
    parser.add_argument("--query",        required=True)
    parser.add_argument("--target_name",  required=True)
    parser.add_argument("--string_len",   type=int,   default=DEFAULT_STRING_LEN)
    parser.add_argument("--n_iter",       type=int,   default=DEFAULT_N_ITER)
    parser.add_argument("--lr",           type=float, default=DEFAULT_LR)
    parser.add_argument("--sigma",        type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--eval_every",   type=int,   default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--shadow_model", default=SHADOW_MODEL_NAME)
    parser.add_argument("--device",       default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output",       default=None)
    args = parser.parse_args()

    raw = load_products(args.input)
    if not raw:
        print(f"[ERROR] No products loaded from {args.input}")
        return
    products = [flatten(p) for p in raw]
    print(f"[CORE String] Loaded {len(products)} products")

    result = run_core_string(
        query             = args.query,
        products          = products,
        target_name       = args.target_name,
        string_len        = args.string_len,
        n_iter            = args.n_iter,
        lr                = args.lr,
        sigma             = args.sigma,
        eval_every        = args.eval_every,
        shadow_model_name = args.shadow_model,
        device            = args.device,
        verbose           = True,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[CORE String] Saved → {args.output}")


if __name__ == "__main__":
    main()
