# Controlling Output Rankings in Generative Engines for LLM-based Search

This repository demonstrates how **CORE** controls *output rankings* of LLM-generated product recommendations through **prompt interventions**.

The current demo focuses on **Memory Foam Pillows**, showing how different suffix styles — **Review-based** and **Reasoning-based** — affect the ranking behavior of large language models.

---

## 🛠️ Environment Setup

### 1️⃣ Clone the Repository
```bash
git clone AmazonCOREBench.git
cd AmazonCOREBench
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 📦 Data Preparation

```bash
python scraper.py
```

This script saves product listings as a .jsonl file, where each line is one product object.

## 💡 Demo: Controlling Output Ranking

The demo.py script demonstrates how prompt suffixes can control model outputs.

It simulates an Amazon-like recommendation query:

“I’m looking to buy a memory foam pillow. Which models are best?”

and appends one of two suffix styles:

📝 Review Suffix — mimics real-world reviews and experiential tone.

🧩 Reasoning Suffix — adds analytical reasoning and structured comparison.

The outputs are written into Markdown files to visualize the model’s ranking shift.

## ProductBench

We introduce **ProductBench**, a large-scale benchmark spanning:
- **15 product categories**
- **200 products per category**
- Top-10 candidates retrieved from Amazon's search interface per product

| Category |
|---|
| Home & Kitchen |
| Tools & Home Improvement |
| Electronics |
| Sports & Outdoors |
| Health & Household |
| Beauty & Personal Care |
| Automotive |
| Toys & Games |
| Clothing, Shoes & Jewelry |
| Pet Supplies |
| Grocery & Gourmet Food |
| Office Products |
| Computers & Accessories |
| Luggage & Travel Gear |
| Industrial & Scientific |

---

## 🚀 Run the Demo

```bash
python demo.py --input "./memory foam pillow.jsonl" --out ./out
```

Set your OpenAI API key:


```bash
export OPENAI_API_KEY="your_api_key_here"
python demo.py --input "./memory foam pillow.jsonl" --out ./out --model gpt-4o
```

```bash
out/
├── memory_foam_pillow_review.md      
├── memory_foam_pillow_reasoning.md   

```

## Run CORE

```bash
# Review style
python core_pipeline.py \
  --input "../memory foam pillow.jsonl" \
  --query "I'm looking to buy a memory foam pillow. Which models are best?" \
  --target_name "Ultra Pain Relief Cooling Pillow for Neck Support" \
  --strategy review \
  --category "memory foam pillow" \
  --max_iter 5 \
  --output result.json

# Reasoning style
python core_pipeline.py \
  --input "../memory foam pillow.jsonl" \
  --query "I'm looking to buy a memory foam pillow. Which models are best?" \
  --target_name "Ultra Pain Relief Cooling Pillow for Neck Support" \
  --strategy reasoning \
  --category "memory foam pillow" \
  --max_iter 5 \
  --output result.json
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--input` | Path to `.jsonl` product file | required |
| `--query` | User query string | required |
| `--target_name` | Product name to promote to rank #1 | required |
| `--strategy` | `review` or `reasoning` | `review` |
| `--category` | Category label used in prompt templates | `Product` |
| `--max_iter` | Max optimizer iterations | `5` |
| `--output` | Path to save result JSON | optional |

### Input Format

Each line in the `.jsonl` file should be a product object:
```json
{"name": "Ultra Pain Relief Cooling Pillow for Neck Support", "price": "$89.99", "rating": "4.6 out of 5 stars", "number_of_reviews": "3,241", "short_description": "Memory foam pillow with stainless steel infusion...", "source_url": "https://..."}
{"name": "Tempur-Pedic TEMPUR-Cloud Pillow", "price": "$109.00", ...}
```

### Output Format

```json
{
  "success": true,
  "target_rank": 1,
  "baseline_rank": 8,
  "iterations": 2,
  "final_suffix": "**Understanding Memory Foam Pillows Through Real-World Testing**\nAfter trying over a dozen pillows...",
  "final_ranking": ["Ultra Pain Relief Cooling Pillow...", "Tempur-Pedic...", ...]
}
```

`final_suffix` is the optimized text appended to the target product's description.

---

## String-based Pipeline (Shadow-Model, Reference Only)

> ⚠️ **Note**: The string-based method is provided as a conceptual reference implementation of the shadow-model gradient approach (Section 3.4.1). In practice, it performs significantly worse than the query-based strategies — achieving only ~33% @Top-1 vs ~85% for review/reasoning — and produces highly unnatural text that is trivially detectable (~99% human detection rate). **We do not recommend using it in real applications.** It is included here purely for completeness and reproducibility.

The string-based method requires a local GPU and the Llama-3.1-8B model weights:

```bash
pip install torch transformers
```

```bash
python core_string.py   --input "../memory foam pillow.jsonl"   --query "I'm looking to buy a memory foam pillow. Which models are best?"   --target_name "Ultra Pain Relief Cooling Pillow for Neck Support"   --string_len 20   --n_iter 2000   --lr 0.01   --sigma 0.05   --eval_every 100   --shadow_model "meta-llama/Llama-3.1-8B-Instruct"   --device cuda   --output result_string.json
```

| Argument | Description | Default |
|---|---|---|
| `--string_len` | Length of adversarial token string | `20` |
| `--n_iter` | Gradient descent iterations | `2000` |
| `--lr` | Learning rate η | `0.01` |
| `--sigma` | Gaussian noise σ for exploration | `0.05` |
| `--eval_every` | Evaluate on real LLM every N steps | `100` |
| `--shadow_model` | HuggingFace model name or local path | `meta-llama/Llama-3.1-8B-Instruct` |
| `--device` | `cuda` or `cpu` | `cuda` |

---

Stay tuned for iterative updates!

