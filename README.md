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

Stay tuned for iterative updates!

