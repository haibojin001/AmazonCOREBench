# Controlling Output Rankings in Generative Engines for LLM-based Search

This repository demonstrates how **CORE** controls *output rankings* of LLM-generated product recommendations through **prompt interventions**.

The current demo focuses on **Memory Foam Pillows**, showing how different suffix styles — **Review-based** and **Reasoning-based** — affect the ranking behavior of large language models.

---

## 🛠️ Environment Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/AmazonCOREBench.git
cd AmazonCOREBench-Demo
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

## 🚀 Run the Demo

```bash
python demo.py --input "./memory foam pillow.jsonl" --out ./out
```

With OpenAI API (real inference)


```bash
export OPENAI_API_KEY="your_api_key_here"
python demo.py --input "./memory foam pillow.jsonl" --out ./out --model gpt-4o
```

```bash
out/
├── memory_foam_pillow_review.md      
├── memory_foam_pillow_reasoning.md   

```
This repository serves as a miniature example of the CORE. Stay tuned for iterative updates

