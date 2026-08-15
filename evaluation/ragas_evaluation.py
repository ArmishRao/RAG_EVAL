import json
import os
import time
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from ragas.run_config import RunConfig

print("Step 1")

EVAL_FILE = "evaluation/results_3.json"
CHECKPOINT_FILE = "evaluation/checkpoint_3.json"
OUTPUT_CSV = "evaluation/ragas_results_3.csv"
AVERAGES_CSV = "evaluation/ragas_averages_3.csv"

BATCH_SIZE = 10          # samples per batch — small enough that a 503 doesn't waste much
BATCH_RETRIES = 5        # how many times to retry a failed batch
BATCH_RETRY_BASE_WAIT = 30  # seconds, multiplied by attempt number

# ---------------------------------------------------------------------------
# API key: load from environment instead of hardcoding.
# Set it before running, e.g. (PowerShell):
#   $env:MISTRAL_API_KEY = "your-key-here"
# or put it in a .env file and load with python-dotenv.

# ---------------------------------------------------------------------------

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY not found in environment. "
        "Set it before running, e.g. $env:MISTRAL_API_KEY='your-key' (PowerShell) "
        "or use a .env file with python-dotenv."
    )
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

MISTRAL_MODEL = "mistral-small-latest"

print("📂 Loading evaluation data...")
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print(f"✅ Loaded {len(eval_data)} samples")
print("Step 2")

print(f"\nInitializing Mistral with model: {MISTRAL_MODEL}")
llm = ChatMistralAI(
    model=MISTRAL_MODEL,
    temperature=0,
    max_retries=2,
)

# Mistral embeddings — give the client its OWN retry/backoff.
# Note: RunConfig's max_retries only governs RAGAS-level retries,
# NOT the langchain_mistralai client's internal tenacity retries.
# Without this, a 503 from the embeddings endpoint can exhaust the
# client's default retries before RAGAS even gets a chance to retry.
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    max_retries=8,
)

print("✅ Connected to Mistral API")
print("Step 3")

# Mistral free tier is strict (~1 req/sec). Run near-sequentially with
# generous retries/backoff so a 429/503 pauses and resumes instead of failing.
run_config = RunConfig(
    max_workers=1,       # only 1 request in flight at a time
    max_retries=10,      # retry a rate-limited/failed call up to 10 times
    max_wait=60,         # wait up to 60s between retries
    timeout=180,         # allow slow individual calls to finish
)

# ---------------------------------------------------------------------------
# Resume from checkpoint if one exists, so a crash/503 mid-run doesn't
# throw away already-evaluated samples.
# ---------------------------------------------------------------------------
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    done = len(all_results)
    print(f"🔄 Found checkpoint with {done} samples already evaluated. Resuming...")
else:
    all_results = []
    done = 0

remaining = eval_data[done:]
print(f"Step 4: {len(remaining)} samples remaining out of {len(eval_data)}")

if not remaining:
    print("✅ All samples already evaluated (per checkpoint). Skipping to final report.")
else:
    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

        batch_dataset = Dataset.from_list(
            [
                {
                    "user_input": item["question"],
                    "response": item["generated_answer"],
                    "retrieved_contexts": item["retrieved_contexts"],
                    "reference": item["ground_truth"],
                }
                for item in batch
            ]
        )

        print(f"\n⏳ Evaluating batch {batch_num}/{total_batches} ({len(batch)} samples)...")

        for attempt in range(1, BATCH_RETRIES + 1):
            try:
                result = evaluate(
                    dataset=batch_dataset,
                    metrics=[
                        faithfulness,
                        answer_relevancy,
                        context_precision,
                        context_recall,
                    ],
                    llm=llm,
                    embeddings=embeddings,
                    run_config=run_config,
                    raise_exceptions=True,
                )
                df_batch = result.to_pandas()
                all_results.extend(df_batch.to_dict(orient="records"))

                # Save checkpoint after every successful batch
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)

                print(
                    f"✅ Batch {batch_num}/{total_batches} done, checkpoint saved "
                    f"({len(all_results)}/{len(eval_data)} total)"
                )
                break

            except Exception as e:
                print(
                    f"⚠️ Batch {batch_num} failed on attempt {attempt}/{BATCH_RETRIES} "
                    f"({type(e).__name__}: {e})"
                )
                if attempt == BATCH_RETRIES:
                    print(
                        f"❌ Batch {batch_num} failed after {BATCH_RETRIES} attempts. "
                        f"Stopping. Progress is saved in '{CHECKPOINT_FILE}' — "
                        f"just re-run the script later to resume from here."
                    )
                    raise
                wait = BATCH_RETRY_BASE_WAIT * attempt
                print(f"   retrying in {wait}s...")
                time.sleep(wait)

# ---------------------------------------------------------------------------
# Final report — built from all_results (checkpoint + this run combined)
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("📋 BUILDING FINAL REPORT")
print("=" * 50)

df = pd.DataFrame(all_results)

print("\n" + "=" * 50)
print("📋 PER-QUESTION RESULTS (first 5)")
print("=" * 50)
print(df.head(5))

metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
metric_cols = [c for c in metric_cols if c in df.columns]  # only ones that ran

averages = df[metric_cols].mean()

print("\n" + "=" * 50)
print("📊 AVERAGE SCORES ACROSS ALL SAMPLES")
print("=" * 50)
for metric, score in averages.items():
    print(f"{metric:20s}: {score:.4f}")

overall_average = averages.mean()
print(f"\n{'Overall (all metrics)':20s}: {overall_average:.4f}")

df.to_csv(OUTPUT_CSV, index=False)

averages_df = averages.to_frame(name="average_score")
averages_df.loc["overall"] = overall_average
averages_df.to_csv(AVERAGES_CSV)

print(f"\n✅ Results saved to: {OUTPUT_CSV}")
print(f"✅ Averages saved to: {AVERAGES_CSV}")
print(f"ℹ️  Checkpoint file '{CHECKPOINT_FILE}' left in place — delete it manually before a fresh full re-run.")