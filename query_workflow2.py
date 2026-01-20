import argparse
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import os
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

import dotenv

from spark_nl import (
    get_spark_session,
    get_spark_sql,
    get_spark_agent,
    run_nl_query,
    process_result,
    print_results,
    run_sparksql_query,
)

from benchmark_ds import load_tables, load_query_info
from llm import get_llm
from evaluation import (
    translate_sqlite_to_spark,
    jaccard_index,
    evaluate_spark_sql,
)

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def benchmark_query(
    spark_session,
    query_id: int,
    provider: str,
    loaded_db: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    Benchmarks one query id once (one replica).

    Reuses the provided spark_session.
    Loads DB tables only if db_name differs from loaded_db.

    Returns:
        (record, db_name_loaded)
    """
    database_name, nl_query, golden_query = load_query_info(query_id)
    golden_query_spark = translate_sqlite_to_spark(golden_query)

    # Load tables only when DB changes (big speedup for batch)
    if loaded_db != database_name:
        load_tables(spark_session, database_name)
        loaded_db = database_name
        if verbose:
            print(f"--- Loaded DB '{database_name}' into Spark ---")

    spark_sql = get_spark_sql()
    llm = get_llm(provider=provider)
    agent = get_spark_agent(spark_sql, llm=llm)

    # Run NL query (agent executes Spark tool call if supported)
    run_nl_query(agent, nl_query, llm=llm)
    json_result = process_result()

    # Compute EA/SA when we have an executed inferred query
    jaccard = None
    spider_em = None
    if json_result.get("execution_status") == "VALID":
        # Ground truth (Spark DF) + inferred result (already in json_result)
        ground_truth_df = run_sparksql_query(spark_session, golden_query_spark)
        inferred_result = json_result.get("query_result")
        jaccard = jaccard_index(ground_truth_df, inferred_result)

        pred_sql = json_result.get("sparksql_query")
        if pred_sql:
            spider_em = evaluate_spark_sql(golden_query_spark, pred_sql, spark_session)

    record: Dict[str, Any] = {
        "query_id": query_id,
        "provider": provider,
        "db": database_name,
        "nl_query": nl_query,
        "gold_sqlite": golden_query,
        "gold_spark": golden_query_spark,
        "jaccard": jaccard,
        "spider_em": spider_em,
        **json_result,
    }

    if verbose:
        #print_results(json_result)
        #print(f"NL Query: \033[92m{nl_query}\033[0m")
        #print(f"Golden Query (Spark SQL): \033[93m{golden_query_spark}\033[0m")
        if jaccard is not None:
            print(f"Jaccard Index: {jaccard}")
        if spider_em is not None:
            print(f"Spider Exact Match Score: {spider_em}")

    return record, loaded_db


def parse_ids(args) -> List[int]:
    if args.start is not None and args.end is not None:
        if args.end < args.start:
            raise ValueError("--end must be >= --start")
        return list(range(args.start, args.end + 1))
    return [args.id]

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_summary_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Produces the required table:
    Input Prompt | Gold Query | Inferred Query | EA | SA

    - EA = mean Jaccard over replicas (ignoring failures)
    - SA = mean Spider EM over replicas (ignoring failures)
    - Inferred Query = most frequent inferred SparkSQL query across replicas
      (fallback: first non-empty)
    """
    df = pd.DataFrame(rows)

    # Keep only successful records with the columns we need
    # If your pipeline logs "error", those rows should not count for EA/SA.
    df_ok = df[df["error"].isna()] if "error" in df.columns else df.copy()

    # Ensure expected columns exist
    for col in ["nl_query", "gold_spark", "sparksql_query", "jaccard", "spider_em", "query_id"]:
        if col not in df_ok.columns:
            df_ok[col] = None

    def pick_inferred(group: pd.DataFrame) -> str:
        vals = [v for v in group["sparksql_query"].tolist() if isinstance(v, str) and v.strip()]
        if not vals:
            return ""
        # Most common inferred SQL across replicas
        return Counter(vals).most_common(1)[0][0]

    summary = (
        df_ok.groupby(["query_id"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "Input Prompt": g["nl_query"].iloc[0] if len(g) else "",
                    "Gold Query": g["gold_spark"].iloc[0] if len(g) else "",
                    "Inferred Query": pick_inferred(g),
                    "EA": float(pd.to_numeric(g["jaccard"], errors="coerce").mean()),
                    "SA": float(pd.to_numeric(g["spider_em"], errors="coerce").mean()),
                }
            )
        )
        .reset_index(drop=True)
    )

    # Clean NaNs for display
    summary["EA"] = summary["EA"].fillna(0.0)
    summary["SA"] = summary["SA"].fillna(0.0)

    return summary


def save_table_outputs(summary: pd.DataFrame, out_prefix: str) -> None:
    # CSV for easy grading / analysis
    csv_path = f"{out_prefix}.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8")

    # Markdown table for your report / README
    md_path = f"{out_prefix}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary.to_markdown(index=False))

    print("\n=== Required Table (preview) ===\n")
    print(summary.to_markdown(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


def save_aggregated_plots(summary: pd.DataFrame, out_path: str) -> None:
    """
    Saves two simple aggregated plots:
    - Distribution of EA across queries
    - Distribution of SA across queries
    """
    plt.figure()
    summary["EA"].plot(kind="hist", bins=20)
    plt.xlabel("EA (mean Jaccard per query)")
    plt.ylabel("Count of queries")
    plt.title("Execution Accuracy (EA) Distribution")
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_EA_hist.png"), dpi=200)
    plt.close()

    plt.figure()
    summary["SA"].plot(kind="hist", bins=20)
    plt.xlabel("SA (mean Spider EM per query)")
    plt.ylabel("Count of queries")
    plt.title("Structural Accuracy (SA) Distribution")
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_SA_hist.png"), dpi=200)
    plt.close()

    # Also a combined bar chart (sorted by EA) is often useful
    plt.figure(figsize=(10, 4))
    s = summary.sort_values("EA")
    plt.bar(range(len(s)), s["EA"].tolist())
    plt.xlabel("Queries (sorted by EA)")
    plt.ylabel("EA")
    plt.title("EA per Query (sorted)")
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_EA_per_query.png"), dpi=200)
    plt.close()

    print(f"Saved plots: {out_path.replace('.png','_EA_hist.png')}")
    print(f"Saved plots: {out_path.replace('.png','_SA_hist.png')}")
    print(f"Saved plots: {out_path.replace('.png','_EA_per_query.png')}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark BIRD queries with NL2SQL->Spark.")
    parser.add_argument("--id", type=int, default=1, help="Single query ID (default: 1)")
    parser.add_argument("--provider", type=str, default="google", help="LLM provider (default: google)")

    # Batch args
    parser.add_argument("--start", type=int, default=None, help="Start query id (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End query id (inclusive)")
    parser.add_argument("--replicas", type=int, default=10, help="Replicas per query (default: 10)")
    parser.add_argument("--out", type=str, default="results.jsonl", help="Output JSONL file")
    parser.add_argument("--verbose", action="store_true", help="Print per-run details")
    parser.add_argument("--report_prefix", type=str, default="report", help="Prefix for table outputs (CSV/MD)")
    parser.add_argument("--plots_prefix", type=str, default="plots.png", help="Prefix for plot outputs (PNGs)")

    args = parser.parse_args()
    
    dotenv.load_dotenv() 

    ids = parse_ids(args)
    
    # Fresh output file
    open(args.out, "w", encoding="utf-8").close()

    # Create one Spark session for the whole run
    spark_session = get_spark_session()

    loaded_db: Optional[str] = None

    for qid in ids:
        for r in range(args.replicas):
            try:
                rec, loaded_db = benchmark_query(
                    spark_session=spark_session,
                    query_id=qid,
                    provider=args.provider,
                    loaded_db=loaded_db,
                    verbose=args.verbose,
                )
                rec["replica"] = r
                rec["timestamp"] = time.time()
                append_jsonl(args.out, rec)

            except Exception as e:
                append_jsonl(
                    args.out,
                    {
                        "query_id": qid,
                        "replica": r,
                        "provider": args.provider,
                        "timestamp": time.time(),
                        "error": str(e),
                    },
                )
    
    # --- Produce required outputs (table + aggregated plots) ---
    rows = read_jsonl(args.out)
    summary = build_summary_table(rows)

    save_table_outputs(summary, args.report_prefix)
    save_aggregated_plots(summary, args.plots_prefix)

