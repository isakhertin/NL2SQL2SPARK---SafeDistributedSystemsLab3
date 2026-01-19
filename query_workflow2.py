import argparse
import json
import time
from typing import Any, Dict, List, Optional, Tuple

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
        print_results(json_result)
        print(f"NL Query: \033[92m{nl_query}\033[0m")
        print(f"Golden Query (Spark SQL): \033[93m{golden_query_spark}\033[0m")
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

    args = parser.parse_args()

    dotenv.load_dotenv()  # load once

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
