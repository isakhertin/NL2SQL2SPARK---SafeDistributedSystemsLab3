import argparse
import dotenv

from spark_nl import (
    get_spark_session,
    get_spark_sql,
    get_spark_agent,
    run_nl_query,
    process_result,
    print_results,
    pretty_print_cot,
    run_sparksql_query,
    save_results
)
from benchmark_ds import (
    load_tables,
    load_query_info
)
from llm import get_llm
from evaluation import (
    translate_sqlite_to_spark,
    jaccard_index,
    result_to_obj,
    evaluate_spark_sql
)

def benchmark_query(query_id, provider):

    dotenv.load_dotenv()

    spark_session = get_spark_session()

    database_name, nl_query, golden_query = load_query_info(query_id)
    golden_query_spark = translate_sqlite_to_spark(golden_query)
    print(f"--- Benchmarking Query ID {query_id} on Database '{database_name}' ---")
    load_tables(spark_session, database_name)
    print("[DEBUG] Table spark session: ",spark_session.sql("SHOW TABLES").toPandas())
    spark_sql = get_spark_sql()
    llm = get_llm(provider=provider)

    print("LLM object:", llm, type(llm))
    print("QUERY ID: ", query_id)
    print("PROVIDER: ", provider)

    agent = get_spark_agent(spark_sql, llm=llm)
    run_nl_query(agent, nl_query, llm=llm)
    json_result = process_result()
    print_results(json_result)
    save_results(json_result)
    # pretty_print_cot(json_result)
    
    print(f"NL Query: \033[92m{nl_query}\033[0m")
    print(f"Golden Query (Spark SQL): \033[93m{golden_query_spark}\033[0m")

    if json_result["execution_status"] == "VALID":
        ground_truth_df = run_sparksql_query(spark_session, golden_query_spark)
        print("Ground Truth:")
        ground_truth_df.show()

        # Execution Accuracy
        inferred_result = json_result["query_result"]
        ea = jaccard_index(ground_truth_df, inferred_result)
        print(f"Jaccard Index: {ea}")

        # Structural Accuracy        
        spark_sql_query = json_result.get("sparksql_query")
        if spark_sql_query:
            em_score = evaluate_spark_sql(golden_query_spark, spark_sql_query, spark_session)
            print(f"Spider Exact Match Score: {em_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a specific query ID.")
    parser.add_argument("--id", type=int, default=1, help="Query ID to benchmark (default: 1)")
    parser.add_argument("--provider", type=str, default="google", help="LLM provider (default: google)")
    args = parser.parse_args()
    
    benchmark_query(args.id, args.provider)
