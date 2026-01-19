from config import (
    DB_PATH,
    BENCHMARK_FILE
)
import os
import json

import sqlite3
import pandas as pd

#TODO
def load_tables(spark_session, db_name):
    """
    Loads all tables from a SQLite database into a Spark session.

    Args:
        spark_session: Spark session to use for loading tables.
        db_name: Name of the SQLite database file to load tables from.
    """
    # Build path: DB_PATH/<db_name>.sqlite OR DB_PATH/<db_name>/<db_name>.sqlite (common BIRD layouts)
    candidates = [
        os.path.join(DB_PATH, f"{db_name}.sqlite"),
        os.path.join(DB_PATH, db_name, f"{db_name}.sqlite"),
        os.path.join(DB_PATH, f"{db_name}.db"),
        os.path.join(DB_PATH, db_name, f"{db_name}.db"),
    ]

    db_file = next((p for p in candidates if os.path.exists(p)), None)
    if db_file is None:
        raise FileNotFoundError(
            f"Could not find SQLite DB for '{db_name}'. Tried:\n" + "\n".join(candidates)
        )

    print("[DB] Using SQLite file:", db_file)


    conn = sqlite3.connect(db_file)
    try:
        # Get table names (skip sqlite internal tables)
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()
        table_names = [r[0] for r in table_rows]

        if not table_names:
            raise ValueError(f"No tables found in database: {db_file}")

        for table in table_names:
            pdf = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)

            # Create Spark DF & register as temp view
            sdf = spark_session.createDataFrame(pdf)
            sdf.createOrReplaceTempView(table)
        print("[DB] Tables:", table_names[:10], "..." if len(table_names) > 10 else "")

    finally:
        conn.close()


def load_query_info(query_id: int):

    query_data_file = os.path.join(DB_PATH, BENCHMARK_FILE)
    with open(query_data_file, 'r') as f:
        all_queries = json.load(f)

    query_info = None
    for query_entry in all_queries:
        if query_entry['question_id'] == query_id:
            query_info = query_entry
            break

    if query_info is None:
        raise ValueError(f"Query ID {query_id} not found")

    database_name = query_info['db_id']
    question = " ".join([
        query_info["question"],
        query_info["evidence"]
    ])
    golden_query = query_info["SQL"]

    return database_name, question, golden_query
