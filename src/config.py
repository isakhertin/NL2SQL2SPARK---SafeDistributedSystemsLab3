import os
DB_PATH = os.getcwd() + "/db"
BENCHMARK_FILE = "dev.json"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PROMPT_SUFIX = "You are an agent with access to tools.CRITICAL RULES: - Do NOT write Scala, Python, or pseudocode. - Do NOT describe steps. - You MUST call the tool 'query_sql_db' (or the Spark SQL execution tool available) exactly once to answer. - The tool input MUST be a single Spark SQL query string. - Use only tables and columns that exist. - After the tool returns, respond with the result only. -Match literal types: if schema shows numeric/double, compare using a number (no quotes)."
SCHEMA_LOOP_COUNT = 3

from enum import Enum

class Provider(Enum):
    GOOGLE = "google"
    CLOUDFLARE = "cloudflare"


metrics = {
    "total_time": -1,
    "spark_exec_time": -1,
    "translation_time": -1,
    "sparksql_query": None,
    "answer": None
}

DEFAULT_MODELS = {
    Provider.GOOGLE: "gemini-2.5-flash",
    Provider.CLOUDFLARE: '@cf/meta/llama-4-scout-17b-16e-instruct'
}