import ast
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from langchain_community.utilities import SQLDatabase
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# -------------------------------------------------------------------------
# 1. Database & Tool Setup
# -------------------------------------------------------------------------
DB_PATH = "resources/Chinook.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")


@tool
def sql_query(query: str) -> str:
    """Obtain information from the database using SQL queries"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


# -------------------------------------------------------------------------
# 2. Agent Initialization
# -------------------------------------------------------------------------
model = init_chat_model(
    model="gemma4",
    model_provider="openai",
    api_key="dummy",
    base_url="http://localhost:8080/v1",
)

SYSTEM_PROMPT = """You are a SQLite database assistant with access to the `sql_query` tool.

    ENGINE & DIALECT:
    - Engine: SQLite.
    - Do NOT use MySQL syntax like `SHOW TABLES;`. 
    FIRST ALWAYS USE `PRAGMA table_list;` or `PRAGMA table_info(table_name);` to find the relevant table.

    CRITICAL CONSTRAINTS:
    1. NEVER GUESS FOREIGN KEYS OR COLUMNS:
    - Always inspect table columns before forming JOINs.
    - If two tables do not share a common key, inspect intermediate tables.
    2. NO REPEATED QUERIES:
    - If a query returns an OperationalError or syntax error, NEVER submit the exact same query again.
    3. SQL CLAUSE ORDER ENFORCEMENT:
    - SELECT -> FROM -> [JOIN] -> WHERE -> GROUP BY -> HAVING -> ORDER BY -> LIMIT.

    WORKFLOW:
    1. Inspect schema if needed.
    2. Execute the SELECT query with verified column names.
    3. Provide the answer based strictly on the retrieved data.
"""

agent = create_agent(model=model, tools=[sql_query], system_prompt=SYSTEM_PROMPT)

# -------------------------------------------------------------------------
# 3. Ground Truth Test Suite
# -------------------------------------------------------------------------
TEST_SUITE = [
    {
        "id": "1_filtering_sorting",
        "prompt": "List the first name, last name, and email of all customers from Brazil, sorted alphabetically by their last name.",
        "ground_truth_sql": "SELECT FirstName, LastName, Email FROM Customer WHERE Country = 'Brazil' ORDER BY LastName ASC;",
    },
    {
        "id": "2_limit_ordering",
        "prompt": "Find the 5 longest tracks by duration in milliseconds, displaying the track name and length.",
        "ground_truth_sql": "SELECT Name, Milliseconds FROM Track ORDER BY Milliseconds DESC LIMIT 5;",
    },
    {
        "id": "3_group_by_count",
        "prompt": "Show the top 10 artists by total number of albums. Display the artist name and album count, ordered from most albums to least.",
        "ground_truth_sql": """
            SELECT art.Name AS ArtistName, COUNT(alb.AlbumId) AS AlbumCount
            FROM Artist art
            JOIN Album alb ON art.ArtistId = alb.ArtistId
            GROUP BY art.ArtistId, art.Name
            ORDER BY AlbumCount DESC
            LIMIT 10;
        """,
    },
    {
        "id": "4_self_join_null",
        "prompt": "List all employees with their title, alongside their direct manager's full name. Include employees who do not have a manager.",
        "ground_truth_sql": """
            SELECT 
                emp.FirstName || ' ' || emp.LastName AS EmployeeName,
                emp.Title,
                COALESCE(mgr.FirstName || ' ' || mgr.LastName, 'No Manager') AS ManagerName
            FROM Employee emp
            LEFT JOIN Employee mgr ON emp.ReportsTo = mgr.EmployeeId;
        """,
    },
    {
        "id": "5_multi_table_revenue",
        "prompt": "Calculate the top 10 music genres by total sales revenue, rounded to 2 decimal places, ordered by highest sales first.",
        "ground_truth_sql": """
            SELECT 
                g.Name AS Genre,
                ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS TotalSales
            FROM Genre g
            JOIN Track t ON g.GenreId = t.GenreId
            JOIN InvoiceLine il ON t.TrackId = il.TrackId
            GROUP BY g.GenreId, g.Name
            ORDER BY TotalSales DESC
            LIMIT 10;
        """,
    },
    {
        "id": "6_having_filter",
        "prompt": "Find the top 10 highest spending customers who have spent more than $40 in total across all their purchases. Return their ID, full name, and total amount spent.",
        "ground_truth_sql": """
            SELECT 
                c.CustomerId,
                c.FirstName || ' ' || c.LastName AS FullName,
                ROUND(SUM(i.Total), 2) AS TotalSpent
            FROM Customer c
            JOIN Invoice i ON c.CustomerId = i.CustomerId
            GROUP BY c.CustomerId, FullName
            HAVING SUM(i.Total) > 40.0
            ORDER BY TotalSpent DESC
            LIMIT 10;
        """,
    },
    {
        "id": "7_unpurchased_tracks",
        "prompt": "Find the first 10 tracks in the database that have never been purchased, ordered by track ID. Return their track ID and name.",
        "ground_truth_sql": """
            SELECT t.TrackId, t.Name
            FROM Track t
            LEFT JOIN InvoiceLine il ON t.TrackId = il.TrackId
            WHERE il.InvoiceLineId IS NULL
            ORDER BY t.TrackId ASC
            LIMIT 10;
        """,
    },
]

# -------------------------------------------------------------------------
# 4. Result Evaluator
# -------------------------------------------------------------------------
def run_sql_readonly(query: str) -> Tuple[bool, Optional[List[Tuple]], Optional[str]]:
    """Executes query in read-only mode and normalizes floats."""
    conn = None
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(query.strip().rstrip(";"))
        rows = cursor.fetchall()
        
        # Round floats to 2 decimal places for consistent comparison
        normalized = [
            tuple(round(val, 2) if isinstance(val, float) else val for val in row)
            for row in rows
        ]
        return True, normalized, None
    except Exception as e:
        return False, None, str(e)
    finally:
        if conn:
            conn.close()


def extract_queries_from_messages(messages: List[Any]) -> List[str]:
    """Extracts all SQL queries invoked by the agent across tool calls."""
    queries = []
    for msg in messages:
        # Check tool_calls on AIMessage
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                if call.get("name") == "sql_query":
                    q = call.get("args", {}).get("query")
                    if q:
                        queries.append(q)
    return queries


def evaluate_agent():
    print(f"\n{'='*70}\nSTARTING TEXT-TO-SQL BENCHMARK ({len(TEST_SUITE)} Tests)\n{'='*70}\n")
    
    passed_tests = 0

    for test in TEST_SUITE:
        prompt = test["prompt"]
        gt_sql = test["ground_truth_sql"]
        print(f"[*] Testing: {test['id']}")
        print(f"    Prompt: {prompt}")

        # 1. Invoke the agent (LangSmith tracks this call automatically)
        try:
            res = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"metadata": {"test_id": test["id"]}}
            )
            messages = res.get("messages", [])
        except Exception as e:
            print(f"    [!] Agent execution crashed: {e}\n")
            continue

        # 2. Extract executed queries
        executed_queries = extract_queries_from_messages(messages)
        # Filter out PRAGMA/schema introspection queries to find the actual data query
        data_queries = [q for q in executed_queries if not q.strip().upper().startswith("PRAGMA")]
        final_query = data_queries[-1] if data_queries else (executed_queries[-1] if executed_queries else None)

        if not final_query:
            print("    [-] Verdict: FAIL (Agent did not invoke sql_query with a SELECT query)\n")
            continue

        # 3. Execute both queries and compare results
        gt_ok, gt_rows, gt_err = run_sql_readonly(gt_sql)
        gen_ok, gen_rows, gen_err = run_sql_readonly(final_query)

        if not gen_ok:
            print(f"    [-] Verdict: FAIL (Model SQL syntax/execution error: {gen_err})")
            print(f"        Query: {final_query}\n")
            continue

        # Check equality: exact match OR set match (if row order was not strictly required)
        is_exact = gen_rows == gt_rows
        try:
            is_set = set(gen_rows) == set(gt_rows)
        except TypeError:
            is_set = sorted(str(r) for r in gen_rows) == sorted(str(r) for r in gt_rows)

        passed = is_exact or is_set

        if passed:
            passed_tests += 1
            print(f"    [+] Verdict: PASS ({'Exact Order Match' if is_exact else 'Set Match'})")
            print(f"        Rows returned: {len(gen_rows)}")
        else:
            print(f"    [-] Verdict: FAIL (Result mismatch)")
            print(f"        Model SQL: {final_query}")
            print(f"        Model Rows (count={len(gen_rows)}): {gen_rows[:2]}...")
            print(f"        Expected Rows (count={len(gt_rows)}): {gt_rows[:2]}...")
        print()

    acc = (passed_tests / len(TEST_SUITE)) * 100
    print(f"{'='*70}")
    print(f"BENCHMARK COMPLETE: {passed_tests}/{len(TEST_SUITE)} Passed ({acc:.1f}% Accuracy)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    evaluate_agent()