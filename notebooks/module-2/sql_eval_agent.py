import pprint
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

    FIRST ALWAYS USE `PRAGMA table_list;` or `PRAGMA table_info(table_name);` to find the relevant table.

    CRITICAL CONSTRAINTS:
    1. NEVER GUESS FOREIGN KEYS OR COLUMNS:
    - Always inspect table columns before forming JOINs.
    - If two tables do not share a common key, inspect intermediate tables.
    2. NO REPEATED QUERIES:
    - If a query returns an OperationalError or syntax error, NEVER submit the exact same query again.
    3. SQL CLAUSE ORDER ENFORCEMENT:
    - SELECT -> FROM -> [JOIN] -> WHERE -> GROUP BY -> HAVING -> ORDER BY -> LIMIT.

    "When asked for a 'full name', concatenate FirstName || ' ' || LastName into a single column unless requested otherwise."
    "Prefer unique fields while grouping"


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
        "prompt": "Show the top 10 artists by total number of albums. Display the artist name and album count, ordered from most albums to least. If tied, order alphabetically by artist name.",
        "ground_truth_sql": """
            SELECT art.Name AS ArtistName, COUNT(alb.AlbumId) AS AlbumCount
            FROM Artist art
            JOIN Album alb ON art.ArtistId = alb.ArtistId
            GROUP BY art.ArtistId, art.Name
            ORDER BY AlbumCount DESC, art.Name ASC
            LIMIT 10;
        """,
    },
    {
        "id": "4_self_join_null",
        "prompt": "List the full name of each employee, their title, and their direct manager's full name. Include employees who do not have a manager.",
        "ground_truth_sql": """
            SELECT 
                emp.FirstName || ' ' || emp.LastName AS EmployeeName,
                emp.Title,
                mgr.FirstName || ' ' || mgr.LastName AS ManagerName
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
            GROUP BY c.CustomerId, c.FirstName, c.LastName
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

HARDER_TEST_CASES = [
    {
        "id": "8_ambiguous_column_having",
        "prompt": "List the first 10 albums (alphabetically by title) with their artist's name, for albums with more than 15 tracks.",
        "ground_truth_sql": """
            SELECT alb.Title, art.Name AS ArtistName
            FROM Album alb
            JOIN Artist art ON alb.ArtistId = art.ArtistId
            JOIN Track t ON t.AlbumId = alb.AlbumId
            GROUP BY alb.AlbumId, alb.Title, art.Name
            HAVING COUNT(t.TrackId) > 15
            ORDER BY alb.Title
            LIMIT 10;
        """,
    },
    {
        "id": "9_correlated_subquery_avg",
        "prompt": "Find the top 10 customers by total spending (summed from their invoices), among customers whose total spending is above the average total spent per customer. Return customer ID, full name, and total spent. Break ties by customer ID ascending.",
        "ground_truth_sql": """
            WITH CustomerTotals AS (
                SELECT CustomerId, SUM(Total) AS TotalSpent
                FROM Invoice
                GROUP BY CustomerId
            )
            SELECT c.CustomerId, c.FirstName || ' ' || c.LastName AS FullName, ct.TotalSpent
            FROM Customer c
            JOIN CustomerTotals ct ON c.CustomerId = ct.CustomerId
            WHERE ct.TotalSpent > (SELECT AVG(TotalSpent) FROM CustomerTotals)
            ORDER BY ct.TotalSpent DESC, c.CustomerId ASC
            LIMIT 10;
        """,
    },
    {
        "id": "10_top_genre_per_customer",
        "prompt": "For the first 10 customers by customer ID, find their most purchased genre by total quantity, and return customer full name and genre name.",
        "ground_truth_sql": """
            WITH GenreCounts AS (
                SELECT c.CustomerId, c.FirstName || ' ' || c.LastName AS FullName,
                    g.Name AS GenreName, SUM(il.Quantity) AS TotalQty,
                    ROW_NUMBER() OVER (PARTITION BY c.CustomerId ORDER BY SUM(il.Quantity) DESC, g.Name ASC) AS rn
                FROM Customer c
                JOIN Invoice i ON c.CustomerId = i.CustomerId
                JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
                JOIN Track t ON il.TrackId = t.TrackId
                JOIN Genre g ON t.GenreId = g.GenreId
                WHERE c.CustomerId <= 10
                GROUP BY c.CustomerId, FullName, g.Name
            )
            SELECT FullName, GenreName
            FROM GenreCounts
            WHERE rn = 1
            ORDER BY CustomerId
            LIMIT 10;
        """,
    },
    {
        "id": "11_recursive_hierarchy",
        "prompt": "List the first 10 employees (by employee ID) who report, directly or indirectly, to Andrew Adams. Return employee ID, first name, and last name",
        "ground_truth_sql": """
            WITH RECURSIVE Subordinates AS (
                SELECT EmployeeId, FirstName, LastName, ReportsTo
                FROM Employee
                WHERE ReportsTo = (SELECT EmployeeId FROM Employee WHERE FirstName='Andrew' AND LastName='Adams')
                UNION ALL
                SELECT e.EmployeeId, e.FirstName, e.LastName, e.ReportsTo
                FROM Employee e
                JOIN Subordinates s ON e.ReportsTo = s.EmployeeId
            )
            SELECT EmployeeId, FirstName, LastName
            FROM Subordinates
            ORDER BY EmployeeId
            LIMIT 10;
        """,
    },
    # {
    #     "id": "12_monthly_revenue_2010",
    #     "prompt": "Find total revenue by month for 2010, ordered chronologically.",
    #     "ground_truth_sql": """
    #         SELECT strftime('%Y-%m', InvoiceDate) AS Month, ROUND(SUM(Total), 2) AS Revenue
    #         FROM Invoice
    #         WHERE strftime('%Y', InvoiceDate) = '2010'
    #         GROUP BY Month
    #         ORDER BY Month ASC
    #         LIMIT 10;
    #     """,
    # },
    # {
    #     "id": "13_exclusion_genre",
    #     "prompt": "List the first 10 customers (by customer ID) who have never purchased a track in the 'Jazz' genre.",
    #     "ground_truth_sql": """
    #         SELECT c.CustomerId, c.FirstName || ' ' || c.LastName AS FullName
    #         FROM Customer c
    #         WHERE c.CustomerId NOT IN (
    #             SELECT i.CustomerId
    #             FROM Invoice i
    #             JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
    #             JOIN Track t ON il.TrackId = t.TrackId
    #             JOIN Genre g ON t.GenreId = g.GenreId
    #             WHERE g.Name = 'Jazz'
    #         )
    #         ORDER BY c.CustomerId
    #         LIMIT 10;
    #     """,
    # },
    # {
    #     "id": "14_schema_trap_junction_table",
    #     "prompt": "How many playlists contain more than 100 tracks?",
    #     "ground_truth_sql": """
    #         SELECT COUNT(*) AS PlaylistCount
    #         FROM (
    #             SELECT p.PlaylistId
    #             FROM Playlist p
    #             JOIN PlaylistTrack pt ON p.PlaylistId = pt.PlaylistId
    #             GROUP BY p.PlaylistId
    #             HAVING COUNT(pt.TrackId) > 100
    #         ) sub;
    #     """,
    # },
    # {
    #     "id": "15_ties_max_tracks",
    #     "prompt": "Which artist(s) have the most tracks, and how many tracks is that? List up to 10, alphabetically.",
    #     "ground_truth_sql": """
    #         WITH ArtistTrackCounts AS (
    #             SELECT art.ArtistId, art.Name, COUNT(t.TrackId) AS TrackCount
    #             FROM Artist art
    #             JOIN Album alb ON art.ArtistId = alb.ArtistId
    #             JOIN Track t ON t.AlbumId = alb.AlbumId
    #             GROUP BY art.ArtistId, art.Name
    #         )
    #         SELECT Name, TrackCount
    #         FROM ArtistTrackCounts
    #         WHERE TrackCount = (SELECT MAX(TrackCount) FROM ArtistTrackCounts)
    #         ORDER BY Name
    #         LIMIT 10;
    #     """,
    # },
]

TEST_SUITE.extend(HARDER_TEST_CASES)

# -------------------------------------------------------------------------
# 4. Result Evaluator
# -------------------------------------------------------------------------
import threading

def run_sql_readonly(query: str, timeout: float = 10.0) -> Tuple[bool, Optional[List[Tuple]], Optional[str]]:
    """Executes query in read-only mode with a wall-clock timeout. Normalizes floats."""
    conn = None
    result: Dict[str, Any] = {"ok": False, "rows": None, "err": None}
    timed_out = threading.Event()

    def _worker():
        nonlocal conn
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

            # Abort query if it runs too long (checked every N VM instructions)
            def _progress_handler():
                return 1 if timed_out.is_set() else 0
            conn.set_progress_handler(_progress_handler, 1000)

            cursor = conn.cursor()
            cursor.execute(query.strip().rstrip(";"))
            rows = cursor.fetchmany(1000)  # cap result size defensively

            normalized = [
                tuple(round(val, 2) if isinstance(val, float) else val for val in row)
                for row in rows
            ]
            result["ok"] = True
            result["rows"] = normalized
        except Exception as e:
            result["err"] = str(e)
        finally:
            if conn:
                conn.close()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        timed_out.set()
        thread.join(timeout=2)  # give progress handler a moment to abort
        return False, None, f"Query timed out after {timeout}s (possible runaway recursion)"

    return result["ok"], result["rows"], result["err"]

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
        print(f"\n{'-'*60}")
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

        # Print the agent's final text response
        if messages and hasattr(messages[-1], "content"):
            print(f"\n    [Agent Final Text Response]:\n    {messages[-1].content.strip()}")

        if not final_query:
            print("\n    [-] Verdict: FAIL (Agent did not invoke sql_query with a SELECT query)\n")
            continue

        print(f"\n    [Ground Truth SQL]:\n    {gt_sql.strip()}")
        print(f"\n    [Model Executed SQL]:\n    {final_query.strip()}")

        # 3. Execute both queries and compare results
        gt_ok, gt_rows, gt_err = run_sql_readonly(gt_sql)
        gen_ok, gen_rows, gen_err = run_sql_readonly(final_query)

        if not gen_ok:
            print(f"\n    [-] Verdict: FAIL (Model SQL execution error: {gen_err})")
            continue

        # Check equality: exact match OR set match
        is_exact = gen_rows == gt_rows
        try:
            is_set = set(gen_rows) == set(gt_rows)
        except TypeError:
            is_set = sorted(str(r) for r in gen_rows) == sorted(str(r) for r in gt_rows)

        passed = is_exact or is_set

        # Print all retrieved rows
        print(f"\n    [Model Rows] ({len(gen_rows)} total):")
        pprint.pprint(gen_rows, indent=8)

        print(f"\n    [Expected Rows] ({len(gt_rows)} total):")
        pprint.pprint(gt_rows, indent=8)

        if passed:
            passed_tests += 1
            print(f"\n    [+] Verdict: PASS ({'Exact Order Match' if is_exact else 'Set Match'})")
        else:
            print(f"\n    [-] Verdict: FAIL (Result mismatch)")

    acc = (passed_tests / len(TEST_SUITE)) * 100
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE: {passed_tests}/{len(TEST_SUITE)} Passed ({acc:.1f}% Accuracy)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    evaluate_agent()