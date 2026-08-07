"""
queries.py - every SQL statement Errandman Runs, in one place.
pipeline.py calls build_insight_tables() once to materialize the summary tables.
app.py calls the get_* functions for filtered, on-demand lookups.
Neither pipeline.py nor app.py contains raw SQL of its own.
"""
import pandas as pd


CREATE_RIDER_STATS = """
CREATE TABLE rider_stats AS
SELECT
     Rider,                
     COUNT(*)                               AS total_orders,
     SUM(is_completed)                      AS completed_orders,
     SUM(is_refunded)                       AS refunded_orders,
     SUM(is_cancelled)                      AS cancelled_orders,
     AVG(Price)                             AS avg_price,
     AVG(accept_min)                        AS avg_accept_min,
     ROUND(100.0 * SUM(is_completed) / COUNT(*), 1) AS completion_rate
FROM orders
WHERE Rider != 'Unassigned'
GROUP BY Rider
"""

CREATE_CUSTOMER_STATS = """
CREATE TABLE customer_stats AS
SELECT
     Customer,                         
     COUNT(*)                          AS order_count,
     SUM(Price)                        AS total_spent,
     AVG(Price)                        AS avg_price,
     SUM(is_completed)                 AS completed_orders,
     CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END AS is_repeat
FROM orders
WHERE Customer != 'Guest'
GROUP BY Customer
ORDER BY order_count DESC
"""

CREATE_ROUTE_STATS = """
CREATE TABLE route_stats AS
SELECT
     "From", "To",
     COUNT(*)              AS trip_count,
     AVG(Price)            AS avg_price
FROM orders
WHERE "From" IS NOT NULL AND "To" IS NOT NULL
GROUP BY "From", "To"
ORDER BY trip_count DESC
"""

CREATE_MONTHLY_SUMMARY = """
CREATE TABLE monthly_summary AS
SELECT
     created_month,
     COUNT(*)                                              AS order_count,
     SUM(CASE WHEN is_completed = 1 THEN Price ELSE 0 END) AS revenue,
     ROUND(100.0 * SUM(is_completed) / COUNT(*), 1)        AS completion_rate
FROM orders
GROUP BY created_month
ORDER BY created_month
"""

CREATE_SERVICE_RELIABILITY = """
CREATE TABLE service_reliability AS
SELECT
     Title,
     COUNT(*)                                       AS order_count,
     ROUND(100.0 * SUM(is_refunded) / COUNT(*), 1)  AS refund_rate,
     ROUND(100.0 * SUM(is_cancelled) / COUNT(*), 1) AS cancel_rate,
     ROUND(100.0 * SUM(is_completed) / COUNT(*), 1) AS completion_rate,
     AVG(Price)                                     AS avg_price
FROM orders
GROUP BY Title
"""

CREATE_DEMAND_HEATMAP = """
CREATE TABLE demand_heatmap AS
SELECT created_dow, created_hour, COUNT(*) AS order_count
FROM orders
GROUP BY created_dow, created_hour
"""

CREATE_ASSIGNMENT_BY_SERVICE = """
CREATE TABLE assignment_by_service AS
SELECT
    Title,
    COUNT(*)                                        AS order_count,
    SUM(was_assigned)                                AS assigned_count,
    ROUND(100.0 * SUM(was_assigned) / COUNT(*), 1)  AS assignment_rate
FROM orders
GROUP BY Title
"""

CREATE_ASSIGNMENT_BY_HOUR = """
CREATE TABLE assignment_by_hour AS
SELECT
    created_hour,
    COUNT(*)                                        AS order_count,
    SUM(was_assigned)                                AS assigned_count,
    ROUND(100.0 * SUM(was_assigned) / COUNT(*), 1)  AS assignment_rate
FROM orders
GROUP BY created_hour
ORDER BY created_hour
"""

INSIGHTS_TABLE_QUERIES = [
    CREATE_RIDER_STATS,
    CREATE_CUSTOMER_STATS,
    CREATE_ROUTE_STATS,
    CREATE_MONTHLY_SUMMARY,
    CREATE_SERVICE_RELIABILITY,
    CREATE_DEMAND_HEATMAP,
    CREATE_ASSIGNMENT_BY_SERVICE,
    CREATE_ASSIGNMENT_BY_HOUR,
]

def build_insight_tables(conn):
    """
    Drop and rebuild every summary table in the insights table.
    """
    cur = conn.cursor()
    for sql in INSIGHTS_TABLE_QUERIES:
        table = sql.split("CREATE TABLE")[1].split("AS")[0].strip()
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        cur.execute(sql)
        conn.commit()

# app-side queries

def get_filter_options(conn):
    """Distinct Type/Status values & the date bounds, for populating sidebar widgets."""

    types = pd.read_sql("SELECT DISTINCT Type FROM orders ORDER BY Type", conn)["Type"].tolist()
    statuses =  pd.read_sql("SELECT DISTINCT Status FROM orders ORDER BY Status", conn)["Status"].tolist()
    bounds = pd.read_sql("SELECT MIN(date(Created)) AS min_d, MAX(date(Created)) AS max_d FROM orders", conn).iloc[0]
    return types, statuses, bounds["min_d"], bounds["max_d"]

def get_filtered_orders(conn, date_from, date_to, types, statuses):
    """Orders matching the sidebar filters - a real WHERE query, not a pandas mask."""
    type_ph = ",".join("?" * len(types))
    status_ph = ",".join("?" * len(statuses))
    query = f"""
        SELECT * FROM orders
        WHERE date(Created) BETWEEN ? AND ?
        AND TYPE IN ({type_ph})
        AND STATUS IN ({status_ph})
    """
    params = [str(date_from), str(date_to), *types, *statuses]
    fdf = pd.read_sql(query, conn, params=params)
    fdf["Created"] = pd.to_datetime(fdf["Created"])
    return fdf

def get_filtered_rider_stats(conn, date_from, date_to, types, statuses):
    """Rider stats scoped to the same filter - also a query, not a re-filter of get_filtered_orders."""
    type_ph = ",".join("?" * len(types))
    status_ph = ",".join("?" * len(statuses))
    query = f"""
        SELECT
             Rider,
             COUNT(*)                                       AS total_orders,
             SUM(is_completed)                              AS completed_orders,
             SUM(is_refunded)                               AS refunded_orders,
             SUM(is_cancelled)                              AS cancelled_orders,
             AVG(Price)                                     AS avg_price,
             AVG(accept_min)                                AS avg_accept_min,
             ROUND(100.0 * SUM(is_completed) / COUNT(*), 1) AS completion_rate
        FROM orders
        WHERE Rider != 'Unassigned'
        AND date(Created) BETWEEN ? AND ?
        AND TYPE IN ({type_ph})
        AND STATUS IN ({status_ph})
        GROUP BY Rider
    """
    params = [str(date_from), str(date_to), *types, *statuses]
    return  pd.read_sql(query, conn, params=params)

GLOBAL_TABLE_NAMES = [
    "customer_stats", "route_stats", "monthly_summary",
    "service_reliability", "demand_heatmap", "accept_time_training",
    "assignment_by_service", "assignment_by_hour", "assignment_training",
]

def get_global_tables(conn):
    """Full-history insight tables (not scoped to sidebar filters) built by pipeline.py."""
    return {name: pd.read_sql(f"SELECT * FROM {name}", conn) for name in GLOBAL_TABLE_NAMES}
