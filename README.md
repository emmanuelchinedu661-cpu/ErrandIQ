# ErrandIQ

A multi-page BI dashboard and machine learning analytics suite built on real order data
from **ErrandMan**, an errand/delivery service operating in Umuahia, Nigeria. Built with
Python, SQLite, scikit-learn, and Streamlit.

## Live Features

- **Order intelligence dashboard** — six pages covering order status, revenue trends,
  rider performance, routes, and customer behavior, all filterable by date/type/status
- **Rider accept-time prediction** — a regression model estimating how long a rider
  will take to accept an order, with a live what-if predictor
- **Assignment-likelihood model** — a classification model predicting whether an order
  will ever get assigned a rider, using AUC (not just accuracy) to honestly evaluate
  performance on an imbalanced outcome
- **SQL-first architecture** — every insight table is built via real SQL queries
  (`CREATE TABLE ... AS SELECT`), not pandas aggregation, with parameterized queries
  for sidebar-filtered views

## Tech Stack

Python · SQLite · pandas · scikit-learn (Ridge regression, Logistic Regression) ·
Streamlit (multi-page, `st.navigation`) · Plotly · Font Awesome

## Project Structure
## Project Structure

- **`pipeline.py`** — Cleans the raw CSV, builds the SQLite database, trains both ML models
- **`queries.py`** — Every SQL statement in the project: table-creation queries and filtered app-side lookups
- **`charts.py`** — All Plotly chart-building functions, using a custom teal/gold/coral brand palette
- **`shared.py`** — Cached data loaders, the branded header, the filters expander, and KPI badge row
- **`pages_content.py`** — One function per dashboard page
- **`app.py`** — Entry point; wires pages into Streamlit's navigation system
- **`.streamlit/config.toml`** — App-wide dark theme matching the brand palette


## Setup
```bash
pip install -r requirements.txt
python pipeline.py
streamlit run app.py