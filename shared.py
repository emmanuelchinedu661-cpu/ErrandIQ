"""
shared.py — data loading, the filters expander, and the KPI row.
Every page in pages_content.py starts by calling render_filters() then render_kpis(fdf),
so filter state and KPI styling stay consistent across the whole app without duplicating code.
"""
import sqlite3
import pickle
from datetime import datetime as dt

import streamlit as st


import queries

DB_PATH = "errandiq.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

@st.cache_data
def load_filter_options():
    connection = get_connection()
    result = queries.get_filter_options(connection)
    return result

@st.cache_data
def load_filtered_orders(date_from, date_to, types, statuses):
    connection = get_connection()
    result = queries.get_filtered_orders(connection, date_from, date_to, types, statuses)
    return result

@st.cache_data
def load_filtered_rider_stats(date_from, date_to, types, statuses):
    connection = get_connection()
    result = queries.get_filtered_rider_stats(connection, date_from, date_to, types, statuses)
    return result

@st.cache_data
def load_global_tables():
    connection = get_connection()
    result = queries.get_global_tables(connection)
    return result

@st.cache_resource
def load_model():
    with open("accept_time_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_assignment_model():
    with open("assignment_model.pkl", "rb") as f:
        return pickle.load(f)

def render_header(page_title, icon_name):
    """Branded top bar — logo, app name, tagline. Call once at the top of every page,
    before render_filters(). Loads Font Awesome via CDN so icons work anywhere below."""
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
            <div style="padding:10px 4px 18px 4px; border-bottom:1px solid #1D3557; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="width:46px; height:46px; border-radius:50%; background:#2A9D8F;
                                display:flex; align-items:center; justify-content:center; border:3px solid #E9C46A;">
                        <i class="fa-solid fa-motorcycle" style="font-size:1.3rem; color:white;"></i>
                    </div>
                    <div>
                        <div style="font-size:1.3rem; font-weight:700; color:#F1FAEE;">ErrandIQ</div>
                        <div style="font-size:0.75rem; color:#A8DADC;">ErrandMan order analytics — Umuahia, Nigeria</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:10px; margin-top:14px;">
                    <i class="fa-solid {icon_name}" style="font-size:1.4rem; color:#E9C46A;"></i>
                    <div style="font-size:1.6rem; font-weight:700; color:#F1FAEE;">{page_title}</div>
                </div>
            </div>
            """,
        unsafe_allow_html=True,
    )

def render_filters():
    """Collapsible filter expander at the top of each page. Same widget keys everywhere —
       safe because only onepage runs per script execution, so there's no collision, and
       session_state keeps the selection consistent when you switch pages."""

    types_all, statuses_all, min_d, max_d = load_filter_options()

    c1, c2 = st.columns([0.15, 1])
    with c1:
        st.markdown(
            """
            <div style="width:40px; height:40px; border-radius:50%; background:#1D3557;
                        border:2px solid #E9C46A; display:flex; align-items:center; justify-content:center;">
                <i class="fa-solid fa-filter" style="color:#E9C46A; font-size:1.1rem;"></i>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        show_filters = st.checkbox("Filters", key="show_filters", value=False)
    if show_filters:
        c1, c2, c3 = st.columns([1.3, 1, 1])
        with c1:
            min_date = dt.strptime(str(min_d), "%Y-%m-%d").date()
            max_date = dt.strptime(str(max_d), "%Y-%m-%d").date()
            date_range = st.date_input(
                "Date range", (min_date, max_date),
                min_value=min_date, max_value=max_date,
                key="filter_date_range",
            )
            with c2:
                type_filter = st.multiselect("Order type", types_all, default=types_all, key="filter_type")
            with c3:
                status_filter = st.multiselect("Status", statuses_all, default=statuses_all, key="filter_status")
    else:
        date_range = st.session_state.get("filter_date_range", (
            dt.strptime(str(min_d), "%Y-%m-%d").date(),
            dt.strptime(str(max_d), "%Y-%m-%d").date(),
        ))
        type_filter = st.session_state.get("filter_type", types_all)
        status_filter = st.session_state.get("filter_status", statuses_all)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        date_from, date_to = date_range
    else:
        date_from = date_to = dt.strptime(str(min_d), "%Y-%m-%d").date()

    fdf = load_filtered_orders(date_from, date_to, type_filter, status_filter)
    rider_stats_f = load_filtered_rider_stats(date_from, date_to, type_filter, status_filter)
    st.caption(f"Showing {len(fdf)} orders matching current filters")
    return fdf, rider_stats_f, types_all, statuses_all

def render_kpis(fdf):
    """Restyled KPI row — bordered cards instead of plain st.metric columns."""
    completed_like = fdf["Status"].isin(["Completed", "Delivered"])
    completion_rate = completed_like.mean() * 100 if len(fdf) else 0
    refund_rate = (fdf["Status"] == "Refunded").mean() * 100 if len(fdf) else 0
    total_revenue = fdf.loc[completed_like, "Price"].sum()
    active_riders = fdf.loc[fdf["Rider"] != "Unassigned", "Rider"].nunique()

    kpis = [
        ("fa-box", "Total Orders", f"{len(fdf)}"),
        ("fa-check", "Completion Rate", f"{completion_rate:.0f}%"),
        ("fa-rotate-left", "Refund Rate", f"{refund_rate:.0f}%"),
        ("fa-sack-dollar", "Revenue", f"₦{total_revenue:.0f}"),
        ("fa-motorcycle", "Active Riders", f"{active_riders}"),
    ]

    cols = st.columns(5)
    for col, (icon_name, label, value) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""
                    <div style="display:flex; flex-direction:column; align-items:center; padding:8px;">
                        <div style="width:56px; height:56px; border-radius:50%; background:#2A9D8F;
                                    display:flex; align-items:center; justify-content:center;
                                    border:3px solid #E9C46A;">
                            <i class="fa-solid {icon_name}" style="font-size:1.4rem; color:white;"></i>
                        </div>
                        <div style="font-size:1.3rem; font-weight:700; margin-top:8px; color:#F1FAEE;">{value}</div>
                        <div style="font-size:0.75rem; color:#A8DADC; text-align:center;">{label}</div>
                    </div>
                    """,
                unsafe_allow_html=True,
            )

    st.markdown("")