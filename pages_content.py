"""
pages_content.py — one function per page. app.py wires these into st.navigation().
Each page calls shared.render_filters() and shared.render_kpis() first, then its own charts.
"""

import streamlit as st
import pandas as pd
import numpy as np

import charts
import shared

def overview_page():
    shared.render_header("Overview", "fa-chart-column")
    fdf, rider_stats_f, types_all, statuses_all = shared.render_filters()
    shared.render_kpis(fdf)

    global_tables = shared.load_global_tables()

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.status_bar(fdf), use_container_width=True)
    c2.plotly_chart(charts.type_pie(fdf), use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.weekly_volume(fdf), use_container_width=True)
    c4.plotly_chart(charts.demand_heatmap(global_tables["demand_heatmap"]), use_container_width=True)

    st.plotly_chart(charts.service_popularity(fdf), use_container_width=True)

def revenue_page():
    shared.render_header("Revenue & Reliability", "fa-sack-dollar")
    fdf, rider_stats_f, types_all, statuses_all = shared.render_filters()
    shared.render_kpis(fdf)

    global_tables = shared.load_global_tables()

    st.plotly_chart(charts.monthly_revenue(global_tables["monthly_summary"]), use_container_width=True)
    st.plotly_chart(charts.service_reliability_chart(global_tables["service_reliability"]),use_container_width=True)

    st.markdown("## Riskiest Services")
    risky = global_tables["service_reliability"].sort_values("refund_rate", ascending=False).head(3)
    st.dataframe(risky[["Title", "order_count", "refund_rate", "cancel_rate", "completion_rate"]],
                 use_container_width=True, hide_index=True)
    st.caption("Highest refund-rate services - worth investigating if refunds cluster around specific vendors or routes.")

def riders_page():
    shared.render_header("Riders", "fa-motorcycle")
    fdf, rider_stats_f, types_all, statuses_all = shared.render_filters()
    shared.render_kpis(fdf)

    rdf = fdf[fdf["Rider"] != "Unassigned"]

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.rider_leaderboard(rider_stats_f), use_container_width=True)
    c2.plotly_chart(charts.rider_completion_rate(rider_stats_f), use_container_width=True)

    st.plotly_chart(charts.rider_outcomes(rdf), use_container_width=True)

    accept_df = rdf[rdf["accept_min"].notna() & (rdf["accept_min"] < 60 * 24 * 3)]
    if len(accept_df):
        st.plotly_chart(charts.rider_accept_box(accept_df), use_container_width=True)
    else:
        st.info("No accept-time data available for the current filter selection.")

def routes_page():
    shared.render_header("Routes & Customers", "fa-location-dot")
    fdf, routes_stats_f, types_all, statuses_all = shared.render_filters()
    shared.render_kpis(fdf)

    global_tables = shared.load_global_tables()

    c1, c2 = st.columns(2)
    with c1:
        top_from = fdf["From"].dropna().value_counts().head(10).reset_index()
        top_from.columns = ["Pickup Location", "Count"]
        st.plotly_chart(charts.top_locations(top_from, "Top Pickup Locations", "Oranges"), use_container_width=True)
    with c2:
        top_to = fdf["To"].dropna().value_counts().head(10).reset_index()
        top_to.columns = ["Drop-off Location", "Count"]
        st.plotly_chart(charts.top_locations(top_to, "Top Drop-off Locations", "Blues"), use_container_width=True)

    st.plotly_chart(charts.top_routes(global_tables["route_stats"]), use_container_width=True)
    st.plotly_chart(charts.repeat_customers(global_tables["customer_stats"]), use_container_width=True)

    n_repeat = global_tables["customer_stats"]["is_repeat"].sum()
    n_total = len(global_tables["customer_stats"])
    st.caption(f"{n_repeat} of {n_total} named customers ({n_repeat/n_total*100:.0f}%) have placed more than one order.")

    st.plotly_chart(charts.price_histogram(fdf), use_container_width=True)

    st.subheader("Order Log")
    show_cols = ["ID", "Type", "Title", "Status", "Rider", "Price", "From", "To", "Created"]
    st.dataframe(fdf[show_cols].sort_values("Created", ascending=False), use_container_width=True, hide_index=True)

def accept_time_page():
    shared.render_header("Accept-Time Model", "fa-stopwatch")
    fdf, rider_stats_f, types_all, statuses_all = shared.render_filters()
    shared.render_kpis(fdf)

    global_tables = shared.load_global_tables()
    model_bundle = shared.load_model()

    st.subheader("Rider Accept-Time Prediction")
    st.caption(
        f"Trained on {model_bundle['n_train']} labeled orders (only orders with a recorded rider-assignment "
        "timestamp carry this level). With this little data, treat prediction as directional, not precise."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Model MAE (cross-validated)", f"{model_bundle['mae_minutes']:.1f} min")
    m2.metric("Naive baseline MAE", f"{model_bundle['baseline_mae_minutes']:.1f} min")
    m3.metric("R² (log space)", f"{model_bundle['r2_log']:.2f}")

    if model_bundle["mae_minutes"] < model_bundle["baseline_mae_minutes"]:
        st.success("Model beats a naive 'always predict the median' baseline - modestly, given the small sample.")
    else:
        st.warning("Model does not clearly beat the naive baseline yet. More labeled data would help most.")

    st.markdown("## Try a what-if prediction")
    riders = sorted(pd.concat([rider_stats_f["Rider"], pd.Series(["Unassigned"])]).unique())
    dows = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    c1, c2, c3 = st.columns(3)
    with c1:
        in_type = st.selectbox("Order type", types_all, key="at_type")
        in_title = st.selectbox("Service", sorted(fdf["Title"].unique()) if len(fdf) else [], key="at_title")
    with c2:
        in_rider = st.selectbox("Rider", riders, key="at_rider")
        in_price = st.number_input("Price (₦)", min_value=0, value=1500, step=100, key="at_price")

    with c3:
        in_dow = st.selectbox("Day of week", dows, key="at_dow")
        in_hour = st.slider("Hour of day", 0, 23, 14, key="at_hour")

    in_weekend = in_dow in ["Saturday", "Sunday"]
    x_new = pd.DataFrame([{
        "Type": in_type, "Title": in_title, "Rider": in_rider, "Price": in_price,
        "created_hour": in_hour, "created_dow": in_dow, "is_weekend": in_weekend,
        "has_from": True, "has_to": True,
    }])[model_bundle["features"]]

    pred_log = model_bundle["model"].predict(x_new)[0]
    pred_min = float(np.expm1(pred_log))
    st.metric("Predicted accept time", f"{pred_min:.0f} minutes", help="±~27 min typical error based on cross-validation")

    st.markdown("## Training data: predicted vs actual")
    st.plotly_chart(charts.model_fit_diagnostic(global_tables["accept_time_training"]), use_container_width=True)

def assignment_page():
    shared.render_header("Assignment Insights", "fa-bullseye")
    fdf, rider_stats_f, types_all, statuses_all = shared.render_filters()
    shared.render_kpis(fdf)

    global_tables = shared.load_global_tables()
    assign_model_bundle = shared.load_assignment_model()

    st.subheader("Will This Order Get Assigned?")
    st.caption(
        f"Trained on all {assign_model_bundle['n_total']} orders ({assign_model_bundle['n_assigned']} were "
        "ever assigned a rider). Rider and Status are excluded as features since they're only known "
        "AFTER assignment happens - including them would let the model 'cheat' by looking at the outcome."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Model AUC (cross-validated)", f"{assign_model_bundle['auc']:.3f}",
              help="0.5 = random guessing, 1.0 = perfect")
    m2.metric("Accuracy", f"{assign_model_bundle['accuracy']:.1%}")
    m3.metric("Majority-class baseline accuracy", f"{assign_model_bundle['baseline_accuracy']:.1%}")

    if assign_model_bundle["auc"] > 0.6:
        st.success("Model shows modest but real signal above random guessing (AUC > 0.6).")
    else:
        st.warning("Model is close to random guessing - treat prediction cautiously.")

    st.caption(
        "Note: accuracy alone is misleading here because most orders are never assigned. "
        "AUC is the more honest metric — it measures whether the model ranks likely-to-be-assigned "
        "orders above unlikely ones, regardless of class imbalance."
    )

    st.markdown("## Assignment rate by service and time")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.assignment_rate_by_service(global_tables["assignment_by_service"]), use_container_width=True)
    c2.plotly_chart(charts.assignment_rate_by_hour(global_tables["assignment_by_hour"]), use_container_width=True)

    st.markdown("## Model diagnostic")
    st.plotly_chart(charts.assignment_model_diagnostic(global_tables["assignment_training"]), use_container_width=True)

    st.markdown("## Try a what-if prediction")
    dows = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    c1, c2, c3 = st.columns(3)
    with c1:
        a_type = st.selectbox("Order type", types_all, key="assign_type")
        a_title = st.selectbox("Service", sorted(fdf["Title"].unique()) if len(fdf) else [], key="assign_title")
    with c2:
        a_price = st.number_input("Price (₦)", min_value=0, value=1500, step=100, key="assign_price")
        a_hour = st.slider("Hour of day", 0, 23, 14, key="assign_hour")
    with c3:
        a_dow = st.selectbox("Day of week", dows, key="assign_dow")
        a_has_route = st.checkbox("Has pickup/dropoff route specified", value=True, key="assign_route")

    a_weekend = a_dow in ["Saturday", "Sunday"]
    x_assign_new = pd.DataFrame([{
        "Type": a_type, "Title": a_title, "Price": a_price,
        "created_hour": a_hour, "created_dow": a_dow, "is_weekend": a_weekend,
        "has_from": a_has_route, "has_to": a_has_route,
    }])[assign_model_bundle["features"]]

    a_proba = assign_model_bundle["model"].predict_proba(x_assign_new)[0][1]
    st.metric("Predicted probability of assignment", f"{a_proba:.0%}")

