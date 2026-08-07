"""
charts.py — all Plotly figure builders for ErrandIQ.
Each function takes a dataframe and returns a Plotly figure.
Keeping these separate from app.py means app.py only handles layout/state.
"""
import plotly.express as px
import plotly.graph_objs as go

STATUS_COLORS = {
    "Completed": "#2ECC71", "Delivered": "#27AE60", "Assigned": "#F39C12",
    "Picked Up": "#3498DB", "Cancelled": "#E74C3C", "Refunded": "#95A5A6",
}

def status_bar(fdf):
    counts = fdf["Status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    fig = px.bar(counts, x="Count", y="Status", orientation="h", color="Status",
                 color_discrete_map=STATUS_COLORS, title="Orders by Status")
    fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    return fig

def type_pie(fdf):
    counts = fdf["Type"].value_counts().reset_index()
    counts.columns = ["Type", "Count"]
    return px.pie(counts, names="Type", values="Count", title="Errand vs Waybill Split",
                  color_discrete_sequence=["#3498DB", "#F39C12"], hole=0.45)

def weekly_volume(fdf):
    trend = fdf.set_index("Created").resample("W")["ID"].count().reset_index()
    trend.columns = ["Week", "Orders"]
    fig = px.area(trend, x="Week", y="Orders", title="Weekly Order Volume")
    fig.update_traces(line_color="#3498DB", fillcolor="rgba(52,152,219,0.25)")
    return fig

def demand_heatmap(demand_df):
    order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = demand_df.pivot(index="created_dow", columns="created_hour", values="order_count").fillna(0)
    pivot = pivot.reindex(order_days)
    return px.imshow(pivot, aspect="auto", color_continuous_scale="Blues",
                     title="Order Demand Heatmap (Day * Hour)", labels=dict(color="Orders"))

def service_popularity(fdf):
    counts = fdf["Title"].value_counts().reset_index()
    counts.columns = ["Errand/Service Type", "Count"]
    return px.bar(counts, x="Errand/Service Type", y="Count", title="Most Requested Services",
                  color="Count", color_continuous_scale="Tealgrn")

def monthly_revenue(monthly_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_df["created_month"], y=monthly_df["revenue"],
                         name="Revenue (₦)", marker=dict(color="#2ECC71")))
    fig.add_trace(go.Scatter(x=monthly_df["created_month"], y=monthly_df["completion_rate"],
                             name="Completion Rate (%)", yaxis="y2", line=dict(color="#E67E22", width=3)))
    fig.update_layout(
        title="Monthly Revenue vs Completion Rate",
        yaxis=dict(title="Revenue (₦)"),
        yaxis2=dict(title="Completion Rate (%)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", y=1.1),
    )
    return fig

def service_reliability_chart(rel_df):
    melted = rel_df.melt(id_vars="Title", value_vars=["refund_rate", "cancel_rate", "completion_rate"],
                          var_name="Metric", value_name="Rate (%)")
    fig = px.bar(melted, x="Title", y="Rate (%)", color="Metric", barmode="group",
                 title="Service Reliability — Refund / Cancel / Completion Rate by Type",
                 color_discrete_map={"refund_rate": "#95A5A6", "cancel_rate": "#E74C3C", "completion_rate": "#2ECC71"})
    fig.update_xaxes(tickangle=-20)
    return fig

def rider_leaderboard(rider_stats_df):
    df = rider_stats_df.sort_values("total_orders", ascending=True)
    return px.bar(df, x="total_orders", y="Rider", orientation="h",
                  title="Rider Leaderboard - Orders Handled", color="total_orders",
                  color_continuous_scale="Purp")

def rider_outcomes(rdf):
    grouped = rdf.groupby(["Rider", "Status"]).size().reset_index(name="Count")
    fig = px.bar(grouped, x="Rider", y="Count", color="Status",
                 title="Rider Outcomes by Status", color_discrete_map=STATUS_COLORS, barmode="stack")
    fig.update_xaxes(tickangle=-30)
    return fig

def rider_accept_box(accept_df):
    fig = px.box(accept_df, x="Rider", y="accept_min", title="Accept Time Distribution by Rider (minutes)",
                 color="Rider")
    fig.update_layout(showlegend=False)
    return fig

def rider_completion_rate(rider_stats_df):
    df = rider_stats_df.sort_values("completion_rate", ascending=True)
    fig = px.bar(df, x="completion_rate", y="Rider", orientation="h",
                 title="Rider Completion Rate (%)", color="completion_rate",
                 color_continuous_scale="RdYlGn", range_color=[0, 100])
    return fig

def top_locations(loc_counts, title, color_scale):
    fig = px.bar(loc_counts, x="Count", y=loc_counts.columns[0], orientation="h",
                 title=title, color="Count", color_continuous_scale=color_scale)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig

def top_routes(route_stats_df, n=10):
    top = route_stats_df.head(n).copy()
    top["route"] = top["From"].str.slice(0, 25) + " → " + top["To"].str.slice(0, 25)
    fig = px.bar(top, x="trip_count", y="route", orientation="h",
                 title="Most Repeated Routes", color="avg_price", color_continuous_scale="Sunset",
                 labels={"trip_count": "Trips", "route": "Route", "avg_price": "Avg Price (₦)"})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig

def price_histogram(fdf):
    return px.histogram(fdf, x="Price", nbins=20, title="Order Price Distribution",
                        color_discrete_sequence=["#16A085"])

def repeat_customers(customer_stats_df, n=10):
    top = customer_stats_df.head(n)
    fig = px.bar(top, x="order_count", y="Customer", orientation="h",
                 title="Top Customers by Order Count", color="is_repeat", color_discrete_map={True: "#2ECC71", False: "#BDC3C7"},
                 labels={"order_count": "Order", "is_repeat": "Repeat Customer"})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig

def model_fit_diagnostic(train_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df["accept_min"], y=train_df["cv_pred_min"], mode="markers",
                             marker=dict(size=10, color="#3498DB"), name="Orders"))
    max_v = max(train_df["accept_min"].max(), train_df["cv_pred_min"].max())
    fig.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines",
                             line=dict(dash="dash", color="gray"), name="Perfect Prediction"))
    fig.update_layout(xaxis_title="Actual accept time (min)", yaxis_title="Cross-validated predicted time(min)",
                      title="Model Fit Diagnostic")
    return fig

# Assignment-likelihood model charts---

def assignment_rate_by_service(assign_service_df):
    df = assign_service_df.sort_values("assignment_rate", ascending=True)
    fig = px.bar(df, x="assignment_rate", y="Title", orientation="h",
                 title="Assignment Rate by Service Type (%)", color="assignment_rate",
                 color_continuous_scale="RdYlGn", range_color=[0, 100],
                 hover_data=["order_count", "assigned_count"])
    return fig

def assignment_rate_by_hour(assign_hour_df):
    fig = px.bar(assign_hour_df, x="created_hour", y="assignment_rate",
                 title="Assignment Rate by Hour of Day (%)", color="assignment_rate",
                 color_continuous_scale="RdYlGn", range_color=[0, 100],
                 labels={"created_hour": "Hour", "assignment_rate": "Assignment Rate (%)"})
    return fig

def assignment_model_diagnostic(assign_training_df):
    df = assign_training_df.copy()
    df["Outcome"] = df["was_assigned"].map({1: "Assigned", 0: "Not Assigned"})
    fig = px.box(df, x="Outcome", y="predicted_proba", color="Outcome",
                 title="Assignment Model: Predicted Probability by Actual Outcome",
                 color_discrete_map={"Assigned": "#2ECC71", "Not Assigned": "#E74C3C"},
                 labels={"predicted_proba": "Predicted Probability of assignment"})
    fig.update_layout(showlegend=False)
    return fig