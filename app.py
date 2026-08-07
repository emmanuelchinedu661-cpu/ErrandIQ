"""
app.py — entry point only. All actual page content lives in pages_content.py;
shared data-loading and the filters/KPI widgets live in shared.py.
Run with: streamlit run app.py
"""
import streamlit as st

import pages_content

st.set_page_config(page_title="ErrandIQ", page_icon="🛵", layout="wide")

pages = [
    st.Page(pages_content.overview_page, title="Overview", icon=":material/bar_chart:", default=True),
    st.Page(pages_content.revenue_page, title="Revenue & Reliability", icon=":material/payments:"),
    st.Page(pages_content.riders_page, title="Riders", icon=":material/two_wheeler:"),
    st.Page(pages_content.routes_page, title="Routes & Customers", icon=":material/location_on:"),
    st.Page(pages_content.accept_time_page, title="Accept-Time Model", icon=":material/timer:"),
    st.Page(pages_content.assignment_page, title="Assignment Insights", icon=":material/track_changes:"),
]

pg = st.navigation(pages, position="top")
pg.run()