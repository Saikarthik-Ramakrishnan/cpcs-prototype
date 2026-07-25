"""
CPCS PoC dashboard — reads cpcs.db and presents a trip the way an operator sees it.

Run:
    streamlit run dashboard.py
If the file-watcher crashes in your Anaconda env:
    streamlit run dashboard.py --server.fileWatcherType none
Different DB file:
    streamlit run dashboard.py -- --db mytrip.db
"""

import argparse
import sqlite3
import sys

import altair as alt
import pandas as pd
import streamlit as st

_ap = argparse.ArgumentParser()
_ap.add_argument("--db", default="cpcs.db")
_args, _ = _ap.parse_known_args(sys.argv[1:])
DB = _args.db

ACCENT = "#0F6E56"
IN_COLOR = "#1D9E75"
OUT_COLOR = "#D85A30"
FLAG_COLOR = "#A32D2D"

st.set_page_config(page_title="CPCS — passenger analytics",
                   page_icon="🚌", layout="wide")

st.markdown(f"""
<style>
  #MainMenu, footer, header {{visibility: hidden;}}
  .block-container {{padding-top: 2rem; max-width: 1100px;}}
  .cpcs-title {{font-size: 26px; font-weight: 600; margin-bottom: 2px;}}
  .cpcs-sub {{color: var(--text-color); opacity: 0.6; font-size: 14px;
              margin-bottom: 18px;}}
  .kpi-grid {{display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr));
              gap: 12px; margin: 8px 0 22px;}}
  .kpi {{background: var(--secondary-background-color); border-radius: 12px;
         padding: 16px 18px;}}
  .kpi .l {{font-size: 13px; opacity: 0.65;}}
  .kpi .v {{font-size: 26px; font-weight: 600; margin-top: 2px;}}
  .kpi.alert {{background: rgba(163,45,45,0.12);}}
  .kpi.alert .v {{color: {FLAG_COLOR};}}
  .kpi.money .v {{color: {FLAG_COLOR};}}
  .sec {{font-size: 17px; font-weight: 600; margin: 26px 0 8px;}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=2)
def load(db):
    con = sqlite3.connect(db)
    trips = pd.read_sql_query("SELECT * FROM trips ORDER BY trip_id DESC", con)
    stops = pd.read_sql_query("SELECT * FROM stops", con)
    events = pd.read_sql_query("SELECT * FROM events", con)
    con.close()
    return trips, stops, events


def compute_metrics(s, fare, capacity):
    valid_pos = s[s.pos_count >= 0]
    leak = valid_pos.discrepancy.clip(lower=0).sum()
    return {
        "total_in": int(s.boardings.sum()),
        "total_out": int(s.alightings.sum()),
        "peak": int(s.occupancy_after.max()),
        "load_factor": round(100 * s.occupancy_after.max() / capacity),
        "flagged": int(s.flagged.sum()),
        "leak_pax": int(leak),
        "revenue_risk": int(leak * fare),
        "n_stops": len(s),
    }


# ---------------- sidebar controls ----------------
st.sidebar.header("Assumptions")
fare = st.sidebar.number_input("Fare per ticket (₹)", 1, 200, 15, 1)
capacity = st.sidebar.number_input("Bus capacity (pax)", 10, 120, 45, 1)
if st.sidebar.button("↻ Refresh data"):
    st.cache_data.clear()
st.sidebar.caption(f"Reading `{DB}`")

# ---------------- load ----------------
try:
    trips, stops, events = load(DB)
except Exception:
    st.error(f"Could not read {DB}. Run cpcs_poc.py first to log a trip.")
    st.stop()
if trips.empty:
    st.warning("No trips yet. Run cpcs_poc.py to log one.")
    st.stop()

st.markdown('<div class="cpcs-title">Camera-based passenger counting</div>',
            unsafe_allow_html=True)

labels = {int(r.trip_id): f"Trip #{int(r.trip_id)} · {r.route} · {r.bus_id}"
          for r in trips.itertuples()}
tid = st.selectbox("Trip", list(labels), format_func=lambda t: labels[t],
                   label_visibility="collapsed")
trip_row = trips[trips.trip_id == tid].iloc[0]
s = stops[stops.trip_id == tid].sort_values("seq").reset_index(drop=True)

st.markdown(
    f'<div class="cpcs-sub">{trip_row.route} · bus {trip_row.bus_id} · '
    f'started {trip_row.started_at}</div>', unsafe_allow_html=True)

if s.empty:
    st.info("This trip has no committed stops yet.")
    st.stop()

m = compute_metrics(s, fare, capacity)

# ---------------- KPI cards ----------------
alert_cls = "kpi alert" if m["flagged"] else "kpi"
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><div class="l">Passengers served</div><div class="v">{m['total_in']}</div></div>
  <div class="kpi"><div class="l">Total alightings</div><div class="v">{m['total_out']}</div></div>
  <div class="kpi"><div class="l">Peak occupancy</div><div class="v">{m['peak']}</div></div>
  <div class="kpi"><div class="l">Peak load factor</div><div class="v">{m['load_factor']}%</div></div>
  <div class="{alert_cls}"><div class="l">Stops flagged</div><div class="v">{m['flagged']}</div></div>
  <div class="kpi money"><div class="l">Revenue at risk</div><div class="v">₹{m['revenue_risk']}</div></div>
</div>
""", unsafe_allow_html=True)

order = list(s.stop_name)

# ---------------- occupancy across route ----------------
st.markdown('<div class="sec">Occupancy across the route</div>',
            unsafe_allow_html=True)
occ = s[["stop_name", "seq", "occupancy_after", "flagged"]].copy()
base = alt.Chart(occ).encode(
    x=alt.X("stop_name:N", sort=order, title=None,
            axis=alt.Axis(labelAngle=-35)))
area = base.mark_area(opacity=0.15, color=ACCENT).encode(
    y=alt.Y("occupancy_after:Q", title="passengers on board"))
line = base.mark_line(color=ACCENT, strokeWidth=2.5).encode(
    y="occupancy_after:Q")
pts = base.mark_point(color=ACCENT, filled=True, size=60).encode(
    y="occupancy_after:Q",
    tooltip=["stop_name", "occupancy_after"])
flags = alt.Chart(occ[occ.flagged == 1]).mark_point(
    color=FLAG_COLOR, size=170, strokeWidth=2, filled=False).encode(
    x=alt.X("stop_name:N", sort=order), y="occupancy_after:Q")
cap = alt.Chart(pd.DataFrame({"y": [capacity]})).mark_rule(
    color="#888", strokeDash=[4, 4]).encode(y="y:Q")
st.altair_chart((area + line + pts + flags + cap).properties(height=280),
                use_container_width=True)
st.caption("Dashed line = bus capacity. Red rings = stops flagged for "
           "camera/ticket mismatch.")

# ---------------- boardings vs alightings (diverging) ----------------
st.markdown('<div class="sec">Boardings and alightings per stop</div>',
            unsafe_allow_html=True)
flow = s[["stop_name", "seq", "boardings", "alightings"]].copy()
flow["alightings"] = -flow["alightings"]
flow_long = flow.melt(id_vars=["stop_name", "seq"],
                      value_vars=["boardings", "alightings"],
                      var_name="type", value_name="count")
flow_long["type"] = flow_long["type"].map(
    {"boardings": "In", "alightings": "Out"})
bar = alt.Chart(flow_long).mark_bar().encode(
    x=alt.X("stop_name:N", sort=order, title=None,
            axis=alt.Axis(labelAngle=-35)),
    y=alt.Y("count:Q", title="out  ·  in"),
    color=alt.Color("type:N",
                    scale=alt.Scale(domain=["In", "Out"],
                                    range=[IN_COLOR, OUT_COLOR]),
                    legend=alt.Legend(title=None, orient="top")),
    tooltip=["stop_name", "type", alt.Tooltip("count:Q")])
st.altair_chart(bar.properties(height=260), use_container_width=True)

# ---------------- per-stop table ----------------
st.markdown('<div class="sec">Per-stop log</div>', unsafe_allow_html=True)
show = s[["seq", "stop_name", "boardings", "alightings", "occupancy_after",
          "pos_count", "discrepancy", "flagged"]].copy()
show["pos_count"] = show["pos_count"].replace(-1, pd.NA)
show = show.rename(columns={
    "seq": "#", "stop_name": "Stop", "boardings": "In", "alightings": "Out",
    "occupancy_after": "Occupancy", "pos_count": "POS",
    "discrepancy": "Δ", "flagged": "Flag"})


def paint(row):
    hit = show.loc[row.name, "Flag"] == 1
    return [f"background-color: rgba(163,45,45,0.16)" if hit else ""] * len(row)


st.dataframe(
    show.drop(columns=["Flag"]).style.apply(paint, axis=1)
        .format({"POS": lambda v: "—" if pd.isna(v) else f"{int(v)}"}),
    use_container_width=True, hide_index=True)

# ---------------- flagged detail + revenue story ----------------
if m["flagged"]:
    st.markdown('<div class="sec">Flagged stops — revenue reconciliation</div>',
                unsafe_allow_html=True)
    for r in s[s.flagged == 1].itertuples():
        gap = int(r.discrepancy)
        if gap > 0:
            st.markdown(
                f"**{r.stop_name}** — camera counted **{int(r.boardings)}** "
                f"boardings, ticket machine sold **{int(r.pos_count)}**. "
                f"{gap} unticketed → ₹{gap * fare} at risk.")
        else:
            st.markdown(
                f"**{r.stop_name}** — camera counted {int(r.boardings)}, "
                f"ticket machine sold {int(r.pos_count)} "
                f"(camera under-counted by {abs(gap)}; worth reviewing).")
    st.info(f"Across this trip, an estimated **{m['leak_pax']} unticketed "
            f"passengers** represent **₹{m['revenue_risk']}** in potential "
            f"revenue leakage. Scaled to a 500-bus fleet running "
            f"multiple trips a day, this is the operator's core ROI case.")
else:
    st.success("Camera and ticket counts agree at every stop.")

# ---------------- counting-method breakdown (technical credibility) ----------------
ev = events[events.trip_id == tid]
if not ev.empty:
    with st.expander("How were these counts derived?"):
        by_how = ev.groupby("how").size()
        st.write("Crossing events by detection path:")
        st.bar_chart(by_how)
        st.caption(
            "`live` = counted while continuously tracked. "
            "`fallback` / `fallback_eof` = recovered by the birth-to-death "
            "safety net when a track ended near the line. A healthy setup is "
            "mostly `live`; a high fallback share flags detection dropouts to "
            "fix with camera placement or a stronger detector.")

st.caption("All processing runs locally on the edge device. "
           "No video leaves the bus.")
