"""
Build a self-contained analytics dashboard from cpcs.db.

Unlike the Streamlit app, this has no framework ceiling and no server:
it bakes the trip data into one HTML file rendered with ECharts. Open the
file in any browser. Fare and capacity are live client-side inputs, the
per-stop table sorts on header click, and you can switch between trips.

Run:
    python build_dashboard.py                 # reads cpcs.db -> cpcs_dashboard.html
    python build_dashboard.py --db my.db --out report.html

Then just double-click cpcs_dashboard.html.
"""

import argparse
import json
import sqlite3


def load(db):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    trips = [dict(r) for r in con.execute(
        "SELECT * FROM trips ORDER BY trip_id DESC")]
    stops = [dict(r) for r in con.execute(
        "SELECT * FROM stops ORDER BY trip_id, seq")]
    events = [dict(r) for r in con.execute("SELECT * FROM events")]
    con.close()
    by_trip = {}
    for t in trips:
        tid = t["trip_id"]
        by_trip[str(tid)] = {
            "trip": t,
            "stops": [s for s in stops if s["trip_id"] == tid],
            "methods": _method_counts(
                [e for e in events if e["trip_id"] == tid]),
        }
    return trips, by_trip


def _method_counts(evs):
    out = {}
    for e in evs:
        out[e["how"]] = out.get(e["how"], 0) + 1
    return out


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CPCS - passenger analytics</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.0/echarts.min.js"></script>
<style>
  :root{
    --bg:#0e1116; --panel:#161b22; --panel-2:#1c232d; --line:#232b36;
    --ink:#e6edf3; --muted:#8b98a5; --faint:#5b6774;
    --accent:#2dd4a7; --accent-dim:#1d9e75; --in:#2dd4a7; --out:#e08657;
    --flag:#f2555a; --flag-bg:rgba(242,85,90,0.12);
    --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  }
  html[data-theme="light"]{
    --bg:#f6f8fa; --panel:#ffffff; --panel-2:#f0f3f6; --line:#e2e8f0;
    --ink:#1a2029; --muted:#5b6774; --faint:#8b98a5;
    --flag-bg:rgba(242,85,90,0.10);
  }
  *{box-sizing:border-box;}
  body{margin:0; background:var(--bg); color:var(--ink);
       font-family:var(--sans); font-size:14px; line-height:1.5;}
  .wrap{max-width:1240px; margin:0 auto; padding:22px 26px 60px;}
  .topbar{display:flex; align-items:center; justify-content:space-between;
          gap:16px; flex-wrap:wrap; margin-bottom:6px;}
  .brand{font-size:20px; font-weight:600; letter-spacing:-0.01em;}
  .brand .dot{color:var(--accent);}
  .meta{color:var(--muted); font-size:13px; margin-bottom:20px;}
  .controls{display:flex; gap:14px; align-items:center; flex-wrap:wrap;}
  .controls label{color:var(--muted); font-size:12px; margin-right:6px;}
  select,input[type=number]{
    background:var(--panel-2); color:var(--ink); border:1px solid var(--line);
    border-radius:8px; padding:7px 10px; font-size:13px; font-family:var(--sans);}
  input[type=number]{width:78px;}
  .btn{background:var(--panel-2); color:var(--ink); border:1px solid var(--line);
       border-radius:8px; padding:7px 12px; font-size:13px; cursor:pointer;}
  .btn:hover{border-color:var(--accent-dim);}
  .kpis{display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin:18px 0 22px;}
  @media(max-width:900px){.kpis{grid-template-columns:repeat(3,1fr);}}
  @media(max-width:560px){.kpis{grid-template-columns:repeat(2,1fr);}}
  .kpi{background:var(--panel); border:1px solid var(--line); border-radius:12px;
       padding:14px 16px;}
  .kpi .l{color:var(--muted); font-size:12px;}
  .kpi .v{font-size:24px; font-weight:600; margin-top:3px; font-variant-numeric:tabular-nums;}
  .kpi.alert{border-color:var(--flag); background:var(--flag-bg);}
  .kpi.alert .v,.kpi.money .v{color:var(--flag);}
  .grid{display:grid; grid-template-columns:1fr 1fr; gap:16px;}
  @media(max-width:900px){.grid{grid-template-columns:1fr;}}
  .card{background:var(--panel); border:1px solid var(--line); border-radius:12px;
        padding:16px 18px;}
  .card h3{margin:0 0 2px; font-size:14px; font-weight:600;}
  .card .sub{color:var(--muted); font-size:12px; margin-bottom:10px;}
  .span2{grid-column:1 / -1;}
  .chart{width:100%; height:300px;}
  .chart.sm{height:240px;}
  table{width:100%; border-collapse:collapse; font-size:13px;}
  th,td{padding:8px 10px; text-align:right; border-bottom:1px solid var(--line);
        font-variant-numeric:tabular-nums;}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left;}
  th{color:var(--muted); font-weight:500; cursor:pointer; user-select:none;
     position:sticky; top:0; background:var(--panel);}
  th:hover{color:var(--ink);}
  tr.flagged td{background:var(--flag-bg);}
  .tag{color:var(--flag); font-weight:600;}
  .flag-item{padding:10px 0; border-bottom:1px solid var(--line);}
  .flag-item:last-child{border-bottom:none;}
  .flag-item b{color:var(--ink);}
  .money-note{margin-top:12px; padding:12px 14px; background:var(--panel-2);
              border-radius:10px; color:var(--muted); font-size:13px;}
  .money-note b{color:var(--flag);}
  .foot{color:var(--faint); font-size:12px; margin-top:26px; text-align:center;}
  .theme-toggle{cursor:pointer;}
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">CPCS<span class="dot">.</span> passenger analytics</div>
    <div class="controls">
      <span><label>Trip</label><select id="tripSel"></select></span>
      <span><label>Fare Rs</label><input type="number" id="fare" value="15" min="1" max="200"/></span>
      <span><label>Capacity</label><input type="number" id="cap" value="45" min="10" max="120"/></span>
      <button class="btn theme-toggle" id="themeBtn">Light</button>
    </div>
  </div>
  <div class="meta" id="meta"></div>

  <div class="kpis" id="kpis"></div>

  <div class="grid">
    <div class="card span2">
      <h3>Occupancy across the route</h3>
      <div class="sub">Dashed line = capacity, red marks = flagged stops</div>
      <div class="chart" id="occChart"></div>
    </div>
    <div class="card">
      <h3>Boardings and alightings</h3>
      <div class="sub">In above the axis, out below</div>
      <div class="chart sm" id="flowChart"></div>
    </div>
    <div class="card">
      <h3>Counting method</h3>
      <div class="sub">How each crossing was resolved</div>
      <div class="chart sm" id="methodChart"></div>
    </div>
    <div class="card span2">
      <h3>Per-stop log</h3>
      <div class="sub">Click a column header to sort</div>
      <div style="max-height:360px; overflow:auto;">
        <table id="stopTable"><thead></thead><tbody></tbody></table>
      </div>
    </div>
    <div class="card span2" id="flagCard">
      <h3>Revenue reconciliation</h3>
      <div class="sub">Camera counts vs ticket machine</div>
      <div id="flagBody"></div>
    </div>
  </div>

  <div class="foot">Processed locally on the edge device, no video leaves the bus,
    ticket-machine feed simulated for this proof of concept</div>
</div>

<script>
const DATA = /*__DATA__*/;
let sortKey = "seq", sortDir = 1;

const $ = id => document.getElementById(id);
const fmt = n => n.toLocaleString();

function metrics(stops, fare, cap){
  const inSum = stops.reduce((a,s)=>a+s.boardings,0);
  const outSum = stops.reduce((a,s)=>a+s.alightings,0);
  const peak = stops.reduce((a,s)=>Math.max(a,s.occupancy_after),0);
  const flagged = stops.filter(s=>s.flagged===1).length;
  const leak = stops.filter(s=>s.pos_count>=0)
                    .reduce((a,s)=>a+Math.max(0,s.discrepancy),0);
  return {inSum,outSum,peak,
          load:Math.round(100*peak/cap),
          flagged, leak, risk:leak*fare};
}

function renderKPIs(m){
  const cells = [
    ["Passengers served", fmt(m.inSum), ""],
    ["Total alightings", fmt(m.outSum), ""],
    ["Peak occupancy", fmt(m.peak), ""],
    ["Peak load factor", m.load+"%", ""],
    ["Stops flagged", m.flagged, m.flagged? "alert":""],
    ["Revenue at risk", "Rs "+fmt(m.risk), "money"],
  ];
  $("kpis").innerHTML = cells.map(([l,v,c])=>
    `<div class="kpi ${c}"><div class="l">${l}</div><div class="v">${v}</div></div>`
  ).join("");
}

function axisStyle(){
  const cs = getComputedStyle(document.documentElement);
  return {ink:cs.getPropertyValue('--ink').trim(),
          muted:cs.getPropertyValue('--muted').trim(),
          line:cs.getPropertyValue('--line').trim(),
          panel:cs.getPropertyValue('--panel').trim()};
}

let occ, flow, method;
function initCharts(){
  occ = echarts.init($("occChart"));
  flow = echarts.init($("flowChart"));
  method = echarts.init($("methodChart"));
  window.addEventListener("resize", ()=>{occ.resize();flow.resize();method.resize();});
}

function renderOcc(stops, cap){
  const c = axisStyle();
  const names = stops.map(s=>s.stop_name);
  const vals = stops.map(s=>s.occupancy_after);
  const flags = stops.filter(s=>s.flagged===1)
                     .map(s=>({value:[s.stop_name,s.occupancy_after]}));
  occ.setOption({
    grid:{left:44,right:16,top:18,bottom:64},
    tooltip:{trigger:"axis"},
    xAxis:{type:"category",data:names,axisLabel:{rotate:32,color:c.muted},
           axisLine:{lineStyle:{color:c.line}}},
    yAxis:{type:"value",name:"on board",nameTextStyle:{color:c.muted},
           axisLabel:{color:c.muted},splitLine:{lineStyle:{color:c.line}}},
    series:[{
      type:"line",data:vals,smooth:true,symbolSize:7,
      lineStyle:{width:2.5,color:"#2dd4a7"},itemStyle:{color:"#2dd4a7"},
      areaStyle:{color:"rgba(45,212,167,0.14)"},
      markLine:{silent:true,symbol:"none",lineStyle:{color:c.muted,type:"dashed"},
                data:[{yAxis:cap,name:"capacity"}],
                label:{color:c.muted,formatter:"capacity"}},
      markPoint:{symbol:"circle",symbolSize:16,
                 itemStyle:{color:"transparent",borderColor:"#f2555a",borderWidth:2},
                 data:flags}
    }]
  });
}

function renderFlow(stops){
  const c = axisStyle();
  const names = stops.map(s=>s.stop_name);
  flow.setOption({
    grid:{left:36,right:12,top:30,bottom:64},
    legend:{data:["In","Out"],top:0,textStyle:{color:c.muted},
            itemWidth:10,itemHeight:10},
    tooltip:{trigger:"axis",axisPointer:{type:"shadow"},
             formatter:p=>p.map(x=>`${x.seriesName}: ${Math.abs(x.value)}`).join("<br>")},
    xAxis:{type:"category",data:names,axisLabel:{rotate:32,color:c.muted},
           axisLine:{lineStyle:{color:c.line}}},
    yAxis:{type:"value",axisLabel:{color:c.muted,formatter:v=>Math.abs(v)},
           splitLine:{lineStyle:{color:c.line}}},
    series:[
      {name:"In",type:"bar",stack:"f",data:stops.map(s=>s.boardings),
       itemStyle:{color:"#2dd4a7"}},
      {name:"Out",type:"bar",stack:"f",data:stops.map(s=>-s.alightings),
       itemStyle:{color:"#e08657"}}
    ]
  });
}

function renderMethod(methods){
  const c = axisStyle();
  const palette={live:"#2dd4a7",fallback:"#e0a657",fallback_eof:"#e08657"};
  const data = Object.entries(methods).map(([k,v])=>
    ({name:k,value:v,itemStyle:{color:palette[k]||"#8b98a5"}}));
  method.setOption({
    tooltip:{trigger:"item",formatter:"{b}: {c} ({d}%)"},
    legend:{bottom:0,textStyle:{color:c.muted}},
    series:[{type:"pie",radius:["45%","70%"],center:["50%","44%"],
             label:{color:c.ink},data:data.length?data:[{name:"no events",value:1,
             itemStyle:{color:"#8b98a5"}}]}]
  });
}

function renderTable(stops){
  const cols=[["seq","#"],["stop_name","Stop"],["boardings","In"],
    ["alightings","Out"],["occupancy_after","Occ"],["pos_count","POS"],
    ["discrepancy","D"]];
  const thead=$("stopTable").querySelector("thead");
  thead.innerHTML="<tr>"+cols.map(([k,l])=>{
    const arrow = k===sortKey ? (sortDir>0?" \u25B2":" \u25BC") : "";
    return `<th data-k="${k}">${l}${arrow}</th>`;}).join("")+"</tr>";
  thead.querySelectorAll("th").forEach(th=>th.onclick=()=>{
    const k=th.dataset.k;
    if(k===sortKey) sortDir*=-1; else {sortKey=k; sortDir=1;}
    renderTable(stops);
  });
  const rows=[...stops].sort((a,b)=>{
    const x=a[sortKey],y=b[sortKey];
    return (x>y?1:x<y?-1:0)*sortDir;});
  const tb=$("stopTable").querySelector("tbody");
  tb.innerHTML=rows.map(s=>{
    const pos = s.pos_count<0 ? "\u2014" : s.pos_count;
    const d = s.pos_count<0 ? "" : (s.discrepancy>0?`<span class="tag">+${s.discrepancy}</span>`:s.discrepancy);
    return `<tr class="${s.flagged?'flagged':''}"><td>${s.seq}</td>
      <td>${s.stop_name}</td><td>${s.boardings}</td><td>${s.alightings}</td>
      <td>${s.occupancy_after}</td><td>${pos}</td><td>${d}</td></tr>`;
  }).join("");
}

function renderFlags(stops, fare, m){
  const flagged = stops.filter(s=>s.flagged===1);
  const body=$("flagBody");
  if(!flagged.length){
    body.innerHTML=`<div style="color:var(--accent)">Camera and ticket counts agree at every stop.</div>`;
    return;
  }
  let html = flagged.map(s=>{
    if(s.discrepancy>0)
      return `<div class="flag-item"><b>${s.stop_name}</b> - camera counted
        <b>${s.boardings}</b> boardings, ticket machine sold <b>${s.pos_count}</b>.
        ${s.discrepancy} unticketed, Rs ${s.discrepancy*fare} at risk.</div>`;
    return `<div class="flag-item"><b>${s.stop_name}</b> - camera ${s.boardings},
      ticket machine ${s.pos_count} (camera under-counted by ${Math.abs(s.discrepancy)}, review).</div>`;
  }).join("");
  html += `<div class="money-note">An estimated <b>${m.leak} unticketed passengers</b>
    this trip represent <b>Rs ${fmt(m.risk)}</b> in potential leakage. Scaled across a
    500-bus fleet running multiple trips daily, this is the core ROI case.</div>`;
  body.innerHTML=html;
}

function draw(){
  const tid = $("tripSel").value;
  const fare = +$("fare").value, cap = +$("cap").value;
  const bundle = DATA.by_trip[tid];
  const stops = bundle.stops;
  const t = bundle.trip;
  $("meta").textContent = `${t.route} - bus ${t.bus_id} - started ${t.started_at}`;
  const m = metrics(stops, fare, cap);
  renderKPIs(m);
  renderOcc(stops, cap);
  renderFlow(stops);
  renderMethod(bundle.methods);
  renderTable(stops);
  renderFlags(stops, fare, m);
}

function boot(){
  const sel=$("tripSel");
  sel.innerHTML=DATA.trips.map(t=>
    `<option value="${t.trip_id}">Trip #${t.trip_id} - ${t.route} - ${t.bus_id}</option>`
  ).join("");
  initCharts();
  sel.onchange=draw;
  $("fare").oninput=draw;
  $("cap").oninput=draw;
  $("themeBtn").onclick=()=>{
    const h=document.documentElement;
    const dark = h.getAttribute("data-theme")==="dark";
    h.setAttribute("data-theme", dark?"light":"dark");
    $("themeBtn").textContent = dark?"Dark":"Light";
    draw();
  };
  draw();
}
boot();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cpcs.db")
    ap.add_argument("--out", default="cpcs_dashboard.html")
    args = ap.parse_args()

    trips, by_trip = load(args.db)
    if not trips:
        print(f"No trips found in {args.db}. Run cpcs_poc.py first.")
        return

    payload = {"trips": trips, "by_trip": by_trip}
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(payload))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}  ({len(trips)} trip(s))")
    print(f"open it:  double-click {args.out}  (or drag into a browser tab)")


if __name__ == "__main__":
    main()
