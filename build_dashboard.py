"""
Build a self-contained municipal-grade analytics report from cpcs.db.

Design targets (city-corporation users):
  - 5-second rule: the top KPI strip answers "how are we doing" unscrolled
  - Two levels: Fleet overview (commissioner) -> Trip detail (ops/revenue)
  - Bilingual labels: English / Hindi toggle
  - Print-ready: one click produces a paper report (light theme, no controls)
  - Auditability: every count is tagged live/coast/stitch/fallback; the report
    shows a Data Confidence figure instead of an accuracy claim
  - Accessibility: color-blind-safe palette; flagged stops marked by glyph and
    shading, never color alone; tabular numerals throughout

Run:
    python build_dashboard.py                    # cpcs.db -> cpcs_dashboard.html
    python build_dashboard.py --db my.db --out report.html
Open the HTML in any browser. No server needed.
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
    events = [dict(r) for r in con.execute(
        "SELECT trip_id, stop_seq, frame, direction, how FROM events")]
    con.close()
    by_trip = {}
    for t in trips:
        tid = t["trip_id"]
        by_trip[str(tid)] = {
            "trip": t,
            "stops": [s for s in stops if s["trip_id"] == tid],
            "events": [e for e in events if e["trip_id"] == tid],
        }
    return trips, by_trip


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
    --accent:#2dd4a7; --accent-dim:#1d9e75;
    --in:#2dd4a7; --out:#e08657; --coast:#c9a227; --fall:#8d76d8;
    --flag:#f2555a; --flag-bg:rgba(242,85,90,0.12);
    --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans',
           'Noto Sans Devanagari',Helvetica,Arial,sans-serif;
  }
  html[data-theme="light"]{
    --bg:#f5f7f9; --panel:#ffffff; --panel-2:#eef1f4; --line:#dde3ea;
    --ink:#182029; --muted:#54616e; --faint:#8b98a5;
    --flag-bg:rgba(242,85,90,0.10);
  }
  *{box-sizing:border-box;}
  body{margin:0; background:var(--bg); color:var(--ink);
       font-family:var(--sans); font-size:14px; line-height:1.5;}
  .wrap{max-width:1280px; margin:0 auto; padding:20px 26px 60px;}
  .topbar{display:flex; align-items:center; justify-content:space-between;
          gap:14px; flex-wrap:wrap;}
  .brand{font-size:20px; font-weight:650; letter-spacing:-0.01em;}
  .brand .dot{color:var(--accent);}
  .controls{display:flex; gap:10px; align-items:center; flex-wrap:wrap;}
  .controls label{color:var(--muted); font-size:12px; margin-right:4px;}
  select,input[type=number]{
    background:var(--panel-2); color:var(--ink); border:1px solid var(--line);
    border-radius:8px; padding:7px 10px; font-size:13px; font-family:var(--sans);}
  input[type=number]{width:76px;}
  .btn{background:var(--panel-2); color:var(--ink); border:1px solid var(--line);
       border-radius:8px; padding:7px 12px; font-size:13px; cursor:pointer;}
  .btn:hover{border-color:var(--accent-dim);}
  .tabs{display:flex; gap:8px; margin:18px 0 4px;}
  .tab{padding:8px 16px; border-radius:10px 10px 0 0; cursor:pointer;
       color:var(--muted); border:1px solid transparent; font-weight:550;}
  .tab.active{color:var(--ink); background:var(--panel);
       border-color:var(--line); border-bottom-color:var(--panel);}
  .meta{color:var(--muted); font-size:13px; margin:8px 0 14px;}
  .kpis{display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin:12px 0 20px;}
  @media(max-width:960px){.kpis{grid-template-columns:repeat(3,1fr);}}
  @media(max-width:580px){.kpis{grid-template-columns:repeat(2,1fr);}}
  .kpi{background:var(--panel); border:1px solid var(--line); border-radius:12px;
       padding:13px 15px;}
  .kpi .l{color:var(--muted); font-size:12px;}
  .kpi .v{font-size:23px; font-weight:650; margin-top:2px;
          font-variant-numeric:tabular-nums;}
  .kpi .s{color:var(--faint); font-size:11px; margin-top:2px;}
  .kpi.alert{border-color:var(--flag); background:var(--flag-bg);}
  .kpi.alert .v,.kpi.money .v{color:var(--flag);}
  .kpi.good .v{color:var(--accent);}
  .grid{display:grid; grid-template-columns:1fr 1fr; gap:16px;}
  @media(max-width:900px){.grid{grid-template-columns:1fr;}}
  .card{background:var(--panel); border:1px solid var(--line); border-radius:12px;
        padding:15px 18px;}
  .card h3{margin:0 0 2px; font-size:14px; font-weight:650;}
  .card .sub{color:var(--muted); font-size:12px; margin-bottom:10px;}
  .span2{grid-column:1 / -1;}
  .chart{width:100%; height:300px;} .chart.sm{height:238px;}
  table{width:100%; border-collapse:collapse; font-size:13px;}
  th,td{padding:8px 10px; text-align:right; border-bottom:1px solid var(--line);
        font-variant-numeric:tabular-nums;}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left;}
  th{color:var(--muted); font-weight:550; cursor:pointer; user-select:none;
     position:sticky; top:0; background:var(--panel); z-index:2;}
  th:hover{color:var(--ink);}
  tr.flagged td{background:var(--flag-bg);}
  .tag{color:var(--flag); font-weight:650;}
  .conf-hi{color:var(--accent);} .conf-mid{color:var(--coast);}
  .flag-item{padding:10px 0; border-bottom:1px solid var(--line);}
  .flag-item:last-child{border-bottom:none;} .flag-item b{color:var(--ink);}
  .money-note{margin-top:12px; padding:12px 14px; background:var(--panel-2);
              border-radius:10px; color:var(--muted); font-size:13px;}
  .money-note b{color:var(--flag);}
  .glossary{color:var(--muted); font-size:12px; margin-top:10px;}
  .glossary b{color:var(--ink); font-weight:600;}
  .foot{color:var(--faint); font-size:12px; margin-top:26px; text-align:center;}
  .viewbtn{padding:4px 10px; font-size:12px;}
  @media print{
    .controls,.tabs,.viewbtn{display:none !important;}
    body{background:#fff;} .wrap{padding:0;}
    .card,.kpi{break-inside:avoid; border-color:#ccc;}
    #tab-overview,#tab-trip{display:block !important;}
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">CPCS<span class="dot">.</span> <span data-i18n="app_title">passenger analytics</span></div>
    <div class="controls">
      <span><label data-i18n="lbl_fare">Fare Rs</label><input type="number" id="fare" value="15" min="1" max="200"/></span>
      <span><label data-i18n="lbl_cap">Capacity</label><input type="number" id="cap" value="45" min="10" max="120"/></span>
      <button class="btn" id="langBtn">हिंदी</button>
      <button class="btn" id="themeBtn">Light</button>
      <button class="btn" id="printBtn" data-i18n="btn_print">Print report</button>
    </div>
  </div>

  <div class="tabs">
    <div class="tab active" id="tabBtnOverview" data-i18n="tab_overview">Fleet overview</div>
    <div class="tab" id="tabBtnTrip" data-i18n="tab_trip">Trip detail</div>
  </div>

  <!-- ======================= FLEET OVERVIEW ======================= -->
  <div id="tab-overview">
    <div class="meta" data-i18n="meta_overview">All logged trips, aggregated. Set fare and capacity above; every figure updates instantly.</div>
    <div class="kpis" id="ovKpis"></div>
    <div class="grid">
      <div class="card">
        <h3 data-i18n="sec_pax_trip">Passengers per trip</h3>
        <div class="sub" data-i18n="sub_pax_trip">Total boardings recorded by the camera on each trip</div>
        <div class="chart sm" id="ovPaxChart"></div>
      </div>
      <div class="card">
        <h3 data-i18n="sec_risk_trip">Revenue at risk per trip</h3>
        <div class="sub" data-i18n="sub_risk_trip">Unticketed boardings x fare, from camera-vs-POS reconciliation</div>
        <div class="chart sm" id="ovRiskChart"></div>
      </div>
      <div class="card span2">
        <h3 data-i18n="sec_trips">Trips</h3>
        <div class="sub" data-i18n="sub_trips">Select View to open the full trip report</div>
        <table id="tripTable"><thead></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <!-- ======================= TRIP DETAIL ======================= -->
  <div id="tab-trip" style="display:none">
    <div class="controls" style="margin-top:6px">
      <span><label data-i18n="lbl_trip">Trip</label><select id="tripSel"></select></span>
    </div>
    <div class="meta" id="meta"></div>
    <div class="kpis" id="kpis"></div>
    <div class="grid">
      <div class="card span2">
        <h3 data-i18n="sec_occ">Occupancy across the route</h3>
        <div class="sub" data-i18n="sub_occ">How many people were on the bus after each stop. Dashed line = capacity. Flag glyph = camera and ticket counts disagreed.</div>
        <div class="chart" id="occChart"></div>
      </div>
      <div class="card">
        <h3 data-i18n="sec_flow">Boardings and alightings</h3>
        <div class="sub" data-i18n="sub_flow">In above the axis, out below</div>
        <div class="chart sm" id="flowChart"></div>
      </div>
      <div class="card">
        <h3 data-i18n="sec_method">Counting method by stop</h3>
        <div class="sub" data-i18n="sub_method">Direct observation vs recovered counts - the audit trail</div>
        <div class="chart sm" id="methodChart"></div>
      </div>
      <div class="card span2">
        <h3 data-i18n="sec_timeline">Crossing events over time</h3>
        <div class="sub" data-i18n="sub_timeline">Every individual count, by video frame. Hover any point for its method.</div>
        <div class="chart sm" id="timeChart"></div>
      </div>
      <div class="card span2">
        <h3 data-i18n="sec_table">Per-stop log</h3>
        <div class="sub" data-i18n="sub_table">Click a column header to sort. Conf = share of counts at this stop from direct observation.</div>
        <div style="max-height:380px; overflow:auto;">
          <table id="stopTable"><thead></thead><tbody></tbody></table>
        </div>
      </div>
      <div class="card span2">
        <h3 data-i18n="sec_recon">Revenue reconciliation</h3>
        <div class="sub" data-i18n="sub_recon">Camera counts vs ticket machine, in plain language</div>
        <div id="flagBody"></div>
        <div class="glossary" data-i18n="glossary">
          <b>live</b> = counted while continuously tracked (direct observation).
          <b>coast</b> = detection dropped near the line; the crossing was completed
          from the person's measured speed and direction (inference, max 0.4s).
          <b>fallback</b> = person entered the camera view on one side of the line
          and left on the other. All non-live counts are retained in the log for audit.
        </div>
      </div>
    </div>
  </div>

  <div class="foot" data-i18n="foot">Processed locally on the edge device - no video leaves the bus - ticket-machine feed simulated for this proof of concept - counts carry a per-event audit trail</div>
</div>

<script>
const DATA = /*__DATA__*/;
let sortKey="seq", sortDir=1, lang="en", currentTrip=null;
const $=id=>document.getElementById(id);
const fmt=n=>n.toLocaleString("en-IN");
const HOW_COLORS={live:"#2dd4a7",coast:"#c9a227",fallback:"#8d76d8",fallback_eof:"#8d76d8"};

const I18N={
 en:{app_title:"passenger analytics",tab_overview:"Fleet overview",tab_trip:"Trip detail",
  lbl_fare:"Fare Rs",lbl_cap:"Capacity",lbl_trip:"Trip",btn_print:"Print report",
  meta_overview:"All logged trips, aggregated. Set fare and capacity above; every figure updates instantly.",
  kpi_trips:"Trips logged",kpi_served:"Passengers served",kpi_alight:"Total alightings",
  kpi_peak:"Peak occupancy",kpi_load:"Peak load factor",kpi_flag:"Stops flagged",
  kpi_risk:"Revenue at risk",kpi_conf:"Data confidence",
  sec_pax_trip:"Passengers per trip",sub_pax_trip:"Total boardings recorded by the camera on each trip",
  sec_risk_trip:"Revenue at risk per trip",sub_risk_trip:"Unticketed boardings x fare, from camera-vs-POS reconciliation",
  sec_trips:"Trips",sub_trips:"Select View to open the full trip report",
  sec_occ:"Occupancy across the route",
  sub_occ:"How many people were on the bus after each stop. Dashed line = capacity. Flag glyph = camera and ticket counts disagreed.",
  sec_flow:"Boardings and alightings",sub_flow:"In above the axis, out below",
  sec_method:"Counting method by stop",sub_method:"Direct observation vs recovered counts - the audit trail",
  sec_timeline:"Crossing events over time",sub_timeline:"Every individual count, by video frame. Hover any point for its method.",
  sec_table:"Per-stop log",sub_table:"Click a column header to sort. Conf = share of counts at this stop from direct observation.",
  sec_recon:"Revenue reconciliation",sub_recon:"Camera counts vs ticket machine, in plain language",
  glossary:"<b>live</b> = counted while continuously tracked (direct observation). <b>coast</b> = detection dropped near the line; the crossing was completed from the person's measured speed and direction (inference, max 0.4s). <b>fallback</b> = person entered the camera view on one side of the line and left on the other. All non-live counts are retained in the log for audit.",
  foot:"Processed locally on the edge device - no video leaves the bus - ticket-machine feed simulated for this proof of concept - counts carry a per-event audit trail",
  agree:"Camera and ticket counts agree at every stop.",
  col:["#","Stop","In","Out","Occ","POS","D","Conf"],
  tripcol:["Trip","Route","Bus","Stops","In","Flags","Risk",""],
  view:"View",inw:"In",outw:"Out",
  flag_line:(s,fare)=>`<b>${s.stop_name}</b> - camera counted <b>${s.boardings}</b> boardings, ticket machine sold <b>${s.pos_count}</b>. ${s.discrepancy} unticketed, Rs ${s.discrepancy*fare} at risk.`,
  flag_under:(s)=>`<b>${s.stop_name}</b> - camera ${s.boardings}, ticket machine ${s.pos_count} (camera under-counted by ${Math.abs(s.discrepancy)}, review).`,
  money:(m)=>`An estimated <b>${m.leak} unticketed passengers</b> this trip represent <b>Rs ${fmt(m.risk)}</b> in potential leakage. Scaled across a 500-bus fleet running multiple trips daily, this is the core ROI case.`},
 hi:{app_title:"यात्री विश्लेषण",tab_overview:"बेड़ा सारांश",tab_trip:"यात्रा विवरण",
  lbl_fare:"किराया Rs",lbl_cap:"क्षमता",lbl_trip:"यात्रा",btn_print:"रिपोर्ट प्रिंट करें",
  meta_overview:"सभी दर्ज यात्राओं का सारांश। ऊपर किराया और क्षमता बदलें; सभी आँकड़े तुरंत बदलते हैं।",
  kpi_trips:"दर्ज यात्राएँ",kpi_served:"कुल चढ़े यात्री",kpi_alight:"कुल उतरे यात्री",
  kpi_peak:"अधिकतम यात्री संख्या",kpi_load:"अधिकतम भार",kpi_flag:"चिह्नित स्टॉप",
  kpi_risk:"संभावित राजस्व हानि",kpi_conf:"डेटा विश्वसनीयता",
  sec_pax_trip:"प्रति यात्रा यात्री",sub_pax_trip:"हर यात्रा में कैमरे द्वारा गिने गए कुल चढ़े यात्री",
  sec_risk_trip:"प्रति यात्रा राजस्व जोखिम",sub_risk_trip:"बिना टिकट चढ़े यात्री x किराया (कैमरा बनाम टिकट मशीन)",
  sec_trips:"यात्राएँ",sub_trips:"पूरी रिपोर्ट खोलने के लिए 'देखें' चुनें",
  sec_occ:"मार्ग पर यात्री संख्या",
  sub_occ:"हर स्टॉप के बाद बस में कितने यात्री थे। बिंदीदार रेखा = क्षमता। ध्वज चिह्न = कैमरा और टिकट गिनती में अंतर।",
  sec_flow:"चढ़ना और उतरना",sub_flow:"अक्ष के ऊपर चढ़े, नीचे उतरे",
  sec_method:"गणना विधि (प्रति स्टॉप)",sub_method:"प्रत्यक्ष गिनती बनाम अनुमानित गिनती - ऑडिट ट्रेल",
  sec_timeline:"समय के साथ आवागमन",sub_timeline:"हर एक गिनती, वीडियो फ्रेम के अनुसार।",
  sec_table:"प्रति-स्टॉप विवरण",sub_table:"क्रमबद्ध करने के लिए शीर्षक पर क्लिक करें। Conf = प्रत्यक्ष गिनती का हिस्सा।",
  sec_recon:"राजस्व मिलान",sub_recon:"कैमरा बनाम टिकट मशीन, सरल भाषा में",
  glossary:"<b>live</b> = लगातार ट्रैक करते हुए प्रत्यक्ष गिनती। <b>coast</b> = रेखा के पास पहचान छूटी; गति और दिशा से गिनती पूरी की गई (अनुमान)। <b>fallback</b> = व्यक्ति रेखा के एक ओर दिखा और दूसरी ओर निकला। सभी अनुमानित गिनतियाँ ऑडिट के लिए दर्ज हैं।",
  foot:"सारी प्रोसेसिंग बस के डिवाइस पर - कोई वीडियो बाहर नहीं जाता - टिकट मशीन डेटा इस PoC में सिम्युलेटेड है - हर गिनती का ऑडिट ट्रेल उपलब्ध",
  agree:"हर स्टॉप पर कैमरा और टिकट गिनती मेल खाती है।",
  col:["#","स्टॉप","चढ़े","उतरे","सवार","POS","D","Conf"],
  tripcol:["यात्रा","मार्ग","बस","स्टॉप","चढ़े","ध्वज","जोखिम",""],
  view:"देखें",inw:"चढ़े",outw:"उतरे",
  flag_line:(s,fare)=>`<b>${s.stop_name}</b> - कैमरे ने <b>${s.boardings}</b> यात्री गिने, टिकट मशीन ने <b>${s.pos_count}</b> टिकट बेचे। ${s.discrepancy} बिना टिकट, Rs ${s.discrepancy*fare} जोखिम में।`,
  flag_under:(s)=>`<b>${s.stop_name}</b> - कैमरा ${s.boardings}, टिकट मशीन ${s.pos_count} (कैमरे ने ${Math.abs(s.discrepancy)} कम गिने, जाँच करें)।`,
  money:(m)=>`इस यात्रा में अनुमानित <b>${m.leak} बिना टिकट यात्री</b> = <b>Rs ${fmt(m.risk)}</b> की संभावित हानि। 500 बसों के बेड़े पर यह मुख्य ROI आधार है।`}
};
const T=()=>I18N[lang];

function applyI18n(){
  document.querySelectorAll("[data-i18n]").forEach(el=>{
    const k=el.dataset.i18n;
    if(T()[k]!==undefined) el.innerHTML=T()[k];
  });
  $("langBtn").textContent = lang==="en" ? "हिंदी" : "English";
}

function confOf(events){
  if(!events.length) return null;
  const live=events.filter(e=>e.how==="live").length;
  return Math.round(100*live/events.length);
}

function metrics(stops, events, fare, cap){
  const inSum=stops.reduce((a,s)=>a+s.boardings,0);
  const outSum=stops.reduce((a,s)=>a+s.alightings,0);
  const peak=stops.reduce((a,s)=>Math.max(a,s.occupancy_after),0);
  const flagged=stops.filter(s=>s.flagged===1).length;
  const leak=stops.filter(s=>s.pos_count>=0)
                  .reduce((a,s)=>a+Math.max(0,s.discrepancy),0);
  return {inSum,outSum,peak,load:cap?Math.round(100*peak/cap):0,
          flagged,leak,risk:leak*fare,conf:confOf(events)};
}

function kpiCell(l,v,cls,s){
  return `<div class="kpi ${cls||""}"><div class="l">${l}</div><div class="v">${v}</div>${s?`<div class="s">${s}</div>`:""}</div>`;
}

function axisC(){
  const cs=getComputedStyle(document.documentElement);
  return {ink:cs.getPropertyValue('--ink').trim(),
          muted:cs.getPropertyValue('--muted').trim(),
          line:cs.getPropertyValue('--line').trim()};
}

let charts={};
function chart(id){
  if(!charts[id]) charts[id]=echarts.init($(id));
  return charts[id];
}
window.addEventListener("resize",()=>Object.values(charts).forEach(c=>c.resize()));

/* ---------------- fleet overview ---------------- */
function drawOverview(){
  const fare=+$("fare").value, cap=+$("cap").value;
  const rows=DATA.trips.map(t=>{
    const b=DATA.by_trip[String(t.trip_id)];
    const m=metrics(b.stops,b.events,fare,cap);
    return {t,m,stops:b.stops.length};
  });
  const agg=rows.reduce((a,r)=>({in:a.in+r.m.inSum,fl:a.fl+r.m.flagged,
    risk:a.risk+r.m.risk,ev:a.ev.concat(DATA.by_trip[String(r.t.trip_id)].events),
    peak:Math.max(a.peak,r.m.peak)}),{in:0,fl:0,risk:0,ev:[],peak:0});
  const conf=confOf(agg.ev);
  $("ovKpis").innerHTML=
    kpiCell(T().kpi_trips,rows.length)+
    kpiCell(T().kpi_served,fmt(agg.in))+
    kpiCell(T().kpi_peak,agg.peak)+
    kpiCell(T().kpi_flag,agg.fl,agg.fl?"alert":"")+
    kpiCell(T().kpi_risk,"Rs "+fmt(agg.risk),"money")+
    kpiCell(T().kpi_conf,conf===null?"-":conf+"%","good");
  const c=axisC();
  const names=rows.map(r=>"#"+r.t.trip_id+" "+r.t.route);
  chart("ovPaxChart").setOption({
    grid:{left:40,right:12,top:14,bottom:40},tooltip:{},
    xAxis:{type:"category",data:names,axisLabel:{color:c.muted}},
    yAxis:{type:"value",axisLabel:{color:c.muted},splitLine:{lineStyle:{color:c.line}}},
    series:[{type:"bar",data:rows.map(r=>r.m.inSum),itemStyle:{color:"#2dd4a7",borderRadius:[4,4,0,0]}}]});
  chart("ovRiskChart").setOption({
    grid:{left:44,right:12,top:14,bottom:40},tooltip:{valueFormatter:v=>"Rs "+fmt(v)},
    xAxis:{type:"category",data:names,axisLabel:{color:c.muted}},
    yAxis:{type:"value",axisLabel:{color:c.muted},splitLine:{lineStyle:{color:c.line}}},
    series:[{type:"bar",data:rows.map(r=>r.m.risk),itemStyle:{color:"#f2555a",borderRadius:[4,4,0,0]}}]});
  const th=$("tripTable").querySelector("thead");
  th.innerHTML="<tr>"+T().tripcol.map(x=>`<th>${x}</th>`).join("")+"</tr>";
  $("tripTable").querySelector("tbody").innerHTML=rows.map(r=>
    `<tr><td>#${r.t.trip_id}</td><td>${r.t.route}</td><td>${r.t.bus_id}</td>
     <td>${r.stops}</td><td>${fmt(r.m.inSum)}</td>
     <td>${r.m.flagged?'<span class="tag">&#9873; '+r.m.flagged+'</span>':'0'}</td>
     <td>Rs ${fmt(r.m.risk)}</td>
     <td><button class="btn viewbtn" onclick="openTrip('${r.t.trip_id}')">${T().view}</button></td></tr>`
  ).join("");
}

/* ---------------- trip detail ---------------- */
function drawTrip(){
  const tid=currentTrip;
  const fare=+$("fare").value, cap=+$("cap").value;
  const b=DATA.by_trip[tid];
  const s=b.stops, t=b.trip, ev=b.events;
  $("meta").textContent=`${t.route} - bus ${t.bus_id} - ${t.started_at}`;
  const m=metrics(s,ev,fare,cap);
  $("kpis").innerHTML=
    kpiCell(T().kpi_served,fmt(m.inSum))+
    kpiCell(T().kpi_alight,fmt(m.outSum))+
    kpiCell(T().kpi_peak,m.peak)+
    kpiCell(T().kpi_load,m.load+"%",m.load>100?"alert":"")+
    kpiCell(T().kpi_flag,m.flagged,m.flagged?"alert":"")+
    kpiCell(T().kpi_risk,"Rs "+fmt(m.risk),"money");
  const c=axisC();
  const names=s.map(x=>x.stop_name);

  // occupancy
  chart("occChart").setOption({
    grid:{left:44,right:16,top:20,bottom:66},tooltip:{trigger:"axis"},
    xAxis:{type:"category",data:names,axisLabel:{rotate:32,color:c.muted}},
    yAxis:{type:"value",axisLabel:{color:c.muted},splitLine:{lineStyle:{color:c.line}}},
    series:[{type:"line",data:s.map(x=>x.occupancy_after),smooth:true,symbolSize:7,
      lineStyle:{width:2.5,color:"#2dd4a7"},itemStyle:{color:"#2dd4a7"},
      areaStyle:{color:"rgba(45,212,167,0.13)"},
      markLine:{silent:true,symbol:"none",lineStyle:{color:c.muted,type:"dashed"},
        data:[{yAxis:cap}],label:{color:c.muted,formatter:"capacity"}},
      markPoint:{symbol:"pin",symbolSize:34,
        itemStyle:{color:"#f2555a"},label:{formatter:"\u2691",color:"#fff",fontSize:13},
        data:s.filter(x=>x.flagged===1).map(x=>({value:"\u2691",coord:[x.stop_name,x.occupancy_after]}))}
    }]});

  // flow
  chart("flowChart").setOption({
    grid:{left:38,right:12,top:30,bottom:66},
    legend:{data:[T().inw,T().outw],top:0,textStyle:{color:c.muted},itemWidth:10,itemHeight:10},
    tooltip:{trigger:"axis",axisPointer:{type:"shadow"},
      formatter:p=>p.map(x=>`${x.seriesName}: ${Math.abs(x.value)}`).join("<br>")},
    xAxis:{type:"category",data:names,axisLabel:{rotate:32,color:c.muted}},
    yAxis:{type:"value",axisLabel:{color:c.muted,formatter:v=>Math.abs(v)},
      splitLine:{lineStyle:{color:c.line}}},
    series:[
      {name:T().inw,type:"bar",stack:"f",data:s.map(x=>x.boardings),itemStyle:{color:"#2dd4a7"}},
      {name:T().outw,type:"bar",stack:"f",data:s.map(x=>-x.alightings),itemStyle:{color:"#e08657"}}]});

  // method stacked per stop
  const hows=["live","coast","fallback","fallback_eof"];
  const perStop={};
  ev.forEach(e=>{
    perStop[e.stop_seq]=perStop[e.stop_seq]||{};
    perStop[e.stop_seq][e.how]=(perStop[e.stop_seq][e.how]||0)+1;});
  chart("methodChart").setOption({
    grid:{left:38,right:12,top:30,bottom:66},
    legend:{top:0,textStyle:{color:c.muted},itemWidth:10,itemHeight:10},
    tooltip:{trigger:"axis",axisPointer:{type:"shadow"}},
    xAxis:{type:"category",data:names,axisLabel:{rotate:32,color:c.muted}},
    yAxis:{type:"value",axisLabel:{color:c.muted},splitLine:{lineStyle:{color:c.line}}},
    series:hows.map(h=>({name:h,type:"bar",stack:"m",
      data:s.map(x=>(perStop[x.seq]&&perStop[x.seq][h])||0),
      itemStyle:{color:HOW_COLORS[h]}}))});

  // timeline scatter
  chart("timeChart").setOption({
    grid:{left:70,right:16,top:16,bottom:44},
    tooltip:{formatter:p=>`frame ${p.value[0]} - ${p.value[1]} (${p.value[2]})`},
    xAxis:{type:"value",name:"frame",nameTextStyle:{color:c.muted},
      axisLabel:{color:c.muted},splitLine:{lineStyle:{color:c.line}}},
    yAxis:{type:"category",data:[T().outw,T().inw],axisLabel:{color:c.muted}},
    series:[{type:"scatter",symbolSize:9,
      data:ev.map(e=>({value:[e.frame,e.direction==="boarding"?T().inw:T().outw,e.how],
        itemStyle:{color:HOW_COLORS[e.how]||"#8b98a5"}}))}]});

  renderTable(s,perStop);
  renderFlags(s,fare,m);
}

function renderTable(s,perStop){
  const cols=[["seq"],["stop_name"],["boardings"],["alightings"],
    ["occupancy_after"],["pos_count"],["discrepancy"],["_conf"]];
  const labels=T().col;
  const th=$("stopTable").querySelector("thead");
  th.innerHTML="<tr>"+cols.map(([k],i)=>{
    const a=k===sortKey?(sortDir>0?" \u25B2":" \u25BC"):"";
    return `<th data-k="${k}">${labels[i]}${a}</th>`;}).join("")+"</tr>";
  th.querySelectorAll("th").forEach(el=>el.onclick=()=>{
    const k=el.dataset.k;
    if(k===sortKey) sortDir*=-1; else {sortKey=k;sortDir=1;}
    renderTable(s,perStop);});
  const withConf=s.map(x=>{
    const p=perStop[x.seq]||{};
    const tot=Object.values(p).reduce((a,v)=>a+v,0);
    return {...x,_conf:tot?Math.round(100*(p.live||0)/tot):null};});
  const rows=[...withConf].sort((a,b)=>{
    const x=a[sortKey],y=b[sortKey];
    return (x>y?1:x<y?-1:0)*sortDir;});
  $("stopTable").querySelector("tbody").innerHTML=rows.map(x=>{
    const pos=x.pos_count<0?"\u2014":x.pos_count;
    const d=x.pos_count<0?"":(x.discrepancy>0?`<span class="tag">+${x.discrepancy}</span>`:x.discrepancy);
    const cf=x._conf===null?"\u2014":`<span class="${x._conf>=90?'conf-hi':'conf-mid'}">${x._conf}%</span>`;
    const flag=x.flagged?'<span class="tag">&#9873;</span> ':'';
    return `<tr class="${x.flagged?'flagged':''}"><td>${x.seq}</td>
      <td>${flag}${x.stop_name}</td><td>${x.boardings}</td><td>${x.alightings}</td>
      <td>${x.occupancy_after}</td><td>${pos}</td><td>${d}</td><td>${cf}</td></tr>`;
  }).join("");
}

function renderFlags(s,fare,m){
  const flagged=s.filter(x=>x.flagged===1);
  const body=$("flagBody");
  if(!flagged.length){
    body.innerHTML=`<div style="color:var(--accent)">${T().agree}</div>`;return;}
  let html=flagged.map(x=>`<div class="flag-item">${
    x.discrepancy>0?T().flag_line(x,fare):T().flag_under(x)}</div>`).join("");
  html+=`<div class="money-note">${T().money(m)}</div>`;
  body.innerHTML=html;
}

/* ---------------- shell ---------------- */
function showTab(which){
  $("tab-overview").style.display=which==="overview"?"block":"none";
  $("tab-trip").style.display=which==="trip"?"block":"none";
  $("tabBtnOverview").classList.toggle("active",which==="overview");
  $("tabBtnTrip").classList.toggle("active",which==="trip");
  redraw();
}
function openTrip(tid){
  currentTrip=String(tid);
  $("tripSel").value=currentTrip;
  showTab("trip");
}
window.openTrip=openTrip;

function redraw(){
  if($("tab-overview").style.display!=="none") drawOverview();
  if($("tab-trip").style.display!=="none") drawTrip();
  setTimeout(()=>Object.values(charts).forEach(c=>c.resize()),0);
}

function boot(){
  const sel=$("tripSel");
  sel.innerHTML=DATA.trips.map(t=>
    `<option value="${t.trip_id}">#${t.trip_id} - ${t.route} - ${t.bus_id}</option>`).join("");
  currentTrip=String(DATA.trips[0].trip_id);
  sel.value=currentTrip;
  sel.onchange=()=>{currentTrip=sel.value;drawTrip();};
  $("fare").oninput=redraw;
  $("cap").oninput=redraw;
  $("tabBtnOverview").onclick=()=>showTab("overview");
  $("tabBtnTrip").onclick=()=>showTab("trip");
  $("langBtn").onclick=()=>{lang=lang==="en"?"hi":"en";applyI18n();redraw();};
  $("themeBtn").onclick=()=>{
    const h=document.documentElement;
    const dark=h.getAttribute("data-theme")==="dark";
    h.setAttribute("data-theme",dark?"light":"dark");
    $("themeBtn").textContent=dark?"Dark":"Light";
    redraw();};
  $("printBtn").onclick=()=>{
    const h=document.documentElement;
    const was=h.getAttribute("data-theme");
    h.setAttribute("data-theme","light");redraw();
    setTimeout(()=>{window.print();
      h.setAttribute("data-theme",was);redraw();},250);};
  applyI18n();
  showTab("overview");
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
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(payload, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out}  ({len(trips)} trip(s))")
    print(f"open it: double-click {args.out}")


if __name__ == "__main__":
    main()
