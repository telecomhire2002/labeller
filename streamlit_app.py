
import io, re, json
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Telecom Hire • Labeller', page_icon='🛰️', layout='wide')

st.markdown(r"""
<style>
[data-testid="stAppViewContainer"] { background: #f6f8fb; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #eef1f5; }
.th-card { background: #ffffff; border: 1px solid #eef1f5; border-radius: 14px; padding: 18px 20px; box-shadow: 0 2px 10px rgba(22,28,45,0.04); }
.th-cta button[kind="primary"] { border-radius: 10px !important; font-weight: 600 !important; }
.th-badge { display:inline-block; padding: 4px 10px; border-radius:999px; background:#eef3ff; color:#3056ff; font-size:12px; border:1px solid #dfe7ff; margin-right:8px; }
.section-title { margin: 4px 0 10px 0; font-weight:700; font-size:18px; }
</style>
<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
  <div class="th-badge">Telecom Hire</div>
  <div>Labeller • sector wiring generator</div>
</div>
""", unsafe_allow_html=True)

st.title('Labeller — Output Generator')
st.caption("Generate sector-wise labelling sheets from your **MAIN DATA CG.xlsx**.")

with st.sidebar:
    st.markdown("### 🛰️ Telecom Hire — Labeller")
    st.markdown("**What it does**")
    st.write("- Creates a labelled output workbook for all sectors of an eNB.")
    st.write("- Uses your MAIN sheet and site ID.")
    st.write("- One click to preview and download.")

CASES = [{"site": "I-MP-ABGR-ENB-H008", "sector": "I-MP-ABGR-ENB-H008-1", "iface_or_ref": "I-MP-ABGR-ENB-H008interface_1", "port": "GigabitEthernet0/0/4", "region": "2.0"}, {"site": "I-MP-ABGR-ENB-H008", "sector": "I-MP-ABGR-ENB-H008-2", "iface_or_ref": "I-MP-ABGR-ENB-H008interface_2", "port": "GigabitEthernet0/0/6", "region": "2.0"}, {"site": "I-MP-ABNP-ENB-6003", "sector": "I-MP-ABNP-ENB-6003-2", "iface_or_ref": "I-MP-ABNP-ENB-6003interface_1", "port": "TenGigabitEthernet0/0/8", "region": "2.0"}, {"site": "I-MP-ABNP-ENB-6003", "sector": "I-MP-ABNP-ENB-6003-3", "iface_or_ref": "I-MP-ABNP-ENB-6003-2", "port": "TenGigabitEthernet0/0/8", "region": "2.0"}, {"site": "I-MP-ABGR-ENB-H004", "sector": "I-MP-ABGR-ENB-H004-1", "iface_or_ref": "I-MP-ABGR-ENB-H004interface_1", "port": "GigabitEthernet0/0/3", "region": "3.0"}, {"site": "I-MP-ABGR-ENB-H004", "sector": "I-MP-ABGR-ENB-H004-2", "iface_or_ref": "I-MP-ABGR-ENB-H004interface_2", "port": "GigabitEthernet0/0/4", "region": "3.0"}, {"site": "I-MP-ABGR-ENB-H004", "sector": "I-MP-ABGR-ENB-H004-3", "iface_or_ref": "I-MP-ABGR-ENB-H004-2", "port": "GigabitEthernet0/0/4", "region": "3.0"}, {"site": "I-MP-ABKP-ENB-G001", "sector": "I-MP-ABKP-ENB-G001-1", "iface_or_ref": "I-MP-ABKP-ENB-G001interface_1", "port": "TenGigabitEthernet0/0/8", "region": "3.0"}, {"site": "I-MP-ABKP-ENB-G001", "sector": "I-MP-ABKP-ENB-G001-2", "iface_or_ref": "I-MP-ABKP-ENB-G001-1", "port": "TenGigabitEthernet0/0/8", "region": "3.0"}, {"site": "I-MP-ABKP-ENB-G001", "sector": "I-MP-ABKP-ENB-G001-3", "iface_or_ref": "I-MP-ABKP-ENB-G001-2", "port": "TenGigabitEthernet0/0/8", "region": "3.0"}, {"site": "I-MP-ABGR-ENB-9008", "sector": "I-MP-ABGR-ENB-9008-1", "iface_or_ref": "I-MP-ABGR-ENB-9008interface_1", "port": "GigabitEthernet0/0/3", "region": "1.0"}]

def norm_tengig(s: str) -> str:
    if not s: return ""
    s = str(s).strip()
    s = s.replace("TenGigabitEthernet", "tenGigEth/")
    s = s.replace("tenGigEth//", "tenGigEth/")
    return s

PORT_RE = re.compile(r"(?:GigabitEthernet\d+/\d+/\d+|TenGigabitEthernet\d+/\d+/\d+|tenGigEth/\d+/\d+/\d+)", re.IGNORECASE)
IFACE_RE = re.compile(r"interface\s*[_-]?(\d)", re.IGNORECASE)
MW_RE    = re.compile(r"MW\s*[_-]?Radio\s*[_-]?Port", re.IGNORECASE)


def parse_vlkp_tag(val: str) -> str:
    """Return 'interface_1', 'interface_2', or 'radio_port' if Port_IP_PMO VLKP ends with one of these.
    Accepts separators like space/_/- and case-insensitive.
    """
    if not val:
        return ''
    s = str(val).strip().lower().replace(' ', '_').replace('-', '_')
    if s.endswith('interface_1'): return 'interface_1'
    if s.endswith('interface_2'): return 'interface_2'
    if s.endswith('mw_radio_port') or s.endswith('radio_port') or s.endswith('radioport'): return 'radio_port'
    return ''

def detect_port_from_cipri(raw: str):
    if not raw: return "", ""
    s = str(raw)
    m = PORT_RE.search(s)
    if m: return norm_tengig(m.group(0)), ""
    m2 = IFACE_RE.search(s)
    if m2: return "", f"interface_{m2.group(1)}"
    if MW_RE.search(s): return "", "MW_Radio_Port"
    return "", ""

def sector_suffix(sector_id: str) -> str:
    m = re.search(r"(\d+)\s*$", str(sector_id)); return m.group(1) if m else str(sector_id)

def find_cases_for_site(enb: str):
    enb = str(enb).strip().lower()
    return [row for row in CASES if str(row["site"]).strip().lower() == enb]

def is_vlkp_site(site_cases):
    for r in site_cases:
        reg = str(r.get('region','')).strip().lower()
        if reg == 'port_ip_pmo vlkp' or 'vlkp' in reg:
            return True
    return False

def case_lookup(site_cases, sector_id: str, iface_hint: str = ""):
    ih = (iface_hint or "").strip().lower()
    s = str(sector_id).strip().lower()
    for r in site_cases:
        if str(r.get("sector","")).strip().lower() == s:
            io = str(r.get("iface_or_ref","")).strip().lower()
            if "interface" in io or "mw_radio_port" in io:
                return {"kind":"main","port": r.get("port","")}
            else:
                return {"kind":"connected","ref_sector": r.get("iface_or_ref",""),"port": r.get("port","")}
    if ih:
        for r in site_cases:
            io = str(r.get("iface_or_ref",""))
            if ih in io.lower():
                return {"kind":"main","port": r.get("port","")}
    return None

def a6_for_sector(df, sector_id):
    r = df[df["GIS SECTOR_ID"].astype(str).str.strip().str.lower() == str(sector_id).strip().lower()]
    if r.empty: return ""
    return str(r.iloc[0].get("A6NEID",""))





def compute_vlkp_decisions(df_site, site_cases):
    """
    Final VLKP decision engine (pattern-only):
      TAGGED sector = Port_IP_PMO VLKP ends with interface_1 / interface_2 / radio_port.
      - Only TAGGED sectors contribute CIPRI ports (from the CIPRI column).
      - UNTAgGED sectors are ALWAYS SF (never MAIN):
          * if first sector overall -> Parent_SF2
          * else -> CONNECTED to previous sector's A6 via _SF2
      - Among TAGGED sectors:
          * 3-sector:
              - all same CIPRI -> lowest-numbered MAIN; other tagged sectors SF to that leader.
              - two share + one unique -> two MAINs (dup leader + unique); remaining tagged sector SF to dup leader.
              - all unique -> CHAIN fallback: first overall -> Parent_SF2;
                               each next TAGGED sector -> PrevA6_{its CIPRI}.
          * 2-sector:
              - same CIPRI -> lowest-numbered MAIN; other tagged SF to leader.
              - distinct CIPRI -> both MAIN.
          * 1-sector:
              - single tagged sector -> MAIN.
      - UNTAgGED sectors get their SF assignment regardless of the tagged grouping above.
    """
    def suffix_num(s):
        m = re.search(r'(\d+)\s*$', str(s))
        return int(m.group(1)) if m else 10**9

    # Sector order
    secs = df_site['GIS SECTOR_ID'].dropna().astype(str).unique().tolist()
    secs = sorted(secs, key=suffix_num)

    # Classify sectors
    tagged = set()
    cipri_port_by_sec = {}
    for _, r in df_site.iterrows():
        sec = str(r.get('GIS SECTOR_ID',''))
        if not sec: continue
        tag = parse_vlkp_tag(str(r.get('Port_IP_PMO VLKP','')))
        if tag:
            tagged.add(sec)
            cipri_raw = str(r.get('A6 CIPRI need to terminate on Parent Node / Beta Port',''))
            cipri_port, _ = detect_port_from_cipri(cipri_raw)
            if cipri_port:
                cipri_port_by_sec[sec] = norm_tengig(cipri_port)

    decisions = {}

    # Helper to set SF for all untagged
    untagged_secs = [s for s in secs if s not in tagged]
    for idx, s in enumerate(secs):
        if s in tagged: 
            continue
        if idx == 0:
            decisions[s] = {'kind': 'parent_sf2'}
        else:
            prev = secs[idx-1]
            decisions[s] = {'kind': 'connected', 'ref_sector': prev}

    # If no tagged sectors, we're done
    if not tagged:
        return decisions

    # Work only with TAGGED for grouped logic
    tagged_order = [s for s in secs if s in tagged]
    # Filter those with parseable CIPRI
    tagged_with_port = [s for s in tagged_order if s in cipri_port_by_sec]

    if len(tagged_order) >= 3:
        # Build groups among tagged_with_port
        groups = {}
        for s in tagged_with_port:
            port = cipri_port_by_sec[s]
            groups.setdefault(port, []).append(s)
        uniq_ports = list(groups.keys())

        if len(uniq_ports) == 1 and sum(len(v) for v in groups.values()) >= 2:
            # All tagged same port -> leader + SF for other tagged
            group = groups[uniq_ports[0]]
            leader = sorted(group, key=suffix_num)[0]
            decisions[leader] = {'kind': 'main', 'port': cipri_port_by_sec[leader]}
            for s in group:
                if s == leader: continue
                decisions[s] = {'kind': 'connected', 'ref_sector': leader}
            return decisions

        if len(uniq_ports) == 2 and any(len(v)==2 for v in groups.values()) and any(len(v)==1 for v in groups.values()):
            items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
            big_port, big_group = items[0]
            small_port, small_group = items[1]
            if len(big_group) == 2 and len(small_group) == 1:
                leader = sorted(big_group, key=suffix_num)[0]
                follower = [s for s in big_group if s != leader][0]
                uniq_sector = small_group[0]
                decisions[leader] = {'kind': 'main', 'port': cipri_port_by_sec[leader]}
                decisions[uniq_sector] = {'kind': 'main', 'port': cipri_port_by_sec[uniq_sector]}
                decisions[follower] = {'kind': 'connected', 'ref_sector': leader}
                return decisions

        # CHAIN fallback across the overall sequence, but only tagged inject CIPRI
        prev = None
        first_set = False
        for s in secs:
            if not first_set:
                # First sector overall is already set above (parent_sf2 if untagged, otherwise leave as-is)
                first_set = True
                prev = s
                continue
            if s in tagged and s in cipri_port_by_sec:
                decisions[s] = {'kind': 'prev_with_cipri', 'prev_sector': prev, 'port': cipri_port_by_sec[s]}
            # advance the anchor regardless of tag
            prev = s
        return decisions

    if len(tagged_order) == 2:
        s1, s2 = tagged_order
        p1 = cipri_port_by_sec.get(s1)
        p2 = cipri_port_by_sec.get(s2)
        if p1 and p2:
            if p1 == p2:
                decisions[s1] = {'kind': 'main', 'port': p1}
                decisions[s2] = {'kind': 'connected', 'ref_sector': s1}
            else:
                decisions[s1] = {'kind': 'main', 'port': p1}
                decisions[s2] = {'kind': 'main', 'port': p2}
        elif p1 and not p2:
            decisions[s1] = {'kind': 'main', 'port': p1}
        elif p2 and not p1:
            # Ensure s2 gets MAIN if only it has a port
            decisions[s2] = {'kind': 'main', 'port': p2}
        # If neither has port, they remain SF from the untagged pass (connected/parent_sf2)

        return decisions

    if len(tagged_order) == 1:
        s = tagged_order[0]
        p = cipri_port_by_sec.get(s)
        if p:
            decisions[s] = {'kind': 'main', 'port': p}
        # else, it remains SF by the untagged assignment
        return decisions

    return decisions
  # no tagged sectors; baked/default apply

    if len(secs) >= 3:
        # Group tagged sectors by CIPRI
        groups = {}
        for s in with_port:
            groups.setdefault(per[s]['cipri_port'], []).append(s)
        uniq_ports = list(groups.keys())

        # All same among tagged
        if len(uniq_ports) == 1 and sum(len(v) for v in groups.values()) >= 2:
            group = groups[uniq_ports[0]]
            leader = sorted(group, key=suffix_num)[0]
            decisions[leader] = {'kind': 'main', 'port': per[leader]['cipri_port']}
            for s in group:
                if s == leader: continue
                decisions[s] = {'kind': 'connected', 'ref_sector': leader}
            # Note: untagged sectors remain undecided here

        # Two share + one unique among tagged
        elif len(uniq_ports) == 2 and any(len(v)==2 for v in groups.values()) and any(len(v)==1 for v in groups.values()):
            items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
            big_port, big_group = items[0]
            small_port, small_group = items[1]
            if len(big_group) == 2 and len(small_group) == 1:
                leader = sorted(big_group, key=suffix_num)[0]
                follower = [s for s in big_group if s != leader][0]
                uniq_sector = small_group[0]
                decisions[leader] = {'kind': 'main', 'port': per[leader]['cipri_port']}
                decisions[uniq_sector] = {'kind': 'main', 'port': per[uniq_sector]['cipri_port']}
                decisions[follower] = {'kind': 'connected', 'ref_sector': leader}

        # All unique among tagged -> chain fallback
        else:
            if len(with_port) >= 2:
                # First sector overall -> Parent _SF2 (even if untagged)
                prev = None
                first_assigned = False
                for s in secs:
                    if not first_assigned:
                        decisions[s] = {'kind': 'parent_sf2'}
                        first_assigned = True
                        prev = s
                        continue
                    if s in per:
                        decisions[s] = {'kind': 'prev_with_cipri', 'prev_sector': prev, 'port': per[s]['cipri_port']}
                    # untagged sectors don't get a preset, but still advance the chain anchor
                    prev = s
                return decisions

        return decisions
  # let baked/default handle

    if len(secs) >= 3:
        # Group by CIPRI port
        groups = {}
        for s in with_port:
            groups.setdefault(per[s]['cipri_port'], []).append(s)
        uniq_ports = list(groups.keys())

        # All three same CIPRI
        if len(uniq_ports) == 1 and sum(len(v) for v in groups.values()) >= 3:
            group = groups[uniq_ports[0]]
            leader = sorted(group, key=suffix_num)[0]
            decisions[leader] = {'kind': 'main', 'port': per[leader]['cipri_port']}
            for s in group:
                if s == leader: continue
                decisions[s] = {'kind': 'connected', 'ref_sector': leader}
            return decisions

        # Exactly two share + one unique
        if len(uniq_ports) == 2:
            items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
            big_port, big_group = items[0]
            small_port, small_group = items[1]
            if len(big_group) == 2 and len(small_group) == 1:
                leader = sorted(big_group, key=suffix_num)[0]
                follower = [s for s in big_group if s != leader][0]
                uniq_sector = small_group[0]
                decisions[leader] = {'kind': 'main', 'port': per[leader]['cipri_port']}
                decisions[uniq_sector] = {'kind': 'main', 'port': per[uniq_sector]['cipri_port']}
                decisions[follower] = {'kind': 'connected', 'ref_sector': leader}
                return decisions

        # All unique CIPRI -> chain fallback if at least two have CIPRI
        if len(with_port) >= 2:
            prev = None
            for s in secs:
                if s not in per:
                    prev = s
                    continue
                if prev is None:
                    decisions[s] = {'kind': 'parent_sf2'}
                else:
                    decisions[s] = {'kind': 'prev_with_cipri', 'prev_sector': prev, 'port': per[s]['cipri_port']}
                prev = s
            return decisions

        return decisions

    if len(secs) == 2:
        if len(with_port) == 2:
            p1 = per[secs[0]]['cipri_port']
            p2 = per[secs[1]]['cipri_port']
            if p1 == p2:
                leader = secs[0]  # lowest-numbered
                follower = secs[1]
                decisions[leader] = {'kind': 'main', 'port': p1}
                decisions[follower] = {'kind': 'connected', 'ref_sector': leader}
                return decisions
            else:
                decisions[secs[0]] = {'kind': 'main', 'port': p1}
                decisions[secs[1]] = {'kind': 'main', 'port': p2}
                return decisions
        elif len(with_port) == 1:
            s = with_port[0]
            decisions[s] = {'kind': 'main', 'port': per[s]['cipri_port']}
            return decisions
        return decisions

    if len(secs) == 1:
        s = secs[0]
        decisions[s] = {'kind': 'main', 'port': per[s]['cipri_port']} if s in per else {}
        return decisions

    return decisions



    if len(secs) >= 3:
        groups = {}
        for s in with_port:
            groups.setdefault(per[s]['cipri_port'], []).append(s)
        uniq_ports = list(groups.keys())

        if len(uniq_ports) == 1 and len(with_port) >= 3 and sum(len(v) for v in groups.values()) >= 3:
            group = groups[uniq_ports[0]]
            leader = sorted(group, key=suffix_num)[0]
            decisions[leader] = {'kind': 'main', 'port': per[leader]['cipri_port']}
            for s in group:
                if s == leader: continue
                decisions[s] = {'kind': 'connected', 'ref_sector': leader}
            return decisions

        if len(uniq_ports) == 2:
            items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
            big_port, big_group = items[0]
            small_port, small_group = items[1]
            if len(big_group) == 2 and len(small_group) == 1:
                leader = sorted(big_group, key=suffix_num)[0]
                follower = [s for s in big_group if s != leader][0]
                uniq_sector = small_group[0]
                decisions[leader] = {'kind': 'main', 'port': per[leader]['cipri_port']}
                decisions[uniq_sector] = {'kind': 'main', 'port': per[uniq_sector]['cipri_port']}
                decisions[follower] = {'kind': 'connected', 'ref_sector': leader}
                return decisions

        if len(with_port) >= 2:
            prev = None
            for s in secs:
                if s not in per:
                    prev = s
                    continue
                if prev is None:
                    decisions[s] = {'kind': 'parent_sf2'}
                else:
                    decisions[s] = {'kind': 'prev_with_cipri', 'prev_sector': prev, 'port': per[s]['cipri_port']}
                prev = s
            return decisions

        return decisions

    if len(secs) == 2:
        if len(with_port) == 2:
            p1 = per[secs[0]]['cipri_port'] if secs[0] in per else None
            p2 = per[secs[1]]['cipri_port'] if secs[1] in per else None
            if p1 and p2 and p1 == p2:
                leader = secs[0]
                follower = secs[1]
                decisions[leader] = {'kind': 'main', 'port': p1}
                decisions[follower] = {'kind': 'connected', 'ref_sector': leader}
                return decisions
            else:
                decisions[secs[0]] = {'kind': 'main', 'port': p1}
                decisions[secs[1]] = {'kind': 'main', 'port': p2}
                return decisions
        elif len(with_port) == 1:
            s = with_port[0]
            decisions[s] = {'kind': 'main', 'port': per[s]['cipri_port']}
            return decisions
        return decisions

    if len(secs) == 1:
        s = secs[0]
        if s in per:
            decisions[s] = {'kind': 'main', 'port': per[s]['cipri_port']}
        return decisions

    return decisions

def build_sector_block(row_main, sector_id, site_cases, df_all, preset=None):
    enb    = str(row_main.get("eNBsiteID","")).strip()
    a6     = str(row_main.get("A6NEID","")).strip()
    parent = str(row_main.get("Parenting NEID","")).strip()

    cipri_raw = str(row_main.get("A6 CIPRI need to terminate on Parent Node / Beta Port",""))
    cipri_port, iface_hint = detect_port_from_cipri(cipri_raw)

    ck = case_lookup(site_cases, sector_id, iface_hint)

    if isinstance(preset, dict) and sector_id in preset:
        p = preset[sector_id]
        if p.get('kind') == 'parent_sf2':
            left_F, right_F = (f"F:{a6}_SF1" if a6 else ""), (f"F:{parent}_SF2" if parent else "")
            left_T, right_T = (f"T:{parent}_SF2" if parent else ""), (f"T:{a6}_SF1" if a6 else "")
        elif p.get('kind') == 'prev_with_cipri':
            ref_sec = p.get('prev_sector','')
            ref_a6  = a6_for_sector(df_all, ref_sec)
            port    = norm_tengig(p.get('port',''))
            left_F, right_F = (f"F:{a6}_SF1" if a6 else ""), (f"F:{ref_a6}_{port}" if (ref_a6 and port) else "")
            left_T, right_T = (f"T:{ref_a6}_{port}" if (ref_a6 and port) else ""), (f"T:{a6}_SF1" if a6 else "")
        elif p.get('kind') == 'connected':
            ref_sec = p.get('ref_sector','')
            ref_a6  = a6_for_sector(df_all, ref_sec)
            left_F, right_F = (f"F:{a6}_SF1" if a6 else ""), (f"F:{ref_a6}_SF2" if ref_a6 else "")
            left_T, right_T = (f"T:{ref_a6}_SF2" if ref_a6 else ""), (f"T:{a6}_SF1" if a6 else "")
        else:  # 'main'
            port = norm_tengig(p.get('port','')) or norm_tengig(cipri_port) or norm_tengig(cipri_raw)
            left_F, right_F = (f"F:{a6}_SF1" if a6 else ""), (f"F:{parent}_{port}" if (parent and port) else "")
            left_T, right_T = (f"T:{parent}_{port}" if (parent and port) else ""), (f"T:{a6}_SF1" if a6 else "")
    elif ck and ck.get("kind") == "connected":
        ref_sec = ck.get("ref_sector", "")
        ref_a6  = a6_for_sector(df_all, ref_sec)
        left_F, right_F = (f"F:{a6}_SF1" if a6 else ""), (f"F:{ref_a6}_SF2" if ref_a6 else "")
        left_T, right_T = (f"T:{ref_a6}_SF2" if ref_a6 else ""), (f"T:{a6}_SF1" if a6 else "")
    elif ck and ck.get("kind") == "main":
        port = norm_tengig(ck.get("port","")) or norm_tengig(cipri_port) or norm_tengig(cipri_raw)
        left_F, right_F = (f"F:{a6}_SF1" if a6 else ""), (f"F:{parent}_{port}" if (parent and port) else "")
        left_T, right_T = (f"T:{parent}_{port}" if (parent and port) else ""), (f"T:{a6}_SF1" if a6 else "")
    else:
        enb_val = str(row_main.get("eNBsiteID",""))
        sib = df_all[df_all["eNBsiteID"].astype(str).str.strip().str.lower()==enb_val.strip().lower()]["GIS SECTOR_ID"].dropna().astype(str).tolist()
        def _suf(s): m = re.search(r"(\d+)\s*$", str(s)); return int(m.group(1)) if m else None
        sib_sorted = sorted(sib, key=lambda s: (_suf(s) is None, _suf(s) if _suf(s) is not None else 99999))
        if sector_id == sib_sorted[0]:
            port = norm_tengig(cipri_port) or norm_tengig(cipri_raw)
            left_F, right_F = (f"F:{a6}_SF1" if a6 else ""), (f"F:{parent}_{port}" if (parent and port) else "")
            left_T, right_T = (f"T:{parent}_{port}" if (parent and port) else ""), (f"T:{a6}_SF1" if a6 else "")
        else:
            prev_index = max(0, sib_sorted.index(sector_id)-1)
            ref_sec = sib_sorted[prev_index]
            ref_a6  = a6_for_sector(df_all, ref_sec)
            left_F, right_F = (f"F:{a6}_SF1" if a6 else ""), (f"F:{ref_a6}_SF2" if ref_a6 else "")
            left_T, right_T = (f"T:{ref_a6}_SF2" if ref_a6 else ""), (f"T:{a6}_SF1" if a6 else "")

    suf = sector_suffix(sector_id)
    rows = []
    rows.append(["eNB ID","Sector ID","A6 NE ID"])
    rows.append([enb, sector_id, a6])
    rows.append([f"SECTOR--{suf}", f"AP End--{suf}", "Switch End"])
    rows.append(["Optical fiber Cabeling", left_F, right_F])
    rows.append(["AP1", left_T, right_T])
    rows.append(["", "", ""])
    rows.append(["Grounding A6", f"G:{a6}", f"G:{a6}"])
    rows.append(["Grounding Rack", f"G:{a6}", f"G:{a6}"])
    rows.append(["", "", ""])
    rows.append(["Power cable labeling", f"F:{a6}", f"F: SMPS_DC_LOAD {suf}"])
    rows.append(["Power cable labeling", f"T: SMPS_DC_LOAD {suf}", f"T:{a6}"])
    rows.append(["", "", ""])
    rows.append(["", "", ""])
    return rows

# ---- UI inputs ----
c1, c2 = st.columns([1.3, 1])
with c1:
    st.markdown('<div class="section-title">1) Upload MAIN</div>', unsafe_allow_html=True)
    st.markdown('<div class="th-card">Attach your MAIN DATA CG workbook.</div>', unsafe_allow_html=True)
    main_file = st.file_uploader("MAIN DATA CG.xlsx", type=["xlsx"])
with c2:
    st.markdown('<div class="section-title">2) Enter eNB ID</div>', unsafe_allow_html=True)
    st.markdown('<div class="th-card">Example: <code>I-MP-RPUR-ENB-A005</code></div>', unsafe_allow_html=True)
    enb_id = st.text_input("eNB ID", value="", placeholder="I-MP-XXXX-ENB-YYYY")

st.markdown('<div class="section-title">3) Generate</div>', unsafe_allow_html=True)
st.markdown('<div class="th-card">Click **Generate** to preview and download your output.</div>', unsafe_allow_html=True)
go = st.button("Generate", type="primary")

@st.cache_data(show_spinner=False)
def _read_main(uploaded):
    return pd.ExcelFile(uploaded, engine='openpyxl').parse(0)

if go:
    if not main_file or not enb_id.strip():
        st.error("Please upload MAIN and enter a valid eNB ID.")
    else:
        try:
            df = _read_main(main_file)
            site_rows = df[df["eNBsiteID"].astype(str).str.strip().str.lower() == enb_id.strip().lower()].copy()
            if site_rows.empty:
                site_rows = df[df["eNBsiteID"].astype(str).str.contains(enb_id, case=False, na=False)].copy()
            if site_rows.empty:
                st.error("No rows found for that eNB ID in MAIN.")
            else:
                sectors = site_rows["GIS SECTOR_ID"].dropna().astype(str).unique().tolist()
                def keyf(s):
                    m = re.search(r"(\d+)\s*$", s)
                    return int(m.group(1)) if m else 99999
                sectors = sorted(sectors, key=keyf)

                site_cases = find_cases_for_site(enb_id)
                # apply VLKP decisions only when marked in baked cases
                vlkp_preset = compute_vlkp_decisions(site_rows, site_cases)

                out_rows = []
                for sec in sectors:
                    r = site_rows[site_rows["GIS SECTOR_ID"].astype(str)==sec]
                    use_row = r.iloc[0] if not r.empty else site_rows.iloc[0]
                    block = build_sector_block(use_row, sec, site_cases, df, preset=vlkp_preset)
                    out_rows.extend(block)

                out_df = pd.DataFrame(out_rows)
                st.success('✅ Generated — preview below')
                st.dataframe(out_df, use_container_width=True, height=420)

                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                    out_df.to_excel(writer, index=False, header=False, sheet_name="Sheet1")
                bio.seek(0)
                st.markdown('<div class="th-cta">', unsafe_allow_html=True)
                st.download_button('⬇️ Download output.xlsx', data=bio.getvalue(),
                                   file_name='output.xlsx',
                                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                   type='primary')
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error("Failed to generate output.")
            st.exception(e)
else:
    st.info("Ready when you are — upload MAIN, enter eNB ID, then click **Generate**.")
