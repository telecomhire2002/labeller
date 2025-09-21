
import io
import re
import zipfile
import pandas as pd
import streamlit as st

st.set_page_config(page_title='A6 • Labeller', page_icon='🛰️', layout='wide')

# --------------------------- helpers ---------------------------
def norm_tengig(s: str) -> str:
    if not s: return ""
    s = str(s).strip()
    s = s.replace("TenGigabitEthernet", "tenGigEth/")
    s = s.replace("tenGigEth//", "tenGigEth/")
    return s

PORT_RE = re.compile(r"(?:GigabitEthernet\d+/\d+/\d+|TenGigabitEthernet\d+/\d+/\d+|tenGigEth/\d+/\d+/\d+)", re.IGNORECASE)
IFACE_RE = re.compile(r"interface\s*[_-]?(\d)", re.IGNORECASE)
MW_RE    = re.compile(r"(?:mw\s*[_-]?)?radio\s*[_-]?port", re.IGNORECASE)

def detect_port_from_cipri(raw: str):
    if raw is None: return "", ""
    s = str(raw).strip()
    # Treat TN3 as a valid "port token"
    if s.upper() == "TN3":
        return "TN3", ""
    m = PORT_RE.search(s)
    if m: return norm_tengig(m.group(0)), ""
    m2 = IFACE_RE.search(s)
    if m2: return "", f"interface_{m2.group(1)}"
    if MW_RE.search(s): return "", "radio_port"
    return "", ""

def parse_vlkp_tag(val: str) -> str:
    if not val: return ""
    s = str(val).strip().lower().replace(" ", "_").replace("-", "_")
    if s.endswith("interface_1"): return "interface_1"
    if s.endswith("interface_2"): return "interface_2"
    if s.endswith("mw_radio_port") or s.endswith("radio_port") or s.endswith("radioport"): return "radio_port"
    return ""

def suffix_num(s):
    m = re.search(r"(\d+)\s*$", str(s))
    return int(m.group(1)) if m else 10**9

def _clean(x):
    if x is None: return ""
    s = str(x).strip()
    if s == "" or s.lower() in {"nan","nat","none","null"}:
        return ""
    return s

def a6_for_sector(df_all, sector_id):
    r = df_all[df_all["GIS SECTOR_ID"].astype(str).str.strip().str.lower() == str(sector_id).strip().lower()]
    if r.empty: return ("","")
    return _clean(r.iloc[0].get("A6NEID","")), _clean(r.iloc[0].get("A6 IP",""))

def preprocess_main(df):
    """
    For each (eNBsiteID, A6NEID):
      - If some rows have CIPRI and others blank -> fill blanks with that CIPRI.
      - If all blank -> set CIPRI to 'TN3'.
    """
    df = df.copy()
    cipri_col = "A6 CIPRI need to terminate on Parent Node / Beta Port"
    if cipri_col not in df.columns:
        return df
    # Normalize blanks
    df[cipri_col] = df[cipri_col].apply(lambda x: "" if _clean(x)=="" else x)
    key_cols = [c for c in ["eNBsiteID", "A6NEID"] if c in df.columns]
    if not key_cols:
        return df
    for keys, g in df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        m = pd.Series(True, index=df.index)
        for col, val in zip(key_cols, keys):
            m &= (df[col] == val)
        block = df.loc[m, cipri_col]
        nonblank = block[block.astype(str).str.strip() != ""]
        if len(nonblank) > 0:
            fill_value = nonblank.iloc[0]
            df.loc[m & (df[cipri_col].astype(str).str.strip() == ""), cipri_col] = fill_value
        else:
            df.loc[m, cipri_col] = "TN3"
    return df

# --------------------------- core logic ---------------------------
def compute_vlkp_decisions(df_site):
    """
    Pattern-only VLKP logic:
      - TAGGED sector = Port_IP_PMO VLKP ends with interface_1 / interface_2 / radio_port.
      - Only TAGGED sectors contribute CIPRI ports (from the CIPRI column).
      - UNTAgGED sectors are ALWAYS SF (never MAIN):
          * if first sector overall -> Parent_SFP2
          * else -> CONNECTED to previous sector's A6 via _SFP2
      - Among TAGGED sectors:
          * 3-sector:
              - all same CIPRI -> lowest-numbered MAIN; other tagged SF to that leader.
              - two share + one unique -> two MAINs (dup leader + unique); remaining tagged SF to dup leader.
              - all unique -> CHAIN fallback: first overall already set; each next TAGGED sector -> PrevA6_{its CIPRI}.
          * 2-sector:
              - same CIPRI -> lowest-numbered MAIN; other tagged SF to leader.
              - distinct CIPRI -> both MAIN.
          * 1-sector:
              - single tagged sector -> MAIN.
      - RADIO_PORT OVERRIDE: any sector with radio_port tag is MAIN (even without concrete CIPRI port).
    """
    secs = df_site['GIS SECTOR_ID'].dropna().astype(str).unique().tolist()
    secs = sorted(secs, key=suffix_num)

    # classify
    tagged = set()
    cipri_port_by_sec = {}
    for _, r in df_site.iterrows():
        sec = str(r.get('GIS SECTOR_ID',''))
        if not sec: continue
        tag = parse_vlkp_tag(_clean(r.get('Port_IP_PMO VLKP','')))
        if tag:
            tagged.add(sec)
            cipri_raw = _clean(r.get('A6 CIPRI need to terminate on Parent Node / Beta Port',''))
            cipri_port, _ = detect_port_from_cipri(cipri_raw)
            if cipri_port:
                cipri_port_by_sec[sec] = norm_tengig(cipri_port)

    # Promote all radio_port-tagged sectors to MAIN (regardless of CIPRI port presence)
    radio_mains = []
    for _, r in df_site.iterrows():
        sec2 = str(r.get('GIS SECTOR_ID',''))
        if not sec2: 
            continue
        if parse_vlkp_tag(_clean(r.get('Port_IP_PMO VLKP',''))) == 'radio_port':
            radio_mains.append(sec2)

    decisions = {}

    # Mark radio_port as MAIN upfront
    for _sec in radio_mains:
        decisions[_sec] = {'kind': 'main'}

    # assign SF to all untagged first (unless already MAIN by radio rule)
    for idx, s in enumerate(secs):
        if s in tagged and s not in decisions:
            continue
        if s in decisions:  # already main via radio rule
            continue
        if idx == 0:
            decisions[s] = {'kind': 'parent_SFP2'}
        else:
            prev = secs[idx-1]
            decisions[s] = {'kind': 'connected', 'ref_sector': prev}

    if not tagged and not radio_mains:
        return decisions

    # tagged order and those with ports
    tagged_order = [s for s in secs if s in tagged]
    tagged_with_port = [s for s in tagged_order if s in cipri_port_by_sec]

    if len(tagged_order) >= 3:
        # group among tagged_with_port
        groups = {}
        for s in tagged_with_port:
            port = cipri_port_by_sec[s]
            groups.setdefault(port, []).append(s)
        uniq_ports = list(groups.keys())

        if len(uniq_ports) == 1 and sum(len(v) for v in groups.values()) >= 2:
            group = groups[uniq_ports[0]]
            leader = sorted(group, key=suffix_num)[0]
            if leader not in decisions:  # don't overwrite radio main
                decisions[leader] = {'kind': 'main', 'port': cipri_port_by_sec[leader]}
            for s in group:
                if s == leader: continue
                if s not in decisions:
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
                if leader not in decisions:
                    decisions[leader] = {'kind': 'main', 'port': cipri_port_by_sec[leader]}
                if uniq_sector not in decisions:
                    decisions[uniq_sector] = {'kind': 'main', 'port': cipri_port_by_sec[uniq_sector]}
                if follower not in decisions:
                    decisions[follower] = {'kind': 'connected', 'ref_sector': leader}
                return decisions

        # chain fallback for tagged with ports (first overall already decided above if untagged)
        prev = None
        first_set = False
        for s in secs:
            if not first_set:
                first_set = True
                prev = s
                continue
            if s in tagged and s in cipri_port_by_sec and s not in decisions:
                decisions[s] = {'kind': 'prev_with_cipri', 'prev_sector': prev, 'port': cipri_port_by_sec[s]}
            prev = s
        return decisions

    if len(tagged_order) == 2:
        s1, s2 = tagged_order
        p1 = cipri_port_by_sec.get(s1)
        p2 = cipri_port_by_sec.get(s2)
        if p1 and p2:
            if p1 == p2:
                if s1 not in decisions:
                    decisions[s1] = {'kind': 'main', 'port': p1}
                if s2 not in decisions:
                    decisions[s2] = {'kind': 'connected', 'ref_sector': s1}
            else:
                if s1 not in decisions:
                    decisions[s1] = {'kind': 'main', 'port': p1}
                if s2 not in decisions:
                    decisions[s2] = {'kind': 'main', 'port': p2}
        elif p1 and not p2:
            if s1 not in decisions:
                decisions[s1] = {'kind': 'main', 'port': p1}
        elif p2 and not p1:
            if s2 not in decisions:
                decisions[s2] = {'kind': 'main', 'port': p2}
        return decisions

    if len(tagged_order) == 1:
        s = tagged_order[0]
        p = cipri_port_by_sec.get(s)
        if p and s not in decisions:
            decisions[s] = {'kind': 'main', 'port': p}
        return decisions

    return decisions

def build_sector_block(row_main, sector_id, df_all, preset):
    # Clean fields
    enb    = _clean(row_main.get("eNBsiteID",""))
    a6     = _clean(row_main.get("A6NEID",""))
    a6_ip  = _clean(row_main.get("A6 IP",""))
    parent = _clean(row_main.get("Parenting NEID",""))
    raw_vlkp = _clean(row_main.get("Port_IP_PMO VLKP",""))

    # Parse CIPRI and VLKP
    cipri_raw = _clean(row_main.get("A6 CIPRI need to terminate on Parent Node / Beta Port",""))
    cipri_port, _ = detect_port_from_cipri(cipri_raw)
    cipri_port = norm_tengig(cipri_port)
    vlkp_tag = parse_vlkp_tag(raw_vlkp)

    # Radio-port-aware suffixes (SFP1/SFP2 for any site that has a radio_port tag)
    def _site_has_radio(enb_id):
        sub = df_all[df_all["eNBsiteID"].astype(str).str.strip().str.lower() == enb_id.strip().lower()]
        if sub.empty: 
            return False
        return any(parse_vlkp_tag(_clean(v)) == "radio_port" for v in sub.get("Port_IP_PMO VLKP", []))
    radio_mode = _site_has_radio(enb)
    suf1 = "SFP1" if radio_mode else "_SFP1"
    suf2 = "SFP2" if radio_mode else "_SFP2"

    # preset decision from engine
    p = preset.get(sector_id, {}) if isinstance(preset, dict) else {}

    # --- Branches ---
    if p.get('kind') == 'parent_SFP2':
        # No parent & token (e.g., TN3): use A6IP_token both for fiber switch end and AP1 left
        if not parent and cipri_port:
            left_F  = (f"F:{a6}_{suf1}" if a6 else "")
            right_F = (f"F:{a6_ip}_{cipri_port}" if a6_ip else f"F:{cipri_port}")
            left_T  = (f"T:{a6_ip}_{cipri_port}" if a6_ip else f"T:{cipri_port}")
            right_T = (f"T:{a6}_{suf1}" if a6 else "")
        else:
            left_F, right_F = (f"F:{a6}_{suf1}" if a6 else ""), (f"F:{parent}_{suf2}" if parent else "")
            left_T, right_T = (f"T:{parent}_{suf2}" if parent else ""), (f"T:{a6}_{suf1}" if a6 else "")

    elif p.get('kind') == 'prev_with_cipri':
        ref_sec = p.get('prev_sector','')
        ref_a6, _  = a6_for_sector(df_all, ref_sec)
        port    = norm_tengig(p.get('port',''))
        # Fiber
        left_F  = (f"F:{a6}_{suf1}" if a6 else "")
        right_F = (f"F:{ref_a6}_{suf2}" if (ref_a6 and not port) else (f"F:{ref_a6}_{port}" if (ref_a6 and port) else ""))
        # AP1
        if port:
            left_T = f"T:{a6_ip}_{port}" if a6_ip else (f"T:{ref_a6}_{port}" if ref_a6 else "")
        else:
            left_T = f"T:{ref_a6}_{suf2}" if ref_a6 else ""
        right_T = (f"T:{a6}_{suf1}" if a6 else "")

    elif p.get('kind') == 'connected':
        ref_sec = p.get('ref_sector','')
        ref_a6, _  = a6_for_sector(df_all, ref_sec)
        left_F, right_F = (f"F:{a6}_{suf1}" if a6 else ""), (f"F:{ref_a6}_{suf2}" if ref_a6 else "")
        left_T, right_T = (f"T:{ref_a6}_{suf2}" if ref_a6 else ""), (f"T:{a6}_{suf1}" if a6 else "")

    elif p.get('kind') == 'main':
        # RADIO_PORT: use parent + raw_vlkp token for switch end; SFP suffixes
        if vlkp_tag == "radio_port" and parent:
            left_F, right_F = (f"F:{a6}_{suf1}" if a6 else ""), f"F:{parent}_{raw_vlkp}"
            left_T, right_T = f"T:{parent}_{raw_vlkp}", (f"T:{a6}_{suf1}" if a6 else "")
        else:
            port = norm_tengig(p.get('port','')) or cipri_port
            if parent and port:
                left_F, right_F = (f"F:{a6}_{suf1}" if a6 else ""), f"F:{parent}_{port}"
                left_T, right_T = f"T:{parent}_{port}", (f"T:{a6}_{suf1}" if a6 else "")
            elif port:
                # No parent but have token (e.g., TN3): A6IP_token fallback
                left_F  = (f"F:{a6}_{suf1}" if a6 else "")
                right_F = (f"F:{a6_ip}_{port}" if a6_ip else f"F:{port}")
                left_T  = (f"T:{a6_ip}_{port}" if a6_ip else f"T:{port}")
                right_T = (f"T:{a6}_{suf1}" if a6 else "")
            else:
                left_F, right_F = (f"F:{a6}_{suf1}" if a6 else ""), ""
                left_T, right_T = "", (f"T:{a6}_{suf1}" if a6 else "")

    else:
        # Safety fallback
        all_secs = df_all[df_all["eNBsiteID"].astype(str).str.strip().str.lower() == enb.lower()]["GIS SECTOR_ID"].dropna().astype(str).tolist()
        all_secs = sorted(all_secs, key=suffix_num)
        if sector_id == all_secs[0]:
            if not parent and cipri_port:
                left_F  = (f"F:{a6}_{suf1}" if a6 else "")
                right_F = (f"F:{a6_ip}_{cipri_port}" if a6_ip else f"F:{cipri_port}")
                left_T  = (f"T:{a6_ip}_{cipri_port}" if a6_ip else f"T:{cipri_port}")
                right_T = (f"T:{a6}_{suf1}" if a6 else "")
            else:
                left_F, right_F = (f"F:{a6}_{suf1}" if a6 else ""), (f"F:{parent}_{suf2}" if parent else "")
                left_T, right_T = (f"T:{parent}_{suf2}" if parent else ""), (f"T:{a6}_{suf1}" if a6 else "")
        else:
            prev = all_secs[all_secs.index(sector_id)-1]
            ref_a6, _ = a6_for_sector(df_all, prev)
            left_F, right_F = (f"F:{a6}_{suf1}" if a6 else ""), (f"F:{ref_a6}_{suf2}" if ref_a6 else "")
            left_T, right_T = (f"T:{ref_a6}_{suf2}" if ref_a6 else ""), (f"T:{a6}_{suf1}" if a6 else "")

    # Assemble rows
    sufnum = int(suffix_num(sector_id))
    rows = []
    rows.append(["eNB ID","Sector ID","A6 NE ID"])
    rows.append([enb, sector_id, a6])
    rows.append([f"SECTOR--{sufnum}", f"AP End--{sufnum}", "Switch End"])
    rows.append(["Optical fiber Cabeling", left_F, right_F])
    rows.append(["AP1", left_T, right_T])
    rows.append(["", "", ""])
    rows.append(["Grounding A6", f"G:{a6}", f"G:{a6}"])
    rows.append(["Grounding Rack", f"G:{a6}", f"G:{a6}"])
    rows.append(["", "", ""])
    rows.append(["Power cable labeling", f"F:{a6}", f"F: SMPS_DC_LOAD {sufnum}"])
    rows.append(["Power cable labeling", f"T: SMPS_DC_LOAD {sufnum}", f"T:{a6}"])
    rows.append(["", "", ""])
    rows.append(["", "", ""])
    return rows


def _generate_outputs(df, enb_ids_input: str):
    df = preprocess_main(df)
    requested = _split_enb_list(enb_ids_input)
    outputs = {}
    errors = {}
    for enb_id in requested:
        site_rows = df[df["eNBsiteID"].astype(str).str.strip().str.lower() == enb_id.strip().lower()].copy()
        if site_rows.empty:
            site_rows = df[df["eNBsiteID"].astype(str).str.contains(enb_id, case=False, na=False)].copy()
        if site_rows.empty:
            errors[enb_id] = "No rows found in MAIN."
            continue
        sectors = site_rows["GIS SECTOR_ID"].dropna().astype(str).unique().tolist()
        sectors = sorted(sectors, key=suffix_num)
        preset = compute_vlkp_decisions(site_rows)
        out_rows = []
        for sec in sectors:
            r = site_rows[site_rows["GIS SECTOR_ID"].astype(str) == sec]
            use_row = r.iloc[0] if not r.empty else site_rows.iloc[0]
            block = build_sector_block(use_row, sec, df, preset)
            out_rows.extend(block)
        out_df = pd.DataFrame(out_rows)
        outputs[enb_id] = out_df
    return outputs, errors, requested

# --------------------------- UI ---------------------------
st.title('Labeller — Output Generator')
st.caption("Generate sector-wise labelling sheets from your **MAIN DATA CG.xlsx** for one or multiple eNB IDs.")

with st.sidebar:
    st.markdown("### 🛰️ A6 — Labeller")
    st.write("- Upload one MAIN sheet.")
    st.write("- Enter **one or many eNB IDs** (comma or newline separated).")
    st.write("- Click **Generate** to build all outputs.")
    st.write("- Pick an eNB on the left to preview & download.")

main_file = st.file_uploader("Upload MAIN DATA CG.xlsx", type=["xlsx"])
enb_ids_input = st.text_area("eNB IDs (one per line or comma-separated)", height=120, placeholder="I-XX-YYYY-ENB-0001\nI-XX-YYYY-ENB-0002")
go = st.button("Generate", type="primary")

@st.cache_data(show_spinner=False)
def _read_main(uploaded):
    return pd.ExcelFile(uploaded, engine='openpyxl').parse(0)


def _split_enb_list(text: str):
    if not text:
        return []
    # normalize commas to newlines, then split
    lines = (text or "").replace(",", "").splitlines()
    tokens = [ln.strip() for ln in lines if ln and ln.strip()]
    # de-duplicate while preserving order (case-insensitive key)
    out, seen = [], set()
    for t in tokens:
        key = t.upper()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


if go:
    if not main_file or not enb_ids_input.strip():
        st.error("Please upload MAIN and enter at least one eNB ID.")
    else:
        try:
            df = _read_main(main_file)
            outputs, errors, requested = _generate_outputs(df, enb_ids_input)
            st.session_state["outputs"] = outputs
            st.session_state["errors"] = errors
            st.session_state["requested"] = requested
            # Keep last df bytes to allow re-download without re-upload (optional)
            st.session_state["generated"] = True
        except Exception as e:
            st.error("Failed to generate outputs.")
            st.exception(e)

# Render results if available in session_state (so switching selection doesn't require re-Generate)
if st.session_state.get("generated") and st.session_state.get("outputs"):
    outputs = st.session_state["outputs"]
    errors = st.session_state.get("errors", {})
    left, right = st.columns([0.35, 0.65])

    with left:
        st.subheader("Results")
        ok_list = list(outputs.keys())
        if ok_list:
            default_index = 0
            if "sel_enb" in st.session_state and st.session_state["sel_enb"] in ok_list:
                default_index = ok_list.index(st.session_state["sel_enb"])
            sel = st.radio("Select an eNB to preview:", ok_list, index=default_index, key="sel_enb")
            # Combined ZIP download
            import io, zipfile
            zipbuf = io.BytesIO()
            with zipfile.ZipFile(zipbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for enb, df_out in outputs.items():
                    bio = io.BytesIO()
                    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                        df_out.to_excel(writer, index=False, header=False, sheet_name="Sheet1")
                    bio.seek(0)
                    zf.writestr(f"{enb}_output.xlsx", bio.getvalue())
            zipbuf.seek(0)
            st.download_button("⬇️ Download ALL (ZIP)", data=zipbuf.getvalue(), file_name="labeller_outputs.zip", mime="application/zip")
        else:
            st.info("No valid outputs generated.")

        if errors:
            st.warning("Some IDs could not be generated:")
            for enb, msg in errors.items():
                st.write(f"- **{enb}**: {msg}")

    with right:
        if outputs and ok_list:
            st.subheader(f"Preview — {st.session_state.get('sel_enb', ok_list[0])}")
            st.dataframe(outputs[st.session_state.get('sel_enb', ok_list[0])], use_container_width=True, height=500)
            # Per-eNB download
            import io
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                outputs[st.session_state["sel_enb"]].to_excel(writer, index=False, header=False, sheet_name="Sheet1")
            bio.seek(0)
            st.download_button(f"⬇️ Download {st.session_state['sel_enb']}.xlsx", data=bio.getvalue(),
                               file_name=f"{st.session_state['sel_enb']}_output.xlsx",
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
else:
    st.info("Ready — upload MAIN, paste multiple eNB IDs, then click **Generate**.")
