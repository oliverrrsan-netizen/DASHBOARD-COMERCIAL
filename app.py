import io
import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Dashboard Ventas", page_icon="📊", layout="wide")

COLOR_BLUE   = "#1F4E79"
COLOR_GREEN  = "#00B050"
COLOR_ORANGE = "#ED7D31"
COLOR_PURPLE = "#7030A0"
COLOR_RED    = "#C00000"

# ============================================================
# FUNCIONES DE LIMPIEZA Y SOPORTE
# ============================================================
def norm_txt(s) -> str:
    if pd.isna(s): return ""
    s = str(s).upper().strip().replace("\ufeff", "").replace("\x00", "")
    for a, b in zip("ÁÉÍÓÚÑ", "AEIOUN"):
        s = s.replace(a, b)
    return re.sub(r'[^A-Z0-9]', '', s)

def limpiar_num(v) -> float:
    if v is None or (isinstance(v, float) and pd.isna(v)): return 0.0
    if isinstance(v, (int, float, np.number)): return float(v)
    t = str(v).replace("$", "").replace(",", "").replace(" ", "").replace("\ufeff", "").replace("\x00", "").strip()
    if t == "" or t.lower() == "nan": return 0.0
    try: return float(t)
    except Exception: return 0.0

def forzar_fecha(v):
    if v is None or pd.isna(v): return pd.NaT
    if isinstance(v, pd.Timestamp): return v
    s = str(v).strip().replace("\ufeff", "").replace("\x00", "")
    if " " in s: s = s.split(" ")[0]
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d"):
        try: return pd.to_datetime(s, format=fmt)
        except Exception: pass
    try: return pd.to_datetime(s, dayfirst=True)
    except Exception: return pd.NaT

def find_col(df, *candidatos):
    nmap = {c: norm_txt(c) for c in df.columns}
    for cand in candidatos:
        nc = norm_txt(cand)
        for col, nn in nmap.items():
            if nn == nc: return col
    return None

def detectar_sep(texto: str) -> str:
    primera = texto.split("\n", 1)[0]
    ntab = primera.count("\t")
    ncom = primera.count(",")
    npun = primera.count(";")
    if ntab >= ncom and ntab >= npun and ntab > 0: return "\t"
    if ncom >= npun and ncom > 0: return ","
    if npun > 0: return ";"
    return "\t"

def extraer_unicos(serie):
    vals = []
    for v in serie.dropna().unique():
        s = str(v).strip().replace("\x00", "")
        if s.lower() not in ["nan", "none", "", "ninguno"]:
            vals.append(s)
    return sorted(list(set(vals)))

# ============================================================
# CARGA DE ARCHIVOS
# ============================================================
@st.cache_data(show_spinner="Cargando archivo...")
def cargar_txt(contenido: bytes) -> pd.DataFrame:
    if contenido.startswith(b'\xff\xfe') or contenido.startswith(b'\xfe\xff'):
        texto = contenido.decode('utf-16')
    else:
        try: texto = contenido.decode('utf-8-sig')
        except UnicodeDecodeError: texto = contenido.decode('latin-1')
    
    texto = texto.replace("\x00", "")
    sep = detectar_sep(texto)
    
    df = pd.read_csv(io.StringIO(texto), sep=sep, dtype=str, on_bad_lines="skip")
    df.columns = [str(c).strip().replace("\ufeff", "").replace("\x00", "") for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.replace("\ufeff", "", regex=False).str.replace("\x00", "", regex=False)
    return df

def _preparar_ventas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cf = find_col(df, "Fecha de documento", "Fecha")
    df["_fecha"] = df[cf].apply(forzar_fecha) if cf else pd.NaT

    df["_cantidad"] = df[find_col(df, "Cantidad")].apply(limpiar_num) if find_col(df, "Cantidad") else 0.0
    df["_total"] = df[find_col(df, "Total Neto", "Total")].apply(limpiar_num) if find_col(df, "Total Neto", "Total") else 0.0
    df["_pvp"] = df[find_col(df, "PVP", "Precio sin IVA")].apply(limpiar_num) if find_col(df, "PVP", "Precio sin IVA") else 0.0

    c_marca = find_col(df, "U_Marca", "Marca")
    c_lin = find_col(df, "Línea de Negocio", "Linea de Negocio", "Proveedor")
    
    if c_marca:
        df["_marca"] = df[c_marca].astype(str).str.strip().replace("\x00", "")
        if c_lin:
            mask = df["_marca"].isin(["", "nan", "None", "Ninguno"])
            df.loc[mask, "_marca"] = df.loc[mask, c_lin].astype(str).str.strip().replace("\x00", "")
    elif c_lin:
        df["_marca"] = df[c_lin].astype(str).str.strip().replace("\x00", "")
    else:
        df["_marca"] = ""

    for k, *cands in [("_tienda", "Tienda"), ("_sku", "SKU"), ("_desc", "Descripción", "Descripcion"), 
                      ("_folio", "Folio SAP", "Folio"), ("_gestion", "U_Gestion", "Gestion"), 
                      ("_depto", "U_Departamento", "Departamento"), ("_prenda", "U_Prenda", "Prenda")]:
        c = find_col(df, *cands)
        df[k] = df[c].astype(str).str.strip() if c else ""
    return df

def _preparar_entradas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cf = find_col(df, "Fecha de recepción", "Fecha recepción", "Fecha")
    df["_fecha"] = df[cf].apply(forzar_fecha) if cf else pd.NaT

    df["_cantidad"] = df[find_col(df, "Cantidad Recibida", "Cantidad")].apply(limpiar_num) if find_col(df, "Cantidad Recibida", "Cantidad") else 0.0
    df["_costo"] = df[find_col(df, "Costo", "Costo Unitario", "Precio sin IVA")].apply(limpiar_num) if find_col(df, "Costo", "Costo Unitario", "Precio sin IVA") else 0.0

    c_marca = find_col(df, "Marca", "U_Marca")
    c_lin = find_col(df, "Línea de Negocio", "Proveedor")
    if c_marca:
        df["_marca"] = df[c_marca].astype(str).str.strip()
        if c_lin:
            mask = df["_marca"].isin(["", "nan", "None", "Ninguno"])
            df.loc[mask, "_marca"] = df.loc[mask, c_lin].astype(str).str.strip()
    elif c_lin:
        df["_marca"] = df[c_lin].astype(str).str.strip()
    else:
        df["_marca"] = ""

    for k, *cands in [("_sku", "SKU"), ("_desc", "Descripción"), ("_gestion", "Gestion", "U_Gestion"), 
                      ("_depto", "Departamento", "U_Departamento"), ("_prenda", "Prenda", "U_Prenda")]:
        c = find_col(df, *cands)
        df[k] = df[c].astype(str).str.strip() if c else ""
    return df

def _preparar_oc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for k, *cands in [("_sku", "Número de artículo", "SKU"), ("_desc", "Descripción artículo/serv.", "Descripción"), 
                      ("_estatus", "Estatus"), ("_doc", "Número de documento")]:
        c = find_col(df, *cands)
        df[k] = df[c].astype(str).str.strip() if c else ""

    df["_cantidad"] = df[find_col(df, "Cantidad")].apply(limpiar_num) if find_col(df, "Cantidad") else 0.0
    df["_costo"] = df[find_col(df, "Precio unitario", "Costo")].apply(limpiar_num) if find_col(df, "Precio unitario", "Costo") else 0.0
    
    cf = find_col(df, "Fecha de contabilización", "Fecha")
    df["_fecha"] = df[cf].apply(forzar_fecha) if cf else pd.NaT
    return df

@st.cache_data(show_spinner="Procesando ventas...")
def preparar_ventas(contenido: bytes) -> pd.DataFrame:
    return _preparar_ventas(cargar_txt(contenido))

@st.cache_data(show_spinner="Procesando entradas...")
def preparar_entradas(contenido: bytes) -> pd.DataFrame:
    return _preparar_entradas(cargar_txt(contenido))

@st.cache_data(show_spinner="Calculando costos por SKU...")
def costos_por_sku(contenido_oc: bytes) -> dict:
    if not contenido_oc: return {}
    df = _preparar_oc(cargar_txt(contenido_oc))
    if df.empty: return {}

    df["_sku_up"] = df["_sku"].astype(str).str.upper().str.strip()
    df["_estatus_norm"] = df["_estatus"].map(norm_txt)

    mask = ((df["_estatus_norm"] != "CANCELADO") & (df["_costo"] > 0) & (df["_cantidad"] > 0) & (df["_sku_up"] != ""))
    df = df[mask]
    if df.empty: return {}

    df["_costo_x_cant"] = df["_costo"] * df["_cantidad"]
    agg = df.groupby("_sku_up", as_index=False).agg(costo_x=("_costo_x_cant", "sum"), cant=("_cantidad", "sum"))
    agg = agg[agg["cant"] > 0].copy()
    agg["costo_prom"] = agg["costo_x"] / agg["cant"]
    
    return {str(row["_sku_up"]): float(row["costo_prom"]) for _, row in agg.iterrows()}

def costo_sku(dict_costos: dict, sku: str, fallback: float) -> float:
    return dict_costos.get(str(sku).upper(), fallback)

# ============================================================
# INTERFAZ LATERAL Y CARGA
# ============================================================
st.sidebar.title("📂 Archivos")
st.sidebar.caption("Sube los TXT. Los datos NO se envían a servidores externos.")

def get_file_bytes(label, key, data_folder_names):
    uploaded = st.sidebar.file_uploader(label, type=["txt", "tsv", "csv"], key=key)
    if uploaded is not None: return uploaded.getvalue(), uploaded.name
    for name in data_folder_names:
        p = Path("data") / name
        if p.exists(): return p.read_bytes(), p.name
    return None, None

ventas_bytes, ventas_name = get_file_bytes("📊 Ventas (.txt)", "f_ventas", ["Detalle_Ventas_historico_081126.txt", "ventas.txt"])
entradas_bytes, entradas_name = get_file_bytes("📦 Entradas (.txt)", "f_entradas", ["entradas.txt", "Entradas.txt", "Detalle Entrada de Mercancia_081126.txt"])
oc_bytes, oc_name = get_file_bytes("💰 Detalle OC (.txt)", "f_oc", ["detalle_oc.txt", "DetalleOC.txt", "oc.txt", "Detalle OC_OCT.txt"])

if not ventas_bytes:
    st.title("📊 Dashboard de Ventas")
    st.info("Sube el archivo de **Ventas** en la barra lateral.")
    st.stop()

ventas = preparar_ventas(ventas_bytes)
st.sidebar.success(f"Ventas: {len(ventas):,} filas")

if entradas_bytes:
    entradas = preparar_entradas(entradas_bytes)
    st.sidebar.success(f"Entradas: {len(entradas):,} filas")
else:
    entradas = pd.DataFrame()
    st.sidebar.warning("Sin Entradas: inventario incompleto.")

if oc_bytes:
    dict_costos = costos_por_sku(oc_bytes)
    st.sidebar.success(f"Costos OC: {len(dict_costos):,} SKUs")
else:
    dict_costos = {}
    st.sidebar.warning("Sin OC: costos tomados de Entradas.")

# ============================================================
# FILTROS GLOBALES
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("🔎 Filtros")

marcas = extraer_unicos(ventas["_marca"])
marca_sel = st.sidebar.selectbox("Marca", ["TODAS"] + marcas, index=0)

fechas_validas = ventas["_fecha"].dropna()
if len(fechas_validas) > 0:
    fmin, fmax = fechas_validas.min().date(), fechas_validas.max().date()
else:
    fmin = pd.Timestamp("2020-01-01").date()
    fmax = pd.Timestamp.today().date()
    st.sidebar.warning("⚠️ Ajusta el rango de fechas manualmente.")

col1, col2 = st.sidebar.columns(2)
fdesde = col1.date_input("Desde", value=fmin) 
fhasta = col2.date_input("Hasta", value=fmax)

if fhasta < fdesde: fdesde, fhasta = fhasta, fdesde
st.sidebar.caption(f"Rango: {fdesde.strftime('%d/%m/%Y')} → {fhasta.strftime('%d/%m/%Y')}")

def filtrar_marca_fechas(df: pd.DataFrame, marca: str, d1, d2) -> pd.DataFrame:
    if df.empty: return df
    mask = (df["_fecha"] >= pd.Timestamp(d1)) & (df["_fecha"] <= pd.Timestamp(d2))
    if marca != "TODAS":
        mask &= df["_marca"].str.upper().str.contains(marca.upper(), na=False)
    return df[mask].copy()

ventas_f = filtrar_marca_fechas(ventas, marca_sel, fdesde, fhasta)

# ============================================================
# PESTAÑAS
# ============================================================
tab_dash, tab_cons, tab_inv, tab_st, tab_invtie = st.tabs(
    ["📊 Dashboard", "🔎 Consulta", "📦 Inventario", "📈 Sell-Through", "🏬 Inv. por Tienda"]
)

def kpi(col, label, value, color=COLOR_BLUE):
    col.markdown(f"""
    <div style="background:{color}; color:white; padding:14px; border-radius:10px; text-align:center; font-family:Segoe UI;">
        <div style="font-size:12px; opacity:.85;">{label}</div>
        <div style="font-size:20px; font-weight:700;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def money(x): return f"${x:,.2f}"
def num(x):   return f"{x:,.0f}"

# ============================================================
# TAB 1: DASHBOARD
# ============================================================
with tab_dash:
    st.subheader(f"Dashboard · Marca: {marca_sel} · {fdesde:%d/%m/%Y} al {fhasta:%d/%m/%Y}")
    if ventas_f.empty:
        st.warning("No hay datos para el filtro seleccionado.")
    else:
        venta_total = ventas_f["_total"].sum()
        piezas      = ventas_f["_cantidad"].sum()
        tickets     = ventas_f["_folio"].nunique()
        ticket_prom = venta_total / tickets if tickets else 0

        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Venta Total", money(venta_total), COLOR_BLUE)
        kpi(c2, "Tickets", num(tickets), COLOR_GREEN)
        kpi(c3, "Ticket Promedio", money(ticket_prom), COLOR_ORANGE)
        kpi(c4, "Piezas Netas", num(piezas), COLOR_PURPLE)

        if not entradas.empty:
            entradas_f = entradas.copy()
            if marca_sel != "TODAS":
                entradas_f = entradas_f[entradas_f["_marca"].str.upper().str.contains(marca_sel.upper(), na=False)]
            
            stock_sku = entradas_f.groupby(entradas_f["_sku"].str.upper())["_cantidad"].sum()
            ventas_sku_tot = ventas.groupby(ventas["_sku"].str.upper())["_cantidad"].sum()
            pvp_sku = ventas.groupby(ventas["_sku"].str.upper())["_pvp"].max()
            
            stock_final = (stock_sku.subtract(ventas_sku_tot, fill_value=0)).clip(lower=0)
            stock_final = stock_final[stock_final > 0]

            costo_sku_map = {}
            costo_ent = (entradas_f.assign(cost=entradas_f["_costo"] * entradas_f["_cantidad"])
                          .groupby(entradas_f["_sku"].str.upper())
                          .apply(lambda g: g["_costo"].mul(g["_cantidad"]).sum() / max(g["_cantidad"].sum(), 1), include_groups=False))
            
            for sku in stock_final.index:
                costo_sku_map[sku] = costo_sku(dict_costos, sku, float(costo_ent.get(sku, 0)))

            inv_pvp = sum(stock_final.get(s, 0) * pvp_sku.get(s, 0) for s in stock_final.index)
            capital_inm = sum(stock_final[s] * costo_sku_map[s] for s in stock_final.index if ventas_sku_tot.get(s, 0) == 0)
            st_val = (piezas / (piezas + stock_final.sum())) if (piezas + stock_final.sum()) > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            kpi(c1, "Inventario (PVP)", money(inv_pvp), COLOR_BLUE)
            kpi(c2, "Capital Inmovilizado", money(capital_inm), COLOR_RED)
            kpi(c3, "Liquidez Stock", f"{st_val:.1%}", COLOR_GREEN)
            kpi(c4, "Alerta", "CRÍTICA" if inv_pvp and capital_inm / inv_pvp > 0.3 else "SANA",
                COLOR_RED if inv_pvp and capital_inm / inv_pvp > 0.3 else COLOR_GREEN)

        st.markdown("---")
        df_mes = (ventas_f.assign(mes=ventas_f["_fecha"].dt.strftime("%Y-%m"))
                          .groupby("mes").agg(venta=("_total", "sum"), piezas=("_cantidad", "sum"))
                          .reset_index().sort_values("mes"))
        st.markdown("### Venta por mes")
        st.plotly_chart(px.bar(df_mes, x="mes", y="venta", text="piezas", labels={"mes": "Mes", "venta": "Venta Neta"}, color_discrete_sequence=[COLOR_BLUE]), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Top Tiendas")
            top_t = (ventas_f.groupby("_tienda").agg(venta=("_total", "sum"), piezas=("_cantidad", "sum"))
                            .sort_values("venta", ascending=False).head(15).reset_index())
            fig = px.bar(top_t, x="venta", y="_tienda", orientation="h", labels={"_tienda": "", "venta": "Venta Neta"}, color_discrete_sequence=[COLOR_GREEN])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top_t, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("### Top Artículos")
            top_a = (ventas_f.assign(art=ventas_f["_sku"] + " - " + ventas_f["_desc"].str.slice(0, 40))
                            .groupby("art").agg(venta=("_total", "sum"), piezas=("_cantidad", "sum"))
                            .sort_values("venta", ascending=False).head(15).reset_index())
            fig = px.bar(top_a, x="venta", y="art", orientation="h", labels={"art": "", "venta": "Venta Neta"}, color_discrete_sequence=[COLOR_PURPLE])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top_a, use_container_width=True, hide_index=True)

# ============================================================
# TAB 2: CONSULTA
# ============================================================
with tab_cons:
    st.subheader("Consulta detallada")
    
    prendas = extraer_unicos(ventas["_prenda"])
    deptos  = extraer_unicos(ventas["_depto"])

    c1, c2, c3 = st.columns([2, 2, 2])
    prenda_sel = c1.selectbox("U_Prenda", ["TODAS"] + prendas, key="cons_prenda")
    depto_sel  = c2.selectbox("U_Departamento", ["TODOS"] + deptos, key="cons_depto")
    sku_txt    = c3.text_input("SKU (contiene)", key="cons_sku")

    mask = ((ventas["_fecha"] >= pd.Timestamp(fdesde)) & (ventas["_fecha"] <= pd.Timestamp(fhasta)))
    if marca_sel != "TODAS": mask &= ventas["_marca"].str.upper().str.contains(marca_sel.upper(), na=False)
    if prenda_sel != "TODAS": mask &= ventas["_prenda"].str.upper() == prenda_sel.upper()
    if depto_sel != "TODOS": mask &= ventas["_depto"].str.upper() == depto_sel.upper()
    if sku_txt.strip(): mask &= ventas["_sku"].str.upper().str.contains(sku_txt.strip().upper(), na=False)

    ventas_c = ventas[mask].copy()

    if ventas_c.empty:
        st.warning("Sin resultados para el filtro.")
    else:
        vt = ventas_c["_total"].sum()
        pz = ventas_c["_cantidad"].sum()
        tk = ventas_c["_folio"].nunique()
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Venta Total", money(vt), COLOR_BLUE)
        kpi(c2, "Tickets", num(tk), COLOR_GREEN)
        kpi(c3, "Ticket Promedio", money(vt/tk if tk else 0), COLOR_ORANGE)
        kpi(c4, "Piezas", num(pz), COLOR_PURPLE)

        st.markdown("### Ventas por tienda")
        df_t = (ventas_c.groupby("_tienda").agg(venta=("_total", "sum"), piezas=("_cantidad", "sum"), lineas=("_folio", "count"))
                        .sort_values("venta", ascending=False).reset_index())
        
        # Porcentaje multiplicado por 100
        df_t["participacion"] = (df_t["venta"] / df_t["venta"].sum()) * 100
        
        st.dataframe(df_t.rename(columns={"_tienda": "Tienda", "venta": "Venta Neta", "piezas": "Piezas", "lineas": "Líneas", "participacion": "% Particip."}),
            use_container_width=True, hide_index=True, column_config={"% Particip.": st.column_config.NumberColumn(format="%.2f%%")})

        st.markdown("### Resumen por Semana Retail (SR)")
        df_sem = ventas_c.copy()
        df_sem["inicio"] = df_sem["_fecha"] - pd.to_timedelta(df_sem["_fecha"].dt.weekday, unit="D")
        df_sem["fin"]    = df_sem["inicio"] + pd.Timedelta(days=6)
        df_sem["SR"]     = df_sem["_fecha"].dt.isocalendar().week.astype(int)
        agg_sem = (df_sem.groupby(["SR", "inicio", "fin"]).agg(piezas=("_cantidad", "sum"), venta=("_total", "sum")).reset_index().sort_values("inicio"))
        agg_sem["Semana"] = "SR " + agg_sem["SR"].astype(str)
        
        # Gráfico en Total Efectivo ($)
        fig_sr = px.bar(agg_sem, x="Semana", y="venta", text="venta", labels={"Semana": "Semana Retail", "venta": "Total Efectivo ($)"}, color_discrete_sequence=[COLOR_ORANGE])
        fig_sr.update_traces(texttemplate='%{text:$,.2f}', textposition='outside')
        st.plotly_chart(fig_sr, use_container_width=True)

        st.dataframe(agg_sem[["Semana", "inicio", "fin", "piezas", "venta"]].rename(columns={"inicio": "Inicio (Lunes)", "fin": "Fin (Domingo)", "piezas": "Piezas", "venta": "Venta Neta ($)"}),
            use_container_width=True, hide_index=True)

        st.markdown("### Tendencia diaria")
        agg_dia = (ventas_c.groupby(ventas_c["_fecha"].dt.date).agg(piezas=("_cantidad", "sum"), venta=("_total", "sum")).reset_index().rename(columns={"_fecha": "fecha"}))
        st.plotly_chart(px.line(agg_dia, x="fecha", y="venta", markers=True, labels={"fecha": "Fecha", "venta": "Venta Neta"}, color_discrete_sequence=[COLOR_BLUE]), use_container_width=True)

# ============================================================
# TAB 3: INVENTARIO
# ============================================================
with tab_inv:
    st.subheader("Matriz ABC y decisión de compra")
    if entradas.empty:
        st.warning("Sin archivo de Entradas no es posible calcular inventario.")
    else:
        entradas_f = entradas.copy()
        if marca_sel != "TODAS": entradas_f = entradas_f[entradas_f["_marca"].str.upper().str.contains(marca_sel.upper(), na=False)]

        agg_e = entradas_f.assign(sku_up=entradas_f["_sku"].str.upper()).groupby("sku_up").agg(
            gestion=("_gestion", "first"), depto=("_depto", "first"), prenda=("_prenda", "first"), desc=("_desc", "first")
        ).reset_index().rename(columns={"sku_up": "sku"})
        
        tmp = entradas_f.copy()
        tmp["_costo_x_cant"] = tmp["_costo"] * tmp["_cantidad"]
        sumas = tmp.groupby(tmp["_sku"].str.upper()).agg(cant=("_cantidad", "sum"), costo_x=("_costo_x_cant", "sum"))
        sumas["costo_prom"] = sumas["costo_x"] / sumas["cant"].replace(0, np.nan)

        ventas_sku_total = ventas.groupby(ventas["_sku"].str.upper())["_cantidad"].sum()
        ventas_sku_rango = ventas_f.groupby(ventas_f["_sku"].str.upper()).agg(qty=("_cantidad", "sum"), dinero=("_total", "sum"))
        skus = set(entradas_f["_sku"].str.upper()) | set(ventas["_sku"].str.upper())
        dias = max((pd.Timestamp(fhasta) - pd.Timestamp(fdesde)).days + 1, 1)

        filas = []
        for sku in skus:
            if not sku: continue
            stock = max(float(sumas["cant"].get(sku, 0)) - float(ventas_sku_total.get(sku, 0)), 0.0)
            costo_final = costo_sku(dict_costos, sku, float(sumas["costo_prom"].get(sku, 0)))
            vendido_r = float(ventas_sku_rango["qty"].get(sku, 0)) if sku in ventas_sku_rango.index else 0.0
            dinero_r  = float(ventas_sku_rango["dinero"].get(sku, 0)) if sku in ventas_sku_rango.index else 0.0
            venta_dia = vendido_r / dias

            mask_sku = agg_e["sku"] == sku
            desc = agg_e.loc[mask_sku, "desc"].iloc[0] if mask_sku.any() else ""
            gestion = agg_e.loc[mask_sku, "gestion"].iloc[0] if mask_sku.any() else ""
            depto = agg_e.loc[mask_sku, "depto"].iloc[0] if mask_sku.any() else ""
            prenda = agg_e.loc[mask_sku, "prenda"].iloc[0] if mask_sku.any() else ""

            filas.append({
                "SKU": sku, "Descripción": desc, "U_Gestion": gestion, "U_Departamento": depto, "U_Prenda": prenda,
                "Vendido Rango": vendido_r, "Venta Total ($)": dinero_r, "Stock Final": stock,
                "Costo Prom": costo_final, "Valor Stock ($)": stock * costo_final, "Venta Diaria": venta_dia, "Sugerido Compra": max(venta_dia * 30 - stock, 0),
            })

        df_inv = pd.DataFrame(filas)
        if not df_inv.empty:
            # PANEL DE TOTALES ARRIBA
            st.markdown("### Resumen de Inventario")
            c1, c2, c3, c4 = st.columns(4)
            kpi(c1, "SKUs Totales", num(len(df_inv)), COLOR_BLUE)
            kpi(c2, "Unidades en Stock", num(df_inv["Stock Final"].sum()), COLOR_GREEN)
            kpi(c3, "Valor Stock (Costo)", money(df_inv["Valor Stock ($)"].sum()), COLOR_ORANGE)
            kpi(c4, "Ventas en Rango ($)", money(df_inv["Venta Total ($)"].sum()), COLOR_PURPLE)
            st.markdown("---")

            df_inv = df_inv.sort_values("Valor Stock ($)", ascending=False).reset_index(drop=True)
            tot_val = df_inv["Valor Stock ($)"].sum()
            df_inv["% Acum"] = (df_inv["Valor Stock ($)"].cumsum() / tot_val * 100) if tot_val else 0
            df_inv["Clase"] = df_inv["% Acum"].apply(lambda p: "A" if p <= 80 else ("B" if p <= 95 else "C"))
            df_inv["Estado"] = df_inv.apply(lambda r: "QUIEBRE" if r["Stock Final"] == 0 and r["Vendido Rango"] > 0 else ("REPONER" if r["Sugerido Compra"] > 0 else ("INMOVILIZADO" if r["Venta Diaria"] == 0 and r["Stock Final"] > 0 else "ÓPTIMO")), axis=1)
            df_inv["Rotación ($)"] = np.where(df_inv["Valor Stock ($)"] > 0, df_inv["Venta Total ($)"] / df_inv["Valor Stock ($)"], 0)

            cols_show = ["SKU", "Descripción", "U_Gestion", "U_Departamento", "U_Prenda", "Vendido Rango", "Venta Total ($)", "Stock Final", "Costo Prom", "Valor Stock ($)", "Clase", "% Acum", "Venta Diaria", "Sugerido Compra", "Rotación ($)", "Estado"]
            st.dataframe(df_inv[cols_show], use_container_width=True, hide_index=True, height=520,
                column_config={"% Acum": st.column_config.NumberColumn(format="%.1f"), "Venta Total ($)": st.column_config.NumberColumn(format="$%.2f"), "Valor Stock ($)": st.column_config.NumberColumn(format="$%.2f"), "Costo Prom": st.column_config.NumberColumn(format="$%.2f"), "Rotación ($)": st.column_config.NumberColumn(format="%.2fx"), "Venta Diaria": st.column_config.NumberColumn(format="%.2f")})

            st.markdown("### Proyección de compra por rubro")
            df_rub = (df_inv.assign(Rubro=df_inv["U_Gestion"] + " | " + df_inv["U_Departamento"] + " | " + df_inv["U_Prenda"])
                            .groupby("Rubro").agg(**{"Venta Histórica ($)": ("Venta Total ($)", "sum"), "Sugerido Compra (Pzas)": ("Sugerido Compra", "sum")})
                            .sort_values("Venta Histórica ($)", ascending=False).reset_index())
            st.dataframe(df_rub, use_container_width=True, hide_index=True)

# ============================================================
# TAB 4: SELL-THROUGH
# ============================================================
with tab_st:
    st.subheader("Sell-Through y cobertura de stock")
    if entradas.empty:
        st.warning("Sin archivo de Entradas no es posible calcular Sell-Through.")
    else:
        entradas_f = entradas.copy()
        if marca_sel != "TODAS": entradas_f = entradas_f[entradas_f["_marca"].str.upper().str.contains(marca_sel.upper(), na=False)]

        tmp = entradas_f.copy()
        tmp["_costo_x_cant"] = tmp["_costo"] * tmp["_cantidad"]
        sumas = tmp.groupby(tmp["_sku"].str.upper()).agg(cant=("_cantidad", "sum"), costo_x=("_costo_x_cant", "sum"), desc=("_desc", "first"))
        sumas["costo_prom"] = sumas["costo_x"] / sumas["cant"].replace(0, np.nan)

        ventas_tot_sku = ventas.groupby(ventas["_sku"].str.upper())["_cantidad"].sum()
        ventas_rango_sku = ventas_f.groupby(ventas_f["_sku"].str.upper()).agg(qty=("_cantidad", "sum"), dinero=("_total", "sum"))
        pvp_sku = ventas.groupby(ventas["_sku"].str.upper())["_pvp"].max()
        dias = max((pd.Timestamp(fhasta) - pd.Timestamp(fdesde)).days + 1, 1)
        skus = set(sumas.index) | set(ventas_tot_sku.index)

        filas = []
        for sku in skus:
            if not sku: continue
            stock = max(float(sumas["cant"].get(sku, 0)) - float(ventas_tot_sku.get(sku, 0)), 0.0)
            vendido_r = float(ventas_rango_sku["qty"].get(sku, 0)) if sku in ventas_rango_sku.index else 0.0
            st_val = (vendido_r / (vendido_r + stock)) * 100 if (vendido_r + stock) else 0
            vdia = vendido_r / dias
            cobertura = (stock / vdia) / 7 if vdia > 0 else (999 if stock > 0 else 0)
            pvp = float(pvp_sku.get(sku, 0))
            costo = costo_sku(dict_costos, sku, float(sumas["costo_prom"].get(sku, 0)))
            pventa = pvp * 0.9 if vendido_r > 0 and pvp > 0 else 0
            margen = ((pventa - costo) / pventa * 100) if pventa > 0 and costo > 0 else 0

            alerta = "QUIEBRE" if stock == 0 and vendido_r > 0 else ("SOBRESTOCK" if st_val > 80 and cobertura < 4 else ("LENTO" if st_val < 20 and cobertura > 12 else "OK"))
            filas.append({
                "SKU": sku, "Descripción": str(sumas["desc"].get(sku, "")), "Ventas Rango (Unid)": vendido_r, "Stock Final": stock,
                "Sell-Through": st_val, "PVP Ref": pvp, "Precio Venta Prom": pventa, "Costo Promedio": costo, "Margen %": margen,
                "Cobertura (Semanas)": cobertura, "Alerta": alerta,
            })

        df_st = pd.DataFrame(filas)
        if df_st.empty:
            st.warning("No hay datos de ventas o inventario para calcular el Sell-Through.")
        else:
            df_st = df_st.sort_values("Sell-Through", ascending=False)
            inv_val = (df_st["Stock Final"] * df_st["Costo Promedio"]).sum()
            st_global = (df_st["Ventas Rango (Unid)"].sum() / max(df_st["Ventas Rango (Unid)"].sum() + df_st["Stock Final"].sum(), 1)) * 100
            c1, c2, c3 = st.columns(3)
            kpi(c1, "Ventas Totales", money(float(ventas_f["_total"].sum())), COLOR_BLUE)
            kpi(c2, "Inventario (costo)", money(inv_val), COLOR_ORANGE)
            kpi(c3, "Sell-Through Global", f"{st_global:.1f}%", COLOR_PURPLE)
            
            st.dataframe(df_st, use_container_width=True, hide_index=True, height=520,
                column_config={"Sell-Through": st.column_config.NumberColumn(format="%.1f"), "Margen %": st.column_config.NumberColumn(format="%.1f"), "PVP Ref": st.column_config.NumberColumn(format="$%.2f"), "Precio Venta Prom": st.column_config.NumberColumn(format="$%.2f"), "Costo Promedio": st.column_config.NumberColumn(format="$%.2f"), "Cobertura (Semanas)": st.column_config.NumberColumn(format="%.1f")})

# ============================================================
# TAB 5: INVENTARIO POR TIENDA
# ============================================================
with tab_invtie:
    st.subheader("Inventario y ventas por tienda (estimado)")
    if entradas.empty:
        st.warning("Sin Entradas no es posible estimar el stock.")
    else:
        tiendas = extraer_unicos(ventas["_tienda"])
        tienda_sel = st.selectbox("Tienda", ["TOTALES"] + tiendas, key="invtie_tienda")
        entradas_f = entradas.copy()
        if marca_sel != "TODAS": entradas_f = entradas_f[entradas_f["_marca"].str.upper().str.contains(marca_sel.upper(), na=False)]

        tmp = entradas_f.copy()
        tmp["_costo_x_cant"] = tmp["_costo"] * tmp["_cantidad"]
        sumas_e = tmp.groupby(tmp["_sku"].str.upper()).agg(cant=("_cantidad", "sum"), costo_x=("_costo_x_cant", "sum"), desc=("_desc", "first"))
        sumas_e["costo_prom"] = sumas_e["costo_x"] / sumas_e["cant"].replace(0, np.nan)

        ventas_tot_sku = ventas.groupby(ventas["_sku"].str.upper())["_cantidad"].sum()
        stock_total = (sumas_e["cant"] - ventas_tot_sku.reindex(sumas_e.index, fill_value=0)).clip(lower=0)
        ventas_tie = ventas_f.copy()
        if tienda_sel != "TOTALES": ventas_tie = ventas_tie[ventas_tie["_tienda"] == tienda_sel]

        if ventas_tie.empty:
            st.info("Sin ventas para el filtro.")
        else:
            grp = (ventas_tie.assign(sku_up=ventas_tie["_sku"].str.upper()).groupby(["sku_up", "_tienda"]).agg(vendido=("_cantidad", "sum"), dinero=("_total", "sum")).reset_index())
            total_sku_rango = ventas_tie.groupby(ventas_tie["_sku"].str.upper())["_cantidad"].sum()
            filas = []
            for _, r in grp.iterrows():
                sku = r["sku_up"]
                stk = float(stock_total.get(sku, 0))
                tot = float(total_sku_rango.get(sku, 0))
                prop = (r["vendido"] / tot) if tot > 0 else 0
                stk_asig = round(stk * prop)
                costo = costo_sku(dict_costos, sku, float(sumas_e["costo_prom"].get(sku, 0)))

                filas.append({
                    "Tienda": r["_tienda"], "SKU": sku, "Descripción": str(sumas_e["desc"].get(sku, "")),
                    "Piezas Vendidas": r["vendido"], "Venta Neta ($)": r["dinero"], "Stock Estimado": stk_asig,
                    "Valor Stock ($)": stk_asig * costo, "Costo Promedio": costo,
                    "Origen Costo": "OC" if str(sku).upper() in dict_costos else "Entradas", "Fuente Asignación": "Asignado por % venta",
                })

            if tienda_sel == "TOTALES":
                huerf = stock_total[~stock_total.index.isin(total_sku_rango.index) & (stock_total > 0)]
                if len(huerf):
                    # Corrección de error Index para NumPy
                    costo_herf = np.array([costo_sku(dict_costos, s, float(sumas_e["costo_prom"].get(s, 0))) for s in huerf.index])
                    filas.append({
                        "Tienda": "ALMACÉN / SIN VENTAS", "SKU": "VARIOS", "Descripción": "Productos con stock que no se vendieron en este periodo",
                        "Piezas Vendidas": 0, "Venta Neta ($)": 0, "Stock Estimado": float(huerf.sum()),
                        "Valor Stock ($)": float((huerf.values * costo_herf).sum()), "Costo Promedio": 0,
                        "Origen Costo": "Mixto", "Fuente Asignación": "Stock no vendido",
                    })

            df_tie = pd.DataFrame(filas)
            st.dataframe(df_tie, use_container_width=True, hide_index=True, height=520,
                column_config={"Venta Neta ($)": st.column_config.NumberColumn(format="$%.2f"), "Valor Stock ($)": st.column_config.NumberColumn(format="$%.2f"), "Costo Promedio": st.column_config.NumberColumn(format="$%.2f")})

            st.markdown("**Totales**")
            c1, c2, c3 = st.columns(3)
            kpi(c1, "Piezas Vendidas", num(df_tie["Piezas Vendidas"].sum()), COLOR_GREEN)
            kpi(c2, "Venta Neta", money(df_tie["Venta Neta ($)"].sum()), COLOR_BLUE)
            kpi(c3, "Valor Stock", money(df_tie["Valor Stock ($)"].sum()), COLOR_ORANGE)