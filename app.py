from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    import seaborn as sns  # opcional
    _HAS_SEABORN = True
except ModuleNotFoundError:
    sns = None
    _HAS_SEABORN = False


# -----------------------------
# Configuración general
# -----------------------------
APP_TITLE = "CASO MARKETING BANCARIO"
APP_SUBTITLE = "Exploratory Data Analysis"
DATASET_EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan", "contact", "month",
    "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y",
]


# -----------------------------
# Tema (CSS)
# -----------------------------
def apply_dark_dashboard_theme() -> None:
    """
    Aplica un tema estilo dashboard dark (similar al look&feel de la imagen de referencia).
    Se prioriza: contraste, jerarquía visual, tarjetas tipo KPI, paneles y barra superior.
    """
    css = """
    <style>
      :root {
        --bg0: #071321;
        --bg1: #0b1b2b;
        --bg2: #0f2438;
        --panel: rgba(14, 26, 41, 0.85);
        --panel2: rgba(18, 33, 52, 0.88);
        --stroke: rgba(255, 255, 255, 0.09);
        --stroke2: rgba(255, 255, 255, 0.14);
        --text: rgba(255, 255, 255, 0.92);
        --muted: rgba(255, 255, 255, 0.65);

        --shadow: 0 10px 30px rgba(0,0,0,.35);
        --shadow2: 0 6px 18px rgba(0,0,0,.28);
        --radius: 16px;
      }

      .stApp {
        background: radial-gradient(1200px 800px at 20% 0%, rgba(61,165,255,0.12), transparent 60%),
                    radial-gradient(1000px 700px at 100% 10%, rgba(255,159,26,0.10), transparent 60%),
                    linear-gradient(160deg, var(--bg0) 0%, var(--bg1) 40%, var(--bg2) 100%);
        color: var(--text);
      }

      [data-testid="stHeader"] {
        background: rgba(7, 19, 33, 0.55);
        border-bottom: 1px solid var(--stroke);
        backdrop-filter: blur(8px);
      }

      [data-testid="stSidebar"] {
        background: rgba(7, 19, 33, 0.82);
        border-right: 1px solid var(--stroke);
      }
      [data-testid="stSidebar"] * { color: var(--text); }

      html, body, [class*="css"] {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Liberation Sans", sans-serif;
      }

      .topbar {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 12px;
        padding: 10px 14px;
        margin-top: 14px;
        margin-bottom: 10px;
        border: 1px solid var(--stroke);
        background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
        border-radius: var(--radius);
        box-shadow: var(--shadow2);
      }
      .brand {
        display:flex;
        align-items:center;
        gap: 10px;
        font-weight: 800;
        letter-spacing: .6px;
      }
      .brand .logo {
        width: 36px; height: 36px;
        display:flex; align-items:center; justify-content:center;
        border-radius: 10px;
        background: linear-gradient(145deg, rgba(61,165,255,0.22), rgba(255,159,26,0.18));
        border: 1px solid var(--stroke2);
        box-shadow: 0 10px 22px rgba(0,0,0,.30);
      }
      .brand .title { font-size: 18px; }
      .brand .subtitle {
        display:block;
        font-size: 12px;
        font-weight: 600;
        color: var(--muted);
        margin-top: -2px;
      }
      .top-icons { display:flex; align-items:center; gap: 8px; }
      .iconbtn {
        width: 34px; height: 34px;
        border-radius: 10px;
        display:flex; align-items:center; justify-content:center;
        border: 1px solid var(--stroke);
        background: rgba(255,255,255,0.05);
        box-shadow: 0 10px 20px rgba(0,0,0,.22);
        color: var(--text);
        user-select:none;
      }

      .panel {
        border: 1px solid var(--stroke);
        background: linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%);
        border-radius: var(--radius);
        padding: 12px 12px 10px 12px;
        box-shadow: var(--shadow);
      }
      .panel-title {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 10px;
        margin-bottom: 10px;
      }
      .panel-title h3 {
        margin: 0;
        font-size: 15px;
        letter-spacing: .2px;
      }
      .panel-title .hint {
        color: var(--muted);
        font-size: 12px;
      }

      .kpi {
        border-radius: var(--radius);
        border: 1px solid var(--stroke);
        padding: 12px 14px;
        box-shadow: var(--shadow2);
        position: relative;
        overflow: hidden;
        min-height: 86px;
      }
      .kpi:before {
        content:"";
        position:absolute; inset:-40px -40px auto auto;
        width: 120px; height: 120px;
        border-radius: 999px;
        opacity: 0.30;
      }
      .kpi .kpi-label {
        font-size: 12px;
        font-weight: 700;
        color: rgba(255,255,255,0.78);
        letter-spacing: .3px;
      }
      .kpi .kpi-value {
        margin-top: 6px;
        font-size: 30px;
        font-weight: 900;
        letter-spacing: .2px;
        line-height: 1.05;
      }
      .kpi .kpi-sub {
        margin-top: 4px;
        font-size: 12px;
        color: rgba(255,255,255,0.72);
      }

      .kpi.green { background: linear-gradient(180deg, rgba(20,184,122,0.18), rgba(12,59,45,0.55)); }
      .kpi.green:before { background: rgba(20,184,122,0.45); }

      .kpi.red { background: linear-gradient(180deg, rgba(255,77,109,0.17), rgba(58,15,24,0.55)); }
      .kpi.red:before { background: rgba(255,77,109,0.45); }

      .kpi.blue { background: linear-gradient(180deg, rgba(61,165,255,0.18), rgba(16,42,71,0.55)); }
      .kpi.blue:before { background: rgba(61,165,255,0.45); }

      .kpi.orange { background: linear-gradient(180deg, rgba(255,159,26,0.18), rgba(58,38,15,0.55)); }
      .kpi.orange:before { background: rgba(255,159,26,0.45); }

      [data-baseweb="tab-list"] { gap: 6px; }
      [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        padding: 8px 12px;
      }
      [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(61,165,255,0.14);
        border-color: rgba(61,165,255,0.25);
      }

      .stDataFrame, [data-testid="stDataFrame"] {
        border: 1px solid var(--stroke);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: var(--shadow2);
      }

      .stTextInput input, .stNumberInput input, .stSelectbox div, .stMultiSelect div { border-radius: 12px !important; }
      .stButton button {
        border-radius: 12px !important;
        border: 1px solid rgba(61,165,255,0.25) !important;
        background: rgba(61,165,255,0.12) !important;
        color: var(--text) !important;
      }
      .stButton button:hover {
        border-color: rgba(61,165,255,0.45) !important;
        background: rgba(61,165,255,0.20) !important;
      }

      [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--stroke);
        border-radius: 14px;
        padding: 10px 12px;
        box-shadow: var(--shadow2);
      }

      .block-container { padding-top: 2.4rem; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_topbar() -> None:
    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">
            <div class="logo">🧮</div>
            <div>
              <div class="title">{APP_TITLE}</div>
              <span class="subtitle">{APP_SUBTITLE}</span>
            </div>
          </div>
          <div class="top-icons">
            <div class="iconbtn" title="Usuario">🧠</div>
            <div class="iconbtn" title="Numero">📉</div>
            <div class="iconbtn" title="Estadistica">📊</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", variant: str = "blue") -> None:
    variant = variant if variant in {"green", "red", "blue", "orange"} else "blue"
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else '<div class="kpi-sub">&nbsp;</div>'
    st.markdown(
        f"""
        <div class="kpi {variant}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_open(title: str, hint: str = "") -> None:
    hint_html = f'<div class="hint">{hint}</div>' if hint else '<div class="hint">&nbsp;</div>'
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-title">
            <h3>{title}</h3>
            {hint_html}
          </div>
        """,
        unsafe_allow_html=True,
    )


def panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Utilidades de datos
# -----------------------------
def _infer_separator(sample_text: str) -> str:
    semicolons = sample_text.count(";")
    commas = sample_text.count(",")
    return ";" if semicolons >= max(1, commas) else ","


@st.cache_data(show_spinner=False)
def read_csv_safely(raw_bytes: bytes) -> pd.DataFrame:
    if not raw_bytes:
        raise ValueError("El archivo está vacío.")

    sample = raw_bytes[:4096].decode("utf-8", errors="ignore")
    sep = _infer_separator(sample)

    buf = io.BytesIO(raw_bytes)
    df = pd.read_csv(buf, sep=sep, quotechar='"', encoding="utf-8", engine="python")

    if df.shape[1] == 1 and isinstance(df.columns[0], str) and ";" in df.columns[0]:
        buf = io.BytesIO(raw_bytes)
        df = pd.read_csv(buf, sep=";", quotechar='"', encoding="utf-8", engine="python")

    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


def safe_value_counts(series: pd.Series, top_n: int = 20) -> pd.DataFrame:
    vc = series.value_counts(dropna=False).head(max(1, int(top_n)))
    out = vc.to_frame(name="count")
    out["proportion"] = (out["count"] / out["count"].sum()).round(4)
    out.index = out.index.astype(str)
    return out.reset_index(names="value")


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:.2f}%"


# -----------------------------
# POO – Clase principal
# -----------------------------
@dataclass(frozen=True)
class DataAnalyzer:
    df: pd.DataFrame

    def _numeric_cols(self) -> List[str]:
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def _categorical_cols(self) -> List[str]:
        return self.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def info_text(self) -> str:
        buf = io.StringIO()
        self.df.info(buf=buf)
        return buf.getvalue()

    def nulls_summary(self) -> pd.DataFrame:
        nulls = self.df.isna().sum().sort_values(ascending=False)
        out = nulls.to_frame(name="null_count")
        out["null_pct"] = (out["null_count"] / len(self.df)).round(4)
        return out

    def variable_classification(self) -> Dict[str, List[str]]:
        numeric = self._numeric_cols()
        categorical = self._categorical_cols()
        other = [c for c in self.df.columns if c not in numeric + categorical]
        return {"numéricas": numeric, "categóricas": categorical, "otras": other}

    def describe_numeric(self) -> pd.DataFrame:
        numeric = self._numeric_cols()
        return self.df[numeric].describe().T if numeric else pd.DataFrame()

    def describe_categorical(self) -> pd.DataFrame:
        categorical = self._categorical_cols()
        return self.df[categorical].describe().T if categorical else pd.DataFrame()

    def central_tendency(self, col: str) -> Dict[str, float]:
        if col not in self.df.columns:
            raise KeyError(f"Columna inexistente: {col}")

        s = pd.to_numeric(self.df[col], errors="coerce")
        mode_series = s.mode(dropna=True)

        return {
            "media": float(s.mean(skipna=True)),
            "mediana": float(s.median(skipna=True)),
            "moda": float(mode_series.iloc[0]) if not mode_series.empty else np.nan,
            "std": float(s.std(skipna=True)),
        }

    def acceptance_rate(self) -> float:
        if "y" not in self.df.columns:
            return np.nan
        s = self.df["y"].astype(str).str.strip().str.lower()
        return float((s == "yes").mean())

    def group_acceptance(self, by_col: str, top_n: int = 15) -> pd.DataFrame:
        if "y" not in self.df.columns or by_col not in self.df.columns:
            return pd.DataFrame()

        tmp = self.df[[by_col, "y"]].copy()
        tmp["y_bin"] = tmp["y"].astype(str).str.strip().str.lower().eq("yes").astype(int)

        grp = (
            tmp.groupby(by_col, dropna=False)["y_bin"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "accept_rate"})
            .sort_values(["count", "accept_rate"], ascending=[False, False])
            .head(max(1, int(top_n)))
            .reset_index()
            .rename(columns={by_col: "group"})
        )
        return grp

    def crosstab(self, a: str, b: str, normalize: Optional[str]) -> pd.DataFrame:
        if a not in self.df.columns or b not in self.df.columns:
            return pd.DataFrame()
        return pd.crosstab(self.df[a], self.df[b], normalize=normalize, dropna=False)


# -----------------------------
# Plot helpers (fallback sin seaborn)
# -----------------------------
def _plot_hist(ax, data: pd.Series, bins: int, kde: bool) -> None:
    if _HAS_SEABORN:
        sns.histplot(data, bins=bins, kde=kde, ax=ax)
    else:
        ax.hist(data.dropna().values, bins=bins)
    ax.set_title("Histograma")
    ax.grid(alpha=0.15)


def _plot_box(ax, data: pd.Series) -> None:
    if _HAS_SEABORN:
        sns.boxplot(x=data, ax=ax)
    else:
        ax.boxplot(data.dropna().values, vert=False)
    ax.set_title("Boxplot")
    ax.grid(alpha=0.15)


def _plot_barh(ax, df_plot: pd.DataFrame, x: str, y: str) -> None:
    if _HAS_SEABORN:
        sns.barplot(data=df_plot, x=x, y=y, ax=ax)
    else:
        ax.barh(df_plot[y].astype(str), df_plot[x].values)
    ax.set_title("Conteos")
    ax.grid(alpha=0.15)


def _plot_heatmap(ax, ct: pd.DataFrame) -> None:
    if _HAS_SEABORN:
        sns.heatmap(ct, annot=False, ax=ax)
    else:
        im = ax.imshow(ct.values, aspect="auto")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(ct.shape[1]), labels=[str(c) for c in ct.columns])
        ax.set_yticks(np.arange(ct.shape[0]), labels=[str(i) for i in ct.index])
    ax.set_title("Heatmap")


# -----------------------------
# UI – Páginas
# -----------------------------
def page_home() -> None:
    render_topbar()

    panel_open("¡Bienvenidos!", hint="Proyecto Final")
    st.write(
        "📌 Resumen: Este proyecto es una aplicación en **Streamlit** que permite cargar un **archivo CSV** del " \
        "caso **Bank Marketing** y explorarlo de forma interactiva."\
        "Muestra métricas clave, distribuciones, valores faltantes y comparaciones entre variables" \
        "para entender qué factores se relacionan con la aceptación de la campaña (y), y cerrar con " \
        "conclusiones basadas en los hallazgos del Exploratory Data Analysis o Análisis Exploratorio de Datos (EDA)."

        
    )
    st.write("**📌 Autor:** Jeancarlos Amaya Quispe")
    st.write("**📌 Especialización:** Python for Analytics")
    st.write("**📌 Año:** 2026")
    
    st.write("**📌 Tecnologías:** 🧩 Streamlit / 🧩Pandas / 🧩 NumPy / 🧩 Matplotlib"  + (" / 🧩 Seaborn" if _HAS_SEABORN else ""))
    with st.expander("Variables", expanded=False):
        st.code(", ".join(DATASET_EXPECTED_COLUMNS), language="text")
    panel_close()


def page_load_dataset() -> None:
    render_topbar()

    panel_open("Carga del dataset", hint="CSV requerido para habilitar el EDA")
    st.info("Cargue el archivo **.csv**. Si el dataset no está cargado, el módulo EDA no se ejecutará.")

    uploaded = st.file_uploader("Seleccione el archivo CSV (BankMarketing.csv)", type=["csv"])
    if uploaded is None:
        st.warning("Aún no se ha cargado ningún archivo.")
        st.session_state.pop("df", None)
        panel_close()
        return

    try:
        raw = uploaded.getvalue()
        df = read_csv_safely(raw)

        if df.empty or df.shape[1] < 5:
            raise ValueError("El archivo no parece un dataset válido (sin filas o con muy pocas columnas).")

        st.session_state["df"] = df

        st.success("Archivo cargado correctamente.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Total filas", f"{df.shape[0]:,}", variant="green")
        with c2:
            kpi_card("Total columnas", f"{df.shape[1]:,}", variant="blue")
        with c3:
            kpi_card("Nulos totales", f"{int(df.isna().sum().sum()):,}", variant="red")
        with c4:
            miss = [c for c in DATASET_EXPECTED_COLUMNS if c not in df.columns]
            kpi_card("Columnas faltantes", f"{len(miss)}", sub=("Revisar estructura" if miss else "OK"), variant="orange")

        st.subheader("Vista previa")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Chequeo de columnas")
        missing_cols = [c for c in DATASET_EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            st.warning("El dataset no contiene todas las columnas esperadas. Se continuará con columnas disponibles.")
            st.write("Columnas faltantes:", missing_cols)
        else:
            st.success("Estructura de columnas alineada con lo esperado para BankMarketing.")

    except Exception as exc:
        st.session_state.pop("df", None)
        st.error(f"No se pudo cargar el archivo. Detalle: {exc}")

    panel_close()


def _require_df() -> Optional[pd.DataFrame]:
    df = st.session_state.get("df")
    if df is None:
        st.warning("Primero cargue el dataset en el módulo **Carga del dataset**.")
        return None
    return df


def page_eda() -> None:
    render_topbar()
    df = _require_df()
    if df is None:
        return

    analyzer = DataAnalyzer(df=df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total registros", f"{len(df):,}", variant="green")
    with c2:
        kpi_card("Nulos totales", f"{int(df.isna().sum().sum()):,}", variant="red")
    with c3:
        kpi_card("Aceptación (y=yes)", format_pct(analyzer.acceptance_rate()), sub="Tasa global", variant="blue")
    with c4:
        kpi_card("Columnas", f"{df.shape[1]}", variant="orange")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
        [
            "1) Info general",
            "2) Clasificación",
            "3) Descriptiva",
            "4) Faltantes",
            "5) Numéricas",
            "6) Categóricas",
            "7) Num vs Cat",
            "8) Cat vs Cat",
            "9) Dinámico",
            "10) Hallazgos",
        ]
    )

    with tab1:
        panel_open("Ítem 1 · Información general", hint="Estructura y tipos")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.code(analyzer.info_text(), language="text")
        with col2:
            nulls = analyzer.nulls_summary()
            nulls_show = nulls[nulls["null_count"] > 0].copy()
            if nulls_show.empty:
                st.success("No se detectaron valores nulos.")
            else:
                nulls_show["null_pct"] = nulls_show["null_pct"].map(format_pct)
                st.dataframe(nulls_show, use_container_width=True)
        panel_close()

    with tab2:
        panel_open("Ítem 2 · Clasificación de variables", hint="Numéricas vs Categóricas")
        cls = analyzer.variable_classification()
        left, right = st.columns(2)
        with left:
            st.write(f"**Numéricas ({len(cls['numéricas'])})**")
            st.code("\n".join(cls["numéricas"]) if cls["numéricas"] else "—", language="text")
        with right:
            st.write(f"**Categóricas ({len(cls['categóricas'])})**")
            st.code("\n".join(cls["categóricas"]) if cls["categóricas"] else "—", language="text")
        if cls["otras"]:
            st.warning("Se detectaron columnas no clasificadas (revisar dtypes).")
            st.code("\n".join(cls["otras"]), language="text")
        panel_close()

    with tab3:
        panel_open("Ítem 3 · Estadística descriptiva", hint="Resumen y tendencia central")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Variables numéricas**")
            st.dataframe(analyzer.describe_numeric(), use_container_width=True)
        with c2:
            st.write("**Variables categóricas**")
            st.dataframe(analyzer.describe_categorical(), use_container_width=True)

        num_cols = analyzer.variable_classification()["numéricas"]
        if num_cols:
            st.divider()
            pick = st.selectbox("Seleccione variable numérica", options=num_cols, index=0)
            stats = analyzer.central_tendency(pick)
            a, b, c, d = st.columns(4)
            with a:
                kpi_card("Media", f"{stats['media']:.3f}", variant="blue")
            with b:
                kpi_card("Mediana", f"{stats['mediana']:.3f}", variant="green")
            with c:
                kpi_card("Moda", f"{stats['moda']:.3f}" if not np.isnan(stats["moda"]) else "—", variant="orange")
            with d:
                kpi_card("Desv. Est.", f"{stats['std']:.3f}", variant="red")
        panel_close()

    with tab4:
        panel_open("Ítem 4 · Valores faltantes", hint="Conteo y visualización")
        nulls = analyzer.nulls_summary().reset_index(names="column")
        nulls = nulls[nulls["null_count"] > 0].copy()

        if nulls.empty:
            st.success("No se detectaron valores faltantes.")
        else:
            nulls["null_pct"] = nulls["null_pct"].map(format_pct)
            st.dataframe(nulls, use_container_width=True)

            fig, ax = plt.subplots()
            ax.bar(nulls["column"], nulls["null_count"])
            ax.set_title("Nulos por columna")
            ax.set_xlabel("Columna")
            ax.set_ylabel("Nulos")
            ax.tick_params(axis="x", rotation=75)
            ax.grid(alpha=0.15)
            fig.patch.set_alpha(0)
            st.pyplot(fig, clear_figure=True)
        panel_close()

    with tab5:
        panel_open("Ítem 5 · Distribución numéricas", hint="Histograma y boxplot")
        num_cols = analyzer.variable_classification()["numéricas"]
        if not num_cols:
            st.info("No hay columnas numéricas disponibles.")
            panel_close()
        else:
            col_a, col_b, col_c = st.columns([1.2, 1, 1])
            with col_a:
                num_col = st.selectbox("Variable numérica", options=num_cols, index=0, key="num_dist_col")
            with col_b:
                bins = st.slider("Bins", min_value=10, max_value=80, value=30, step=5)
            with col_c:
                kde = st.checkbox("Mostrar KDE", value=True)

            fig, ax = plt.subplots()
            _plot_hist(ax, df[num_col], bins=int(bins), kde=bool(kde))
            fig.patch.set_alpha(0)
            st.pyplot(fig, clear_figure=True)

            fig2, ax2 = plt.subplots()
            _plot_box(ax2, df[num_col])
            fig2.patch.set_alpha(0)
            st.pyplot(fig2, clear_figure=True)
            panel_close()

    with tab6:
        panel_open("Ítem 6 · Variables categóricas", hint="Frecuencias por categoría")
        cat_cols = analyzer.variable_classification()["categóricas"]
        if not cat_cols:
            st.info("No hay columnas categóricas disponibles.")
            panel_close()
        else:
            c1, c2 = st.columns([1.2, 1])
            with c1:
                cat_col = st.selectbox("Variable categórica", options=cat_cols, index=0, key="cat_col")
            with c2:
                top_n = st.slider("Top N", min_value=5, max_value=30, value=15, step=1)

            vc = safe_value_counts(df[cat_col], top_n=int(top_n))
            st.dataframe(vc, use_container_width=True)

            fig, ax = plt.subplots()
            _plot_barh(ax, vc, x="count", y="value")
            ax.set_xlabel("Conteo")
            ax.set_ylabel("")
            fig.patch.set_alpha(0)
            st.pyplot(fig, clear_figure=True)
            panel_close()

    with tab7:
        panel_open("Ítem 7 · Numérico vs Categórico", hint="Comparación por grupos")
        num_cols = analyzer.variable_classification()["numéricas"]
        cat_cols = analyzer.variable_classification()["categóricas"]

        if not num_cols or not cat_cols:
            st.info("Se requieren columnas numéricas y categóricas.")
            panel_close()
        else:
            left, right = st.columns(2)
            with left:
                x_num = st.selectbox("Variable numérica", options=num_cols, index=0, key="biv_num")
            with right:
                default_cat = "y" if "y" in df.columns else cat_cols[0]
                x_cat = st.selectbox(
                    "Variable categórica (grupo)",
                    options=cat_cols,
                    index=cat_cols.index(default_cat) if default_cat in cat_cols else 0,
                    key="biv_cat",
                )

            unique_vals = sorted(df[x_cat].astype(str).unique().tolist())[:200]
            selected_vals = st.multiselect(
                "Filtrar categorías (opcional)",
                options=unique_vals,
                default=unique_vals[: min(6, len(unique_vals))],
            )

            data = df[[x_num, x_cat]].copy()
            data[x_cat] = data[x_cat].astype(str)
            if selected_vals:
                data = data[data[x_cat].isin(selected_vals)]

            fig, ax = plt.subplots()
            if _HAS_SEABORN:
                sns.boxplot(data=data, x=x_cat, y=x_num, ax=ax)
            else:
                groups = [g[x_num].dropna().values for _, g in data.groupby(x_cat)]
                ax.boxplot(groups, labels=[str(k) for k in data[x_cat].unique()], vert=True)
                ax.set_ylabel(x_num)
                ax.set_xlabel(x_cat)
            ax.set_title(f"{x_num} vs {x_cat}")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(alpha=0.15)
            fig.patch.set_alpha(0)
            st.pyplot(fig, clear_figure=True)

            grp = data.groupby(x_cat, dropna=False)[x_num].agg(["count", "mean", "median", "std"]).reset_index()
            st.dataframe(grp, use_container_width=True)
            panel_close()

    with tab8:
        panel_open("Ítem 8 · Categórico vs Categórico", hint="Crosstab + heatmap")
        cat_cols = analyzer.variable_classification()["categóricas"]
        if len(cat_cols) < 2:
            st.info("Se requieren al menos 2 variables categóricas.")
            panel_close()
        else:
            a, b, c = st.columns([1.2, 1.2, 1])
            with a:
                cat_a = st.selectbox("Categoría A", options=cat_cols, index=0, key="cat_a")
            with b:
                cat_b_opts = [x for x in cat_cols if x != cat_a]
                cat_b = st.selectbox("Categoría B", options=cat_b_opts, index=0, key="cat_b")
            with c:
                norm_opt = st.selectbox(
                    "Normalización",
                    options=["Ninguna", "Por filas", "Por columnas", "Global"],
                    index=1,
                )
            normalize = {"Ninguna": None, "Por filas": "index", "Por columnas": "columns", "Global": "all"}[norm_opt]

            ct = analyzer.crosstab(cat_a, cat_b, normalize=normalize)
            st.dataframe(ct, use_container_width=True)

            if ct.size > 0:
                fig, ax = plt.subplots()
                _plot_heatmap(ax, ct)
                fig.patch.set_alpha(0)
                st.pyplot(fig, clear_figure=True)
            panel_close()

    with tab9:
        panel_open("Ítem 9 · Análisis dinámico", hint="Filtros + KPI de aceptación")
        cls = analyzer.variable_classification()
        num_cols = cls["numéricas"]
        cat_cols = cls["categóricas"]

        st.markdown("**1) Filtros categóricos**")
        filt_cat = st.selectbox("Columna categórica", options=cat_cols, index=0, key="filt_cat_col") if cat_cols else None
        selected_cat_vals: List[str] = []
        if filt_cat:
            vals = sorted(df[filt_cat].astype(str).unique().tolist())
            selected_cat_vals = st.multiselect("Valores permitidos", options=vals, default=vals[: min(5, len(vals))], key="filt_cat_vals")

        st.markdown("**2) Filtros numéricos**")
        filt_num = st.selectbox("Columna numérica", options=num_cols, index=0, key="filt_num_col") if num_cols else None
        num_range: Tuple[float, float] = (0.0, 0.0)
        if filt_num:
            col_min = float(np.nanmin(df[filt_num].values))
            col_max = float(np.nanmax(df[filt_num].values))
            num_range = st.slider("Rango permitido", min_value=col_min, max_value=col_max, value=(col_min, col_max), key="filt_num_rng")

        filtered = df.copy()
        if filt_cat and selected_cat_vals:
            filtered = filtered[filtered[filt_cat].astype(str).isin(selected_cat_vals)]
        if filt_num:
            filtered = filtered[(filtered[filt_num] >= num_range[0]) & (filtered[filt_num] <= num_range[1])]

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Filas filtradas", f"{len(filtered):,}", variant="green")
        with c2:
            kpi_card(
                "Aceptación filtrada",
                format_pct(DataAnalyzer(filtered).acceptance_rate()) if "y" in filtered.columns else "—",
                variant="blue",
            )
        with c3:
            show = st.checkbox("Mostrar muestra filtrada", value=False)

        if show:
            st.dataframe(filtered.head(50), use_container_width=True)
        panel_close()

    with tab10:
        panel_open("Ítem 10 · Hallazgos clave", hint="Síntesis ejecutiva")
        if "y" not in df.columns:
            st.warning("No se encontró la columna 'y'. No se puede calcular aceptación.")
            panel_close()
            return

        rate = analyzer.acceptance_rate()
        st.write(f"**Aceptación global (y=yes):** {format_pct(rate)}")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Aceptación por Job (Top)**")
            g_job = analyzer.group_acceptance("job", top_n=12)
            st.dataframe(g_job, use_container_width=True)
            if not g_job.empty:
                fig, ax = plt.subplots()
                if _HAS_SEABORN:
                    sns.barplot(data=g_job, x="accept_rate", y="group", ax=ax)
                else:
                    ax.barh(g_job["group"].astype(str), g_job["accept_rate"].values)
                ax.set_title("Accept rate por job (Top)")
                ax.set_xlabel("Accept rate")
                ax.set_ylabel("")
                ax.grid(alpha=0.15)
                fig.patch.set_alpha(0)
                st.pyplot(fig, clear_figure=True)

        with c2:
            st.write("**Aceptación por Education (Top)**")
            g_edu = analyzer.group_acceptance("education", top_n=12)
            st.dataframe(g_edu, use_container_width=True)
            if not g_edu.empty:
                fig, ax = plt.subplots()
                if _HAS_SEABORN:
                    sns.barplot(data=g_edu, x="accept_rate", y="group", ax=ax)
                else:
                    ax.barh(g_edu["group"].astype(str), g_edu["accept_rate"].values)
                ax.set_title("Accept rate por education (Top)")
                ax.set_xlabel("Accept rate")
                ax.set_ylabel("")
                ax.grid(alpha=0.15)
                fig.patch.set_alpha(0)
                st.pyplot(fig, clear_figure=True)

        insights: List[str] = []
        insights.append(f"La tasa de aceptación global es {format_pct(rate)}; úsela como baseline para comparar segmentos.")

        if "duration" in df.columns and df["duration"].dtype.kind in "if":
            q75 = float(df["duration"].quantile(0.75))
            q25 = float(df["duration"].quantile(0.25))
            insights.append(
                f"duration muestra dispersión relevante (IQR ≈ {q75 - q25:.1f}); segmentar por percentiles puede clarificar rendimiento de gestión."
            )

        if "pdays" in df.columns and df["pdays"].dtype.kind in "if":
            pct_999 = float((df["pdays"] == 999).mean())
            insights.append(
                f"pdays=999 concentra {format_pct(pct_999)}, indicando codificación de 'sin contacto previo' (decidir recodificación para análisis)."
            )

        if "default" in df.columns:
            unk = float((df["default"].astype(str) == "unknown").mean())
            if unk > 0:
                insights.append(
                    f"default contiene 'unknown' en {format_pct(unk)}; afecta interpretabilidad, conviene controlar/limpiar ese estado."
                )

        if "job" in df.columns:
            g_job2 = analyzer.group_acceptance("job", top_n=12)
            if not g_job2.empty:
                best = g_job2.sort_values("accept_rate", ascending=False).iloc[0]
                insights.append(
                    f"job='{best['group']}' destaca por mayor aceptación (≈ {format_pct(float(best['accept_rate']))}) con n={int(best['count'])}; candidato para priorización."
                )

        st.subheader("Insights (síntesis)")
        for i, s in enumerate(insights[:5], start=1):
            st.write(f"{i}. {s}")
        panel_close()


def page_conclusions() -> None:
    render_topbar()
    df = _require_df()
    if df is None:
        return

    analyzer = DataAnalyzer(df=df)
    rate = analyzer.acceptance_rate()

    panel_open("Conclusiones", hint="basadas en EDA")
    concl: List[str] = []
    concl.append(f"La aceptación global (y=yes) es {format_pct(rate)}; se recomienda gestionarla como KPI por segmento y canal.")

    if "duration" in df.columns and df["duration"].dtype.kind in "if":
        concl.append(
            "La distribución de duration evidencia heterogeneidad de interacción; trabajar con percentiles facilita definir umbrales operativos."
        )
    else:
        concl.append("duration no es numérica consistente; validar y estandarizar su calidad antes de conclusiones operativas.")

    if "pdays" in df.columns and df["pdays"].dtype.kind in "if":
        pct_999 = float((df["pdays"] == 999).mean())
        concl.append(
            f"El valor 999 en pdays (≈ {format_pct(pct_999)}) debe tratarse como categoría especial; recodificarlo evita distorsiones en análisis."
        )
    else:
        concl.append("pdays no está disponible o no es numérica; limitar inferencias de contacto previo.")

    if "job" in df.columns:
        g_job = analyzer.group_acceptance("job", top_n=8)
        if not g_job.empty:
            best = g_job.sort_values("accept_rate", ascending=False).iloc[0]
            concl.append(
                f"Se observa diferenciación por job; el segmento '{best['group']}' presenta mayor aceptación relativa (≈ {format_pct(float(best['accept_rate']))})."
            )
        else:
            concl.append("No fue posible estimar aceptación por job; revisar tipos o presencia de y.")
    else:
        concl.append("job no está presente; no se puede segmentar por ocupación.")

    if "education" in df.columns:
        g_edu = analyzer.group_acceptance("education", top_n=8)
        if not g_edu.empty:
            low = g_edu.sort_values("accept_rate", ascending=True).iloc[0]
            concl.append(
                f"Existen segmentos con menor aceptación (ej. education='{low['group']}', ≈ {format_pct(float(low['accept_rate']))}); ajustar mensaje/canal podría mejorar conversión."
            )
        else:
            concl.append("No fue posible estimar aceptación por education; revisar tipos o presencia de y.")
    else:
        concl.append("education no está presente; no se puede segmentar por nivel educativo.")

    concl = concl[:5]
    for i, c in enumerate(concl, start=1):
        st.write(f"{i}. {c}")

    st.caption("Nota: conclusiones basadas en asociaciones observadas (EDA), sin inferir causalidad.")
    panel_close()


def configure_sidebar() -> str:
    
    st.sidebar.title("Menú")
    st.sidebar.caption("Navegación principal")

    page = st.sidebar.radio(
        "Seleccione un módulo",
        options=["Home", "Carga del dataset", "EDA", "Conclusiones"],
        index=0,
    )

    st.sidebar.divider()
    df = st.session_state.get("df")
    if df is None:
        st.sidebar.warning("Dataset no cargado.")
    else:
        st.sidebar.success("Dataset cargado.")
        st.sidebar.write(f"Filas: **{len(df):,}**")
        st.sidebar.write(f"Columnas: **{df.shape[1]}**")

    st.sidebar.divider()
    st.sidebar.caption("Dependencias")
    st.sidebar.write(f"Seaborn: **{'OK' if _HAS_SEABORN else 'No instalado'}**")

    return page


def main() -> None:
    
    st.set_page_config(page_title="BankMarketing EDA", page_icon="📈", layout="wide")
    
    apply_dark_dashboard_theme()
    
    
    page = configure_sidebar()
    
    if page == "Home":
        page_home()
    elif page == "Carga del dataset":
        page_load_dataset()
    elif page == "EDA":
        page_eda()
    elif page == "Conclusiones":
        page_conclusions()
    else:
        st.error("Módulo inválido.")


if __name__ == "__main__":
    main()
