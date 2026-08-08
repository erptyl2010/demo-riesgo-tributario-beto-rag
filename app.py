"""Demo publica del clasificador tributario y recuperacion semantica RAG.

La aplicacion usa exclusivamente operaciones y fragmentos sinteticos. El modelo
BETO ajustado se carga una sola vez y tambien se reutiliza para obtener vectores
densos, evitando cargar un segundo Transformer en un despliegue con pocos
recursos.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = APP_DIR / "modelos" / "beto_riesgo_experto"
MODEL_DIR = Path(os.getenv("BETO_MODEL_PATH", DEFAULT_MODEL_DIR))
DATA_DIR = APP_DIR / "datos_demo"
LABELS = {0: "Bajo Riesgo", 1: "Medio Riesgo", 2: "Alto Riesgo"}
RISK_COLORS = {
    "Bajo Riesgo": "#1d7a55",
    "Medio Riesgo": "#bd7419",
    "Alto Riesgo": "#b43a32",
}


@dataclass(frozen=True)
class Prediction:
    """Resultado de inferencia del clasificador triclase."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class BetoDemoEngine:
    """Motor BETO compartido para clasificacion y recuperacion densa."""

    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model.to("cpu")
        self.model.eval()

        configured_labels = getattr(self.model.config, "id2label", {})
        self.labels = {
            int(label_id): str(label)
            for label_id, label in configured_labels.items()
        } or LABELS

    def predict(self, text: str) -> Prediction:
        """Clasifica una glosa y devuelve probabilidades para las tres clases."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probabilities = self.torch.softmax(logits, dim=-1).cpu().tolist()

        best_id = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        by_label = {
            self.labels.get(index, LABELS.get(index, f"Clase {index}")): float(score)
            for index, score in enumerate(probabilities)
        }
        return Prediction(
            label=self.labels.get(best_id, LABELS.get(best_id, str(best_id))),
            confidence=float(probabilities[best_id]),
            probabilities=by_label,
        )

    def encode(self, texts: list[str]) -> Any:
        """Obtiene embeddings normalizados mediante mean pooling de BETO."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=160,
        )
        with self.torch.inference_mode():
            outputs = self.model.bert(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        token_embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.torch.nn.functional.normalize(pooled, p=2, dim=1).cpu()


def configure_page() -> None:
    """Configura identidad visual y metadatos de la pagina."""
    st.set_page_config(
        page_title="TributIA | Demo PLN",
        page_icon="RC",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,650&display=swap');
        :root {
            --ink: #142b35;
            --cream: #f4efe4;
            --amber: #d89431;
            --green: #1d7a55;
            --red: #b43a32;
        }
        html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
        h1, h2, h3 { font-family: "Fraunces", Georgia, serif; color: var(--ink); }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 92% 4%, rgba(216,148,49,.16), transparent 25rem),
                linear-gradient(180deg, #fbfaf6 0%, #f4efe4 100%);
        }
        [data-testid="stSidebar"] { background: #142b35; }
        [data-testid="stSidebar"] * { color: #f7f1e6; }
        .hero {
            padding: 2.1rem 2.4rem;
            border-radius: 1.25rem;
            color: #f9f3e8;
            background: linear-gradient(115deg, #142b35 0%, #1b4450 62%, #1d7a55 100%);
            box-shadow: 0 18px 50px rgba(20,43,53,.16);
            margin-bottom: 1.2rem;
        }
        .hero h1 { color: #fff8eb; font-size: clamp(2rem, 4vw, 3.4rem); margin: .15rem 0 .5rem; }
        .hero p { max-width: 58rem; color: #dce9e5; font-size: 1.03rem; margin: 0; }
        .eyebrow { color: #f1b85e; font-weight: 700; letter-spacing: .12em; font-size: .76rem; }
        .risk-card {
            border-left: .45rem solid var(--accent);
            padding: 1rem 1.2rem;
            background: rgba(255,255,255,.78);
            border-radius: .8rem;
            box-shadow: 0 8px 24px rgba(20,43,53,.08);
        }
        .rag-card {
            padding: 1rem 1.2rem;
            border: 1px solid rgba(20,43,53,.14);
            background: rgba(255,255,255,.72);
            border-radius: .9rem;
            margin-bottom: .8rem;
        }
        .tag {
            display: inline-block; padding: .2rem .55rem; border-radius: 999px;
            background: #e5efe9; color: #1d5c45; font-size: .78rem; font-weight: 700;
        }
        .small-note { color: #5f6f74; font-size: .86rem; }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.72); border: 1px solid rgba(20,43,53,.11);
            padding: .8rem 1rem; border-radius: .85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_demo_operations() -> pd.DataFrame:
    """Carga las operaciones sinteticas incluidas en la demo publica."""
    return pd.read_csv(DATA_DIR / "operaciones_sinteticas.csv")


@st.cache_data(show_spinner=False)
def load_demo_documents() -> pd.DataFrame:
    """Carga fragmentos sinteticos con trazabilidad documental."""
    return pd.read_csv(DATA_DIR / "fragmentos_sinteticos.csv")


@st.cache_resource(show_spinner="Cargando BETO ajustado en memoria...")
def load_engine() -> BetoDemoEngine:
    """Carga el modelo una sola vez durante la vida de la aplicacion."""
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    missing = [name for name in required if not (MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "No se encontro el modelo BETO de la demo. Faltan: " + ", ".join(missing)
        )
    return BetoDemoEngine(MODEL_DIR)


@st.cache_resource(show_spinner="Construyendo indice semantico de demostracion...")
def load_dense_index() -> Any:
    """Vectoriza una vez los fragmentos sinteticos usando el BETO cargado."""
    engine = load_engine()
    documents = load_demo_documents()
    return engine.encode(documents["texto"].astype(str).tolist())


def normalize_tokens(text: str) -> set[str]:
    """Normaliza tokens para el componente lexical de la recuperacion hibrida."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }


def retrieve_support(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Combina similitud densa BETO y coincidencia lexical para recuperar evidencia."""
    engine = load_engine()
    documents = load_demo_documents()
    document_embeddings = load_dense_index()
    query_embedding = engine.encode([query])[0]
    dense_scores = (document_embeddings @ query_embedding).tolist()

    query_tokens = normalize_tokens(query)
    results: list[dict[str, Any]] = []
    for position, row in documents.iterrows():
        document_tokens = normalize_tokens(str(row["texto"]))
        union = query_tokens | document_tokens
        lexical_score = len(query_tokens & document_tokens) / max(len(union), 1)
        dense_score = (float(dense_scores[position]) + 1.0) / 2.0
        hybrid_score = (0.80 * dense_score) + (0.20 * lexical_score)
        results.append(
            {
                "doc_id": str(row["doc_id"]),
                "pagina": int(row["pagina"]),
                "tipo": str(row["tipo"]),
                "texto": str(row["texto"]),
                "score": hybrid_score,
                "dense_score": dense_score,
                "lexical_score": lexical_score,
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)[:k]


def apply_hard_rules(
    estado_contribuyente: str,
    estado_comprobante: str,
    condicion_domicilio: str,
) -> str:
    """Aplica la regla de seguridad que prioriza senales SUNAT duras."""
    if estado_contribuyente == "NO HABIDO" or estado_comprobante == "NO VALIDO":
        return "Alto Riesgo"
    if condicion_domicilio in {"POR VERIFICAR", "INCONSISTENTE"}:
        return "Medio Riesgo"
    return "Bajo Riesgo"


def final_risk(rule_label: str, semantic_label: str) -> str:
    """Integra reglas y Transformer, manteniendo alertas duras de alto riesgo."""
    if rule_label == "Alto Riesgo":
        return rule_label
    if rule_label == "Medio Riesgo" and semantic_label == "Bajo Riesgo":
        return rule_label
    return semantic_label


def render_sidebar() -> None:
    """Muestra alcance, arquitectura y controles de privacidad."""
    with st.sidebar:
        st.markdown("## TributIA")
        st.caption("Prototipo academico de Maestria UNI-IA")
        st.markdown("---")
        st.markdown("**Arquitectura activa**")
        st.markdown("1. Reglas SUNAT\n2. BETO triclase\n3. Recuperacion hibrida\n4. Evidencia trazable")
        st.markdown("---")
        st.success("Datos publicos: 100 % sinteticos")
        st.warning("No cargar informacion tributaria real en esta DEMO publica.")
        st.caption("Version demostrativa 1.0 | Agosto 2026")


def render_overview() -> None:
    """Presenta alcance, datos de ejemplo y resultados del holdout."""
    operations = load_demo_operations()
    st.subheader("Vista ejecutiva del prototipo")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Operaciones demo", len(operations))
    metric_columns[1].metric("Clases de riesgo", 3)
    metric_columns[2].metric("Etiquetas expertas", 90)
    metric_columns[3].metric("Accuracy BETO", "44.4 %", help="Holdout estratificado de 18 casos")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("### Registro sintetico")
        st.dataframe(
            operations,
            hide_index=True,
            width="stretch",
            column_config={"importe": st.column_config.NumberColumn(format="S/ %.2f")},
        )
    with right:
        st.markdown("### Comparacion experimental")
        metrics = pd.DataFrame(
            {
                "Modelo": ["Heuristico", "TF-IDF + LogReg", "BETO"],
                "Accuracy": [0.278, 0.389, 0.444],
                "F1 macro": [0.258, 0.291, 0.269],
            }
        ).set_index("Modelo")
        st.bar_chart(metrics, color=["#1d7a55", "#d89431"])
        st.caption(
            "Resultados sobre 18 casos de holdout. Son evidencia preliminar, no una "
            "validacion productiva; la clase Alto Riesgo requiere ampliar el corpus."
        )


def render_classifier() -> None:
    """Permite clasificar una operacion escrita por el usuario."""
    st.subheader("Clasificacion interactiva con BETO")
    st.write(
        "Ingrese una glosa y seleccione las senales estructuradas. El resultado final "
        "combina el Transformer con reglas de seguridad tributaria."
    )

    examples = {
        "Servicio regular": "Servicio mensual de mantenimiento de equipos con factura y conformidad.",
        "Sustento parcial": "Gasto de representacion con comprobante valido, pero evidencia de reunion incompleta.",
        "Operacion observada": "Compra excepcional sin contrato ni constancia de recepcion del bien.",
    }
    selected_example = st.selectbox("Ejemplo rapido", list(examples))
    glosa = st.text_area(
        "Glosa o descripcion de la compra",
        value=examples[selected_example],
        height=115,
        max_chars=800,
    )

    col_a, col_b, col_c = st.columns(3)
    estado_contribuyente = col_a.selectbox("Estado del contribuyente", ["ACTIVO", "NO HABIDO"])
    estado_comprobante = col_b.selectbox("Estado del comprobante", ["VALIDO", "NO VALIDO"])
    condicion_domicilio = col_c.selectbox(
        "Condicion del domicilio",
        ["HABIDO", "POR VERIFICAR", "INCONSISTENTE"],
    )

    if st.button("Analizar operacion", type="primary", width="stretch"):
        if len(glosa.strip()) < 10:
            st.warning("Ingrese una glosa de al menos 10 caracteres.")
            return
        try:
            prediction = load_engine().predict(glosa.strip())
        except Exception as exc:
            st.error(f"No fue posible cargar BETO: {exc}")
            return

        rule_label = apply_hard_rules(
            estado_contribuyente,
            estado_comprobante,
            condicion_domicilio,
        )
        integrated_label = final_risk(rule_label, prediction.label)
        accent = RISK_COLORS[integrated_label]
        st.markdown(
            f"""
            <div class="risk-card" style="--accent:{accent}">
              <span class="tag">Resultado ensemble</span>
              <h2 style="margin:.45rem 0;color:{accent}">{integrated_label}</h2>
              <div>BETO: <b>{prediction.label}</b> ({prediction.confidence:.1%}) &nbsp;|&nbsp;
                   Reglas SUNAT: <b>{rule_label}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        probability_frame = pd.DataFrame(
            {"Clase": list(prediction.probabilities), "Probabilidad": list(prediction.probabilities.values())}
        ).set_index("Clase")
        st.bar_chart(probability_frame, color="#1d7a55")
        st.info(
            "La prediccion prioriza una regla dura cuando el contribuyente figura NO HABIDO "
            "o el comprobante es NO VALIDO. El auditor conserva la decision final."
        )


def render_rag() -> None:
    """Permite consultar el corpus sintetico y revisar evidencia trazable."""
    st.subheader("Auditoria semantica RAG")
    st.write(
        "La consulta se vectoriza con BETO y se combina con coincidencia lexical. "
        "Cada respuesta conserva documento, pagina y puntaje."
    )
    query = st.text_input(
        "Consulta del auditor",
        value="¿Existe sustento para un gasto de representacion con evidencia incompleta?",
    )
    k = st.slider("Fragmentos a recuperar", min_value=1, max_value=5, value=3)

    if st.button("Recuperar sustento", type="primary", width="stretch"):
        if len(query.strip()) < 5:
            st.warning("Formule una consulta mas descriptiva.")
            return
        try:
            results = retrieve_support(query.strip(), k=k)
        except Exception as exc:
            st.error(f"No fue posible consultar el indice: {exc}")
            return

        for rank, result in enumerate(results, start=1):
            st.markdown(
                f"""
                <div class="rag-card">
                  <span class="tag">Resultado {rank}</span>
                  <h3 style="margin:.55rem 0 .25rem">{result['doc_id']}</h3>
                  <div class="small-note">Pagina {result['pagina']} · {result['tipo']} ·
                    score hibrido {result['score']:.3f}</div>
                  <p>{result['texto']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Ver composicion del puntaje"):
                st.write(
                    {
                        "similitud_densa_normalizada": round(result["dense_score"], 4),
                        "coincidencia_lexical": round(result["lexical_score"], 4),
                        "formula": "0.80 * denso + 0.20 * lexical",
                    }
                )


def main() -> None:
    """Construye la aplicacion Streamlit."""
    configure_page()
    render_sidebar()
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">PROCESAMIENTO DEL LENGUAJE NATURAL · UNI-IA</div>
          <h1>Riesgo tributario explicable</h1>
          <p>Clasificacion semantica del Registro de Compras con BETO y recuperacion
          trazable de sustento documentario mediante una arquitectura RAG.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Demo academica con datos sinteticos. No constituye asesoria tributaria ni "
        "reemplaza la revision profesional."
    )

    overview_tab, classifier_tab, rag_tab = st.tabs(
        ["Resumen del proyecto", "Clasificador BETO", "Auditoria RAG"]
    )
    with overview_tab:
        render_overview()
    with classifier_tab:
        render_classifier()
    with rag_tab:
        render_rag()


if __name__ == "__main__":
    main()
