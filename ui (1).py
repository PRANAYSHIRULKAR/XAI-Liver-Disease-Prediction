import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import pickle
import os
from datetime import datetime
from io import BytesIO
import requests
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Load environment variables from .env file
load_dotenv()

# ── Sklearn ──────────────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    BaggingClassifier, AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🫀 Liver Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    .risk-card {
        border-radius: 12px;
        padding: 22px 28px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .risk-high   { background: linear-gradient(135deg,#7f1010,#c0392b); color:#fff; border:2px solid #e74c3c; }
    .risk-medium { background: linear-gradient(135deg,#7f5800,#d68910); color:#fff; border:2px solid #f39c12; }
    .risk-low    { background: linear-gradient(135deg,#0a4f2b,#1e8449); color:#fff; border:2px solid #2ecc71; }

    .metric-box {
        background: #1e2130;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .metric-box .label { font-size: 0.8rem; color: #8899aa; text-transform: uppercase; letter-spacing: 1px; }
    .metric-box .value { font-size: 1.5rem; font-weight: 700; color: #e8eaf6; }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #7986cb;
        border-bottom: 2px solid #2d3250;
        padding-bottom: 6px;
        margin: 20px 0 14px 0;
    }
    .xai-legend {
        background: #1a1d2b;
        border-left: 4px solid #7986cb;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #b0bec5;
        margin-bottom: 12px;
    }
    .info-chip {
        display: inline-block;
        background: #1e2130;
        border: 1px solid #3d4466;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        color: #90a4ae;
        margin: 2px 3px;
    }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161924; }
    [data-testid="stSidebar"] .stMarkdown h2 { color: #7986cb; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
for key in ("model_trained", "results", "best_model", "best_name",
            "preprocessor", "X_raw", "X_train_df", "X_test_df",
            "y_train", "y_test", "transformed_feature_names",
            "shap_values", "sv_high", "base_value_high",
            "X_test_shap", "df_original"):
    if key not in st.session_state:
        st.session_state[key] = None


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def assign_risk(row):
    if row['Status'] == 'D':
        return 2
    elif row['Status'] == 'CL':
        return 1
    else:
        stage = row['Stage']
        if pd.isna(stage) or stage <= 2:
            return 0
        elif stage == 3:
            return 1
        else:
            return 2

RISK_LABELS  = ['Low', 'Medium', 'High']
RISK_EMOJI   = ['🟢', '🟡', '🔴']
RISK_COLOR   = ['#2ecc71', '#f39c12', '#e74c3c']


def format_value_for_report(value):
    if isinstance(value, float):
        return f"{value:.2f}".rstrip('0').rstrip('.')
    if pd.isna(value):
        return "Not available"
    return str(value)


def build_prediction_pdf(
    patient_data,
    best_name,
    risk_label,
    confidence_text,
    probabilities,
    feature_table,
    ai_explanation=None,
):
    if not HAS_REPORTLAB:
        raise ImportError("reportlab is not installed")

    risk_notes = {
        'Low': (
            'The model sees a comparatively reassuring pattern in the input values. '
            'This does not rule out liver disease, but it suggests fewer high-risk markers were detected.'
        ),
        'Medium': (
            'The model found a mixed profile. Some values are reassuring while others point toward '
            'possible liver stress, so follow-up review is advisable.'
        ),
        'High': (
            'The model detected a strong pattern associated with higher liver disease risk. '
            'This report should be reviewed by a qualified clinician as soon as possible.'
        ),
    }

    recommendation_notes = {
        'Low': [
            'Maintain routine monitoring and continue clinically appropriate follow-up.',
            'If symptoms appear or values change, repeat testing and reassessment may be useful.',
        ],
        'Medium': [
            'Discuss the findings with a clinician and consider repeating key liver tests.',
            'Watch for worsening symptoms such as jaundice, fatigue, swelling, or abdominal discomfort.',
        ],
        'High': [
            'Arrange timely medical review and correlate this result with laboratory and clinical findings.',
            'Seek urgent care if there are severe symptoms such as confusion, vomiting blood, or marked swelling.',
        ],
    }

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=rl_colors.HexColor('#12324d'),
        spaceAfter=10,
    )
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=rl_colors.HexColor('#4b5563'),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=rl_colors.HexColor('#12324d'),
        spaceAfter=6,
        spaceBefore=10,
    )
    body_style = ParagraphStyle(
        'BodyTextReport',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=rl_colors.HexColor('#1f2937'),
    )
    body_style.spaceAfter = 6

    story = []
    story.append(Paragraph('Liver Disease Risk Prediction Report', title_style))
    story.append(Paragraph(
        f'Generated on {datetime.now().strftime("%d %b %Y, %I:%M %p")} using {best_name}.',
        subtitle_style,
    ))
    story.append(Spacer(1, 5))

    story.append(Paragraph('Executive Summary', section_style))
    story.append(Paragraph(
        f"<b>Model outcome:</b> The selected model predicts <b>{risk_label} risk</b> of liver disease "
        f"with a confidence of <b>{confidence_text}</b>. {risk_notes[risk_label]}",
        body_style,
    ))
    story.append(Paragraph(
        f"<b>Plain-language interpretation:</b> This result is a decision-support output, not a diagnosis. "
        f"It summarizes how the model reads the submitted clinical values and highlights the strongest factors "
        f"that pushed the prediction toward this risk level.",
        body_style,
    ))

    patient_rows = [[Paragraph('<b>Field</b>', body_style), Paragraph('<b>Value</b>', body_style)]]
    for key, value in patient_data.items():
        patient_rows.append([
            Paragraph(str(key).replace('_', ' '), body_style),
            Paragraph(format_value_for_report(value), body_style),
        ])

    patient_table = Table(patient_rows, colWidths=[58 * mm, 98 * mm], repeatRows=1)
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#dbeafe')),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.HexColor('#12324d')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.8),
        ('LEADING', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.35, rl_colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.whitesmoke, rl_colors.HexColor('#f8fafc')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(Paragraph('Submitted Patient Details', section_style))
    story.append(patient_table)

    probability_rows = [[Paragraph('<b>Risk level</b>', body_style), Paragraph('<b>Probability</b>', body_style)]]
    for label, value in zip(RISK_LABELS, probabilities):
        probability_rows.append([
            Paragraph(label, body_style),
            Paragraph(f'{value * 100:.1f}%', body_style),
        ])

    probability_table = Table(probability_rows, colWidths=[60 * mm, 40 * mm], repeatRows=1)
    probability_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#ecfeff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.HexColor('#0f172a')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.35, rl_colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#f8fafc')]),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(Spacer(1, 4))
    story.append(Paragraph('Probability Breakdown', section_style))
    story.append(probability_table)

    story.append(Spacer(1, 4))
    story.append(Paragraph('Main Factors Influencing the Prediction', section_style))
    if feature_table is not None and not feature_table.empty and 'SHAP Value' in feature_table.columns:
        top_features = feature_table.copy()
        top_features['Abs SHAP'] = top_features['SHAP Value'].abs()
        top_features = top_features.sort_values('Abs SHAP', ascending=False).head(8)
        factor_rows = [[
            Paragraph('<b>Feature</b>', body_style),
            Paragraph('<b>Direction</b>', body_style),
            Paragraph('<b>SHAP value</b>', body_style),
        ]]
        for _, row in top_features.iterrows():
            factor_rows.append([
                Paragraph(str(row['Feature']), body_style),
                Paragraph(str(row['Direction']), body_style),
                Paragraph(f"{row['SHAP Value']:+.4f}", body_style),
            ])

        factor_table = Table(factor_rows, colWidths=[66 * mm, 50 * mm, 28 * mm], repeatRows=1)
        factor_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#fee2e2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.HexColor('#7f1d1d')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.35, rl_colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#f8fafc')]),
            ('FONTSIZE', (0, 0), (-1, -1), 8.7),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(Paragraph(
            'Positive SHAP values push the prediction toward higher risk, while negative values reduce the predicted risk.',
            body_style,
        ))
        story.append(factor_table)
    else:
        story.append(Paragraph(
            'A feature-level explanation was not available for this prediction, so the report focuses on the overall model result and patient inputs.',
            body_style,
        ))

    story.append(Spacer(1, 4))
    story.append(Paragraph('Clinical Interpretation', section_style))
    story.append(Paragraph(
        f"The model summary for this patient is: <b>{risk_label} risk</b> with <b>{confidence_text}</b>. "
        f"The strongest available model signal is best interpreted alongside the patient's history, examination, "
        f"and laboratory results. If symptoms or abnormal values are present, clinical review is recommended.",
        body_style,
    ))

    if ai_explanation:
        story.append(Paragraph('AI Generated Narrative', section_style))
        story.append(Paragraph(ai_explanation.replace('\n', '<br/>'), body_style))

    story.append(Paragraph('Recommended Next Steps', section_style))
    for note in recommendation_notes[risk_label]:
        story.append(Paragraph(f'• {note}', body_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        'Disclaimer: This report is generated automatically from model predictions and is intended for decision support only. '
        'It must not be treated as a medical diagnosis or a substitute for clinical judgment.',
        ParagraphStyle(
            'Disclaimer',
            parent=styles['Italic'],
            fontName='Helvetica-Oblique',
            fontSize=8.6,
            leading=11,
            textColor=rl_colors.HexColor('#6b7280'),
        ),
    ))

    def draw_page_number(canvas_obj, doc_obj):
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(rl_colors.HexColor('#6b7280'))
        canvas_obj.drawRightString(
            A4[0] - doc.rightMargin,
            10 * mm,
            f'Page {canvas_obj.getPageNumber()}',
        )

    doc.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)
    buffer.seek(0)
    return buffer.getvalue()

# ════════════════════════════════════════════════════════════════════════════
# MAIN APP LAYOUT
# ════════════════════════════════════════════════════════════════════════════
st.markdown("# 🫀 Liver Disease Risk Prediction")
st.markdown("""
<span class='info-chip'>Dataset: Cirrhosis Survival (418 patients × 20 features)</span>
<span class='info-chip'>XAI: SHAP TreeExplainer</span>
<span class='info-chip'>9 ML algorithms</span>
<span class='info-chip'>Risk: Low / Medium / High</span>
""", unsafe_allow_html=True)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Data Upload & Config
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    uploaded_file = st.file_uploader(
        "Upload `cirrhosis.xlsx`", type=["xlsx", "csv"],
        help="Upload the Cirrhosis dataset (xlsx or csv)"
    )

    st.markdown("### Model Selection")
    model_options = [
        "Random Forest", "Logistic Regression", "Naive Bayes",
        "Decision Tree", "Bagging", "Gradient Boosting",
        "AdaBoost", "GBC"
    ]
    if HAS_XGBOOST:
        model_options.append("XGBoost")

    selected_models = st.multiselect(
        "Choose algorithms to train",
        model_options,
        default=["Random Forest", "XGBoost"] if HAS_XGBOOST else ["Random Forest", "Gradient Boosting"]
    )
    test_size = st.slider("Test Split %", 10, 40, 20, 5)
    random_seed = st.number_input("Random Seed", value=42, step=1)

    st.markdown("### SHAP Settings")
    shap_sample = st.slider(
        "SHAP background samples", 20, 200, 50, 10,
        help="More = slower but more accurate SHAP values"
    )

    train_btn = st.button("🚀 Train Models", type="primary", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab_eda, tab_train, tab_predict, tab_xai, tab_guide = st.tabs([
    "📊 EDA",
    "🤖 Model Training",
    "🔬 Predict Risk",
    "🧠 XAI Explanations",
    "📖 XAI Guide"
])

# ════════════════════════════════════════════════════════════════════════════
# TRAINING LOGIC (triggered by button)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(file_content, file_name):
    file_name = str(file_name).lower()
    if file_name.endswith('.csv'):
        return pd.read_csv(BytesIO(file_content))
    if file_name.endswith('.xlsx'):
        return pd.read_excel(BytesIO(file_content), engine='openpyxl')
    raise ValueError("Unsupported file format. Please upload a .csv or .xlsx file.")

def build_model_candidates(selected, seed):
    candidates = {}
    if "Random Forest" in selected:
        candidates["Random Forest"] = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_split=8,
            min_samples_leaf=5, class_weight='balanced_subsample',
            random_state=seed, n_jobs=-1
        )
    if "Logistic Regression" in selected:
        candidates["Logistic Regression"] = LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=seed
        )
    if "Naive Bayes" in selected:
        candidates["Naive Bayes"] = GaussianNB()
    if "Decision Tree" in selected:
        candidates["Decision Tree"] = DecisionTreeClassifier(
            max_depth=6, class_weight='balanced', random_state=seed
        )
    if "Bagging" in selected:
        candidates["Bagging"] = BaggingClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        )
    if "Gradient Boosting" in selected:
        candidates["Gradient Boosting"] = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, random_state=seed
        )
    if "AdaBoost" in selected:
        candidates["AdaBoost"] = AdaBoostClassifier(
            n_estimators=200, random_state=seed
        )
    if "GBC" in selected:
        candidates["GBC"] = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed
        )
    if "XGBoost" in selected and HAS_XGBOOST:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=4,
            reg_alpha=0.1, reg_lambda=1.5, objective='multi:softprob',
            eval_metric='mlogloss', random_state=seed, n_jobs=-1
        )
    return candidates

if train_btn and uploaded_file:
    with st.spinner("Loading and preprocessing data…"):
        # Read bytes once and parse from memory to avoid stream pointer issues.
        file_content = uploaded_file.getvalue()
        df = load_data(file_content, uploaded_file.name)

        if df.empty:
            st.error("Uploaded file has no rows. Please upload a valid dataset.")
            st.stop()

        st.session_state.df_original = df
        st.info(f"Loaded {len(df)} rows × {len(df.columns)} columns from {uploaded_file.name}")

        # Target
        df_model = df.drop(columns=['ID','N_Days'], errors='ignore').copy()
        df_model['Risk_Level'] = df_model.apply(assign_risk, axis=1)

        X_raw = df_model.drop(columns=['Risk_Level','Status'])
        y = df_model['Risk_Level']
        st.session_state.X_raw = X_raw

        # Split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=test_size/100,
            random_state=int(random_seed), stratify=y
        )

        # Pipeline
        numeric_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]

        num_pipe = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ('num', num_pipe, numeric_cols),
            ('cat', cat_pipe, categorical_cols)
        ], remainder='drop', verbose_feature_names_out=True)

        X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_test_proc  = preprocessor.transform(X_test_raw)
        feat_names   = preprocessor.get_feature_names_out().tolist()

        X_train_df = pd.DataFrame(X_train_proc, columns=feat_names, index=X_train_raw.index)
        X_test_df  = pd.DataFrame(X_test_proc,  columns=feat_names, index=X_test_raw.index)

        st.session_state.preprocessor              = preprocessor
        st.session_state.transformed_feature_names = feat_names
        st.session_state.X_train_df               = X_train_df
        st.session_state.X_test_df                = X_test_df
        st.session_state.y_train                  = y_train
        st.session_state.y_test                   = y_test

    # Train models
    results = {}
    candidates = build_model_candidates(selected_models, int(random_seed))
    if not candidates:
        st.error("Please select at least one model to train.")
        st.stop()

    prog = st.progress(0)
    for i, (name, model) in enumerate(candidates.items()):
        with st.spinner(f"Training {name}…"):
            model.fit(X_train_proc, y_train)
            y_pred = model.predict(X_test_proc)
            results[name] = {
                'model': model,
                'preds': y_pred,
                'acc':  accuracy_score(y_test, y_pred),
                'f1':   f1_score(y_test, y_pred, average='weighted'),
                'prec': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'rec':  recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'cm':   confusion_matrix(y_test, y_pred),
            }
        prog.progress((i+1)/len(candidates))

    best_name  = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_name]['model']
    st.session_state.results    = results
    st.session_state.best_model = best_model
    st.session_state.best_name  = best_name
    st.session_state.model_trained = True

    # SHAP
    if HAS_SHAP:
        with st.spinner("Computing SHAP values (this may take ~30 s)…"):
            try:
                explainer = shap.TreeExplainer(
                    best_model,
                    data=shap.sample(X_train_df, shap_sample),
                    feature_perturbation='interventional'
                )
                sv = explainer.shap_values(X_test_df)
                if isinstance(sv, list):
                    sv_high = sv[2]
                    base_h  = explainer.expected_value[2]
                elif sv.ndim == 3:
                    sv_high = sv[:, :, 2]
                    base_h  = explainer.expected_value[2]
                else:
                    sv_high = sv
                    base_h  = explainer.expected_value
                st.session_state.sv_high        = sv_high
                st.session_state.base_value_high = base_h
                st.session_state.X_test_shap    = X_test_df
                st.session_state.shap_values    = sv
            except Exception as e:
                st.warning(f"SHAP computation failed: {e}")

    st.success(f"✅ Training complete! Best model: **{best_name}** (F1 = {results[best_name]['f1']:.4f})")

elif train_btn and not uploaded_file:
    st.warning("⬆️ Please upload the `cirrhosis.xlsx` dataset first.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if st.session_state.df_original is None:
        st.info("⬆️ Upload the dataset and click **Train Models** to see EDA.")
    else:
        df = st.session_state.df_original

        # Overview metrics
        c1,c2,c3,c4 = st.columns(4)
        for col, label, val in zip(
            [c1,c2,c3,c4],
            ["Patients","Features","Missing Cells","Stages"],
            [df.shape[0], df.shape[1]-1,
             int(df.isnull().sum().sum()),
             int(df['Stage'].nunique()) if 'Stage' in df.columns else "—"]
        ):
            col.markdown(f"""
            <div class='metric-box'>
                <div class='label'>{label}</div>
                <div class='value'>{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### Dataset Preview")
        st.caption(f"Total rows loaded: {len(df)}")
        show_all_rows = st.checkbox("Show all rows", value=False)
        preview_rows = len(df) if show_all_rows else min(50, len(df))
        st.dataframe(df.head(preview_rows), use_container_width=True)
        if not show_all_rows and len(df) > preview_rows:
            st.caption(f"Showing first {preview_rows} rows. Enable 'Show all rows' to view full dataset.")

        # Plots
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.patch.set_facecolor('#0f1117')
        for ax in axes.flat:
            ax.set_facecolor('#1e2130')
            ax.tick_params(colors='#b0bec5')
            for spine in ax.spines.values():
                spine.set_edgecolor('#2d3250')

        # 1) Status distribution
        ax = axes[0,0]
        vc = df['Status'].value_counts()
        colors = ['#2ecc71','#e74c3c','#f39c12']
        bars = ax.bar(vc.index, vc.values, color=colors[:len(vc)], edgecolor='#0f1117', linewidth=1.2)
        ax.set_title('Status Distribution', color='#e8eaf6', fontweight='bold')
        ax.set_xlabel('Status (C=Alive, D=Dead, CL=Transplant)', color='#90a4ae', fontsize=8)
        ax.set_ylabel('Count', color='#90a4ae')
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                    str(int(b.get_height())), ha='center', color='#e8eaf6', fontsize=9)

        # 2) Stage distribution
        ax = axes[0,1]
        if 'Stage' in df.columns:
            vc2 = df['Stage'].value_counts().sort_index()
            ax.bar(vc2.index.astype(str), vc2.values,
                   color=['#3498db','#9b59b6','#e74c3c','#e67e22'], edgecolor='#0f1117')
        ax.set_title('Stage Distribution', color='#e8eaf6', fontweight='bold')
        ax.set_xlabel('Stage', color='#90a4ae')
        ax.set_ylabel('Count', color='#90a4ae')

        # 3) Bilirubin by Status
        ax = axes[0,2]
        for status, color in zip(['C','D','CL'],['#2ecc71','#e74c3c','#f39c12']):
            sub = df[df['Status']==status]['Bilirubin'].dropna()
            ax.hist(sub, bins=20, alpha=0.65, label=status, color=color)
        ax.set_title('Bilirubin by Status', color='#e8eaf6', fontweight='bold')
        ax.set_xlabel('Bilirubin', color='#90a4ae')
        ax.legend(facecolor='#1e2130', edgecolor='#2d3250', labelcolor='#e8eaf6')

        # 4) Age distribution
        ax = axes[1,0]
        ax.hist(df['Age']/365, bins=25, color='#5c6bc0', edgecolor='#0f1117', alpha=0.9)
        ax.set_title('Age Distribution (years)', color='#e8eaf6', fontweight='bold')
        ax.set_xlabel('Age (years)', color='#90a4ae')

        # 5) Missing values
        ax = axes[1,1]
        mp = (df.isnull().mean()*100).sort_values(ascending=False)
        mp = mp[mp>0]
        if not mp.empty:
            ax.barh(mp.index, mp.values, color='#e74c3c', edgecolor='#0f1117')
            ax.set_title('Missing Values (%)', color='#e8eaf6', fontweight='bold')
            ax.set_xlabel('Missing %', color='#90a4ae')
        else:
            ax.text(0.5,0.5,'No missing values!',ha='center',va='center',color='#2ecc71',fontsize=12)
            ax.set_axis_off()

        # 6) Correlation heatmap
        ax = axes[1,2]
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr = df[num_cols].corr()
        sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=False,
                    linewidths=0.2, xticklabels=True, yticklabels=True,
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Heatmap', color='#e8eaf6', fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=6, colors='#b0bec5')
        ax.tick_params(axis='y', labelsize=6, colors='#b0bec5')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Risk distribution
        if st.session_state.X_raw is not None:
            df_model = st.session_state.df_original.drop(columns=['ID','N_Days'], errors='ignore').copy()
            df_model['Risk_Level'] = df_model.apply(assign_risk, axis=1)
            st.markdown("### Risk Label Distribution")
            vc_risk = df_model['Risk_Level'].value_counts().sort_index()
            fig2, ax2 = plt.subplots(figsize=(5,3))
            fig2.patch.set_facecolor('#0f1117')
            ax2.set_facecolor('#1e2130')
            bars = ax2.bar(
                [f"{RISK_EMOJI[i]} {RISK_LABELS[i]}" for i in vc_risk.index],
                vc_risk.values,
                color=[RISK_COLOR[i] for i in vc_risk.index],
                edgecolor='#0f1117'
            )
            for b in bars:
                ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                         str(int(b.get_height())), ha='center', color='#e8eaf6', fontsize=10)
            ax2.tick_params(colors='#b0bec5')
            for spine in ax2.spines.values(): spine.set_edgecolor('#2d3250')
            ax2.set_ylabel('Count', color='#90a4ae')
            ax2.set_title('Risk Label Distribution', color='#e8eaf6', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown('<div class="section-header">🤖 Model Performance Comparison</div>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.info("Upload the dataset and click **🚀 Train Models** in the sidebar.")
    else:
        results = st.session_state.results
        best_name = st.session_state.best_name

        # Summary table
        comp_df = pd.DataFrame({
            name: {
                'Accuracy': vals['acc'],
                'F1 (Weighted)': vals['f1'],
                'Precision': vals['prec'],
                'Recall': vals['rec']
            }
            for name, vals in results.items()
        }).T.sort_values('F1 (Weighted)', ascending=False)

        st.markdown(f"### 🏆 Best Model: **{best_name}** (F1 = {results[best_name]['f1']:.4f})")
        st.dataframe(
            comp_df.style.format("{:.4f}")
                   .background_gradient(cmap='RdYlGn', subset=['F1 (Weighted)'])
                   .highlight_max(subset=['Accuracy','F1 (Weighted)','Precision','Recall'], color='#1a4a1a'),
            use_container_width=True
        )

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(max(8, len(comp_df)*1.2), 5))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1e2130')
        x = np.arange(len(comp_df))
        w = 0.2
        ax.bar(x-1.5*w, comp_df['Accuracy'],         width=w, label='Accuracy',   color='#3498db')
        ax.bar(x-0.5*w, comp_df['F1 (Weighted)'],    width=w, label='F1',         color='#2ecc71')
        ax.bar(x+0.5*w, comp_df['Precision'],         width=w, label='Precision',  color='#f39c12')
        ax.bar(x+1.5*w, comp_df['Recall'],            width=w, label='Recall',     color='#e74c3c')
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df.index, rotation=15, ha='right', color='#b0bec5')
        ax.tick_params(axis='y', colors='#b0bec5')
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel('Score', color='#90a4ae')
        ax.set_title('Model Comparison (all metrics)', color='#e8eaf6', fontweight='bold')
        ax.legend(facecolor='#1e2130', edgecolor='#2d3250', labelcolor='#e8eaf6', ncol=4, loc='upper center')
        for spine in ax.spines.values(): spine.set_edgecolor('#2d3250')
        ax.axhline(1.0, color='#2d3250', linewidth=0.5, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Confusion matrices
        st.markdown("### Confusion Matrices")
        n = len(results)
        cols_per_row = 3
        rows_needed  = -(-n // cols_per_row)
        result_items = list(results.items())
        for row_i in range(rows_needed):
            row_cols = st.columns(min(cols_per_row, n - row_i*cols_per_row))
            for col_j, col in enumerate(row_cols):
                idx = row_i*cols_per_row + col_j
                if idx >= n: break
                name, vals = result_items[idx]
                fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
                fig_cm.patch.set_facecolor('#0f1117')
                ax_cm.set_facecolor('#1e2130')
                sns.heatmap(
                    vals['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low','Med','High'],
                    yticklabels=['Low','Med','High'],
                    ax=ax_cm, cbar=False,
                    annot_kws={'size': 10, 'color': '#e8eaf6'}
                )
                badge = "🏆 " if name == best_name else ""
                ax_cm.set_title(f"{badge}{name}\nF1={vals['f1']:.3f}", color='#e8eaf6', fontsize=9)
                ax_cm.set_xlabel('Predicted', color='#90a4ae', fontsize=8)
                ax_cm.set_ylabel('Actual', color='#90a4ae', fontsize=8)
                ax_cm.tick_params(colors='#b0bec5', labelsize=8)
                plt.tight_layout()
                col.pyplot(fig_cm)
                plt.close()

        # Feature importance
        best_model = st.session_state.best_model
        if hasattr(best_model, 'feature_importances_'):
            st.markdown("### 🔎 Feature Importances (Best Model)")
            feat_names = st.session_state.transformed_feature_names
            imp_df = pd.DataFrame({
                'Feature': feat_names,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)

            fig_fi, ax_fi = plt.subplots(figsize=(10,5))
            fig_fi.patch.set_facecolor('#0f1117')
            ax_fi.set_facecolor('#1e2130')
            top = imp_df.sort_values('Importance')
            cmap_vals = top['Importance'].values
            norm = plt.Normalize(cmap_vals.min(), cmap_vals.max())
            colors_fi = plt.cm.YlOrRd(norm(cmap_vals))
            ax_fi.barh(top['Feature'], top['Importance'], color=colors_fi, edgecolor='#0f1117')
            ax_fi.set_title(f'Top 15 Feature Importances — {best_name}', color='#e8eaf6', fontweight='bold')
            ax_fi.set_xlabel('Importance', color='#90a4ae')
            ax_fi.tick_params(colors='#b0bec5', labelsize=8)
            for spine in ax_fi.spines.values(): spine.set_edgecolor('#2d3250')
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close()


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-header">🔬 Patient Risk Prediction</div>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.info("Train models first via the sidebar.")
    else:
        best_model   = st.session_state.best_model
        best_name    = st.session_state.best_name
        preprocessor = st.session_state.preprocessor
        X_raw        = st.session_state.X_raw

        st.markdown(f"**Active model:** `{best_name}` &nbsp;|&nbsp; **Features:** `{len(X_raw.columns)} raw`")

        with st.form("patient_form"):
            st.markdown("#### Clinical & Laboratory Values")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Categorical Features**")
                drug      = st.selectbox("Drug", ['D-penicillamine','Placebo'])
                sex       = st.selectbox("Sex", ['F','M'])
                ascites   = st.selectbox("Ascites", ['N','Y'])
                hepato    = st.selectbox("Hepatomegaly", ['N','Y'])
                spiders   = st.selectbox("Spiders", ['N','Y'])
                edema     = st.selectbox("Edema", ['N','S','Y'])
            with c2:
                st.markdown("**Numeric Features (Part 1)**")
                age         = st.number_input("Age (years)", 18, 90, 50) * 365
                bilirubin   = st.number_input("Bilirubin (mg/dL)", 0.1, 30.0, 2.0, 0.1)
                cholesterol = st.number_input("Cholesterol (mg/dL)", 50.0, 1000.0, 250.0, 10.0)
                albumin     = st.number_input("Albumin (gm/dL)", 1.0, 6.0, 3.5, 0.1)
                copper      = st.number_input("Copper (ug/day)", 0.0, 600.0, 80.0, 5.0)
            with c3:
                st.markdown("**Numeric Features (Part 2)**")
                alk_phos    = st.number_input("Alk Phos (U/L)", 100.0, 5000.0, 1000.0, 50.0)
                sgot        = st.number_input("SGOT (U/ml)", 10.0, 500.0, 100.0, 5.0)
                tryg        = st.number_input("Tryglicerides (mg/dL)", 10.0, 600.0, 120.0, 5.0)
                platelets   = st.number_input("Platelets (cu ml/1000)", 20.0, 700.0, 250.0, 10.0)
                prothrombin = st.number_input("Prothrombin (sec)", 8.0, 20.0, 10.5, 0.1)
                stage       = st.selectbox("Stage", [1,2,3,4], index=1)

            submitted = st.form_submit_button("🔮 Predict Risk", type="primary", use_container_width=True)

        if submitted:
            patient_data = {
                'Drug': drug, 'Sex': sex, 'Ascites': ascites,
                'Hepatomegaly': hepato, 'Spiders': spiders, 'Edema': edema,
                'Age': age, 'Bilirubin': bilirubin, 'Cholesterol': cholesterol,
                'Albumin': albumin, 'Copper': copper, 'Alk_Phos': alk_phos,
                'SGOT': sgot, 'Tryglicerides': tryg, 'Platelets': platelets,
                'Prothrombin': prothrombin, 'Stage': float(stage)
            }
            raw_df = pd.DataFrame([patient_data])
            for col in X_raw.columns:
                if col not in raw_df.columns:
                    raw_df[col] = np.nan
            raw_df = raw_df[X_raw.columns]

            proc  = preprocessor.transform(raw_df)
            pred  = best_model.predict(proc)[0]
            prob  = best_model.predict_proba(proc)[0]

            rl  = RISK_LABELS[pred]
            rem = RISK_EMOJI[pred]
            conf = f"{prob[pred]*100:.1f}%"
            ai_explanation = None
            wf_df = None

            css_cls = {'Low':'risk-low','Medium':'risk-medium','High':'risk-high'}[rl]
            st.markdown(f"""
            <div class='risk-card {css_cls}'>
                {rem} {rl} Risk &nbsp;|&nbsp; Confidence: {conf}
            </div>
            """, unsafe_allow_html=True)

            c1,c2,c3 = st.columns(3)
            for col, label, emoji, val in zip(
                [c1,c2,c3],
                ['Low Risk','Medium Risk','High Risk'],
                RISK_EMOJI,
                [f"{p*100:.1f}%" for p in prob]
            ):
                col.markdown(f"""
                <div class='metric-box'>
                    <div class='label'>{emoji} {label}</div>
                    <div class='value'>{val}</div>
                </div>""", unsafe_allow_html=True)

            # Per-patient SHAP waterfall
            if HAS_SHAP and st.session_state.sv_high is not None:
                st.markdown("#### 💬 Why this prediction? (SHAP Waterfall)")
                try:
                    feat_names = st.session_state.transformed_feature_names
                    sv_patient  = st.session_state.shap_values
                    base_h      = st.session_state.base_value_high

                    explainer_local = shap.TreeExplainer(best_model)
                    proc_df = pd.DataFrame(proc, columns=feat_names)
                    sv_local = explainer_local.shap_values(proc_df)

                    if isinstance(sv_local, list):
                        sv_p  = sv_local[2][0]
                        base_p = explainer_local.expected_value[2]
                    elif sv_local.ndim == 3:
                        sv_p  = sv_local[0, :, 2]
                        base_p = explainer_local.expected_value[2]
                    else:
                        sv_p  = sv_local[0]
                        base_p = explainer_local.expected_value

                    exp = shap.Explanation(
                        values=sv_p,
                        base_values=base_p,
                        data=proc_df.iloc[0].values,
                        feature_names=feat_names
                    )
                    fig_wf, ax_wf = plt.subplots(figsize=(10,6))
                    fig_wf.patch.set_facecolor('#0f1117')
                    shap.waterfall_plot(exp, max_display=15, show=False)
                    plt.tight_layout()
                    st.pyplot(fig_wf)
                    plt.close()

                    # Table
                    wf_df = pd.DataFrame({
                        'Feature': feat_names,
                        'SHAP Value': sv_p,
                        'Direction': np.where(sv_p > 0, '⬆️ Raises Risk', '⬇️ Lowers Risk')
                    }).sort_values('SHAP Value', key=abs, ascending=False).head(12)
                    st.dataframe(wf_df, use_container_width=True)

                    # --- Mistral AI Explanation ---
                    st.markdown("#### 🤖 AI Personalized Explanation")
                    api_key = os.getenv("MISTRAL_API_KEY")
                    if not api_key or api_key == "your_mistral_api_key_here":
                        st.warning("⚠️ Mistral API Key not found. Please add a valid `MISTRAL_API_KEY` to the `.env` file to see the AI explanation.")
                    else:
                        with st.spinner("Generating personalized explanation from Mistral AI..."):
                            try:
                                url = "https://api.mistral.ai/v1/chat/completions"
                                headers = {
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {api_key}"
                                }
                                
                                # Format patient data
                                pat_str = ", ".join([f"{k}: {v}" for k, v in patient_data.items()])
                                
                                # Format SHAP data
                                shap_str = ", ".join([f"{row['Feature']} ({row['Direction']})" for _, row in wf_df.iterrows()])
                                
                                prompt = (
                                    f"You are a medical AI assistant. A patient was predicted to have a **{rl} Risk** of liver disease.\n"
                                    f"Here are their clinical details: {pat_str}.\n"
                                    f"Here are the most important factors contributing to this prediction based on SHAP values (in order of importance): {shap_str}.\n"
                                    "Please provide a brief, easy-to-understand explanation for the patient about what these factors mean, "
                                    "what the main problem is based on their inputs, and what they should focus on. "
                                    "Keep it compassionate and informative, and clearly state that this is an AI explanation and they should consult a doctor."
                                )
                                
                                data = {
                                    "model": "mistral-small-latest",
                                    "messages": [{"role": "user", "content": prompt}],
                                    "temperature": 0.3
                                }
                                
                                response = requests.post(url, headers=headers, json=data)
                                if response.status_code == 200:
                                    ai_explanation = response.json()['choices'][0]['message']['content']
                                    st.info(ai_explanation)
                                else:
                                    st.error(f"Failed to generate explanation. API returned status code {response.status_code}: {response.text}")
                            except Exception as e:
                                st.error(f"Error communicating with Mistral AI: {e}")

                except Exception as e:
                    st.info(f"Local SHAP waterfall unavailable: {e}")

            if HAS_REPORTLAB:
                try:
                    report_pdf = build_prediction_pdf(
                        patient_data=patient_data,
                        best_name=best_name,
                        risk_label=rl,
                        confidence_text=conf,
                        probabilities=prob,
                        feature_table=wf_df,
                        ai_explanation=ai_explanation,
                    )
                    file_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    safe_model_name = best_name.lower().replace(' ', '_')
                    st.markdown('#### 📄 Detailed PDF Report')
                    st.caption('Download a polished report with the model result, plain-language interpretation, patient inputs, and feature-level explanation.')
                    st.download_button(
                        label='⬇️ Download PDF Report',
                        data=report_pdf,
                        file_name=f'liver_risk_report_{safe_model_name}_{file_stamp}.pdf',
                        mime='application/pdf',
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f'Could not generate the PDF report: {e}')
            else:
                st.warning('PDF reports require the `reportlab` package. Add it to the environment to enable downloads.')


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — XAI EXPLANATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_xai:
    st.markdown('<div class="section-header">🧠 Explainable AI — SHAP Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.info("Train models first.")
    elif not HAS_SHAP:
        st.error("SHAP is not installed. Run `pip install shap`.")
    elif st.session_state.sv_high is None:
        st.warning("SHAP values were not computed. Re-train and ensure SHAP is installed.")
    else:
        sv_high    = st.session_state.sv_high
        X_test_sh  = st.session_state.X_test_shap
        feat_names = st.session_state.transformed_feature_names
        base_h     = st.session_state.base_value_high
        best_name  = st.session_state.best_name
        y_test     = st.session_state.y_test

        # ── XAI 1: Global Feature Importance ─────────────────────────────
        st.markdown("### 📊 Plot 1 — Global SHAP Feature Importance (Bar)")
        st.markdown("""
        <div class='xai-legend'>
        <b>What it shows:</b> Average absolute SHAP value per feature across ALL test patients.<br>
        <b>How to read:</b> Longer bars = more influential features overall. This is the model's "global attention".<br>
        <b>Clinical insight:</b> Features like Bilirubin, Stage, Albumin typically dominate — they are strong biochemical markers of liver dysfunction.
        </div>
        """, unsafe_allow_html=True)

        mean_abs = np.abs(sv_high).mean(axis=0)
        imp_df_shap = pd.DataFrame({'Feature': feat_names, 'Mean |SHAP|': mean_abs})\
                        .sort_values('Mean |SHAP|', ascending=False).head(20)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        fig1.patch.set_facecolor('#0f1117')
        ax1.set_facecolor('#1e2130')
        top20 = imp_df_shap.sort_values('Mean |SHAP|')
        norm1 = plt.Normalize(top20['Mean |SHAP|'].min(), top20['Mean |SHAP|'].max())
        colors1 = plt.cm.plasma(norm1(top20['Mean |SHAP|'].values))
        ax1.barh(top20['Feature'], top20['Mean |SHAP|'], color=colors1, edgecolor='#0f1117')
        ax1.set_title(f'Global SHAP Importance — {best_name} (High Risk class)',
                      color='#e8eaf6', fontweight='bold')
        ax1.set_xlabel('Mean |SHAP Value|', color='#90a4ae')
        ax1.tick_params(colors='#b0bec5', labelsize=8)
        for spine in ax1.spines.values(): spine.set_edgecolor('#2d3250')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        # ── XAI 2: Beeswarm / Summary Plot ───────────────────────────────
        st.markdown("### 🌊 Plot 2 — SHAP Beeswarm (Summary) Plot")
        st.markdown("""
        <div class='xai-legend'>
        <b>What it shows:</b> Every dot = one patient. X-axis = SHAP value (impact on High Risk prediction).<br>
        <b>Color:</b> 🔴 Red = high feature value, 🔵 Blue = low feature value.<br>
        <b>How to read:</b> Red dots on the <i>right</i> → high feature values push toward High Risk. Blue dots on the <i>right</i> → low feature values push toward High Risk.<br>
        <b>Clinical insight:</b> High Bilirubin (red, right) = strong risk driver. High Albumin (blue, left) = protective.
        </div>
        """, unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(10, 7))
        fig2.patch.set_facecolor('#0f1117')
        shap.summary_plot(
            sv_high, X_test_sh,
            feature_names=feat_names,
            plot_type='dot',
            max_display=15,
            show=False,
            color_bar_label='Feature Value'
        )
        plt.gcf().patch.set_facecolor('#0f1117')
        plt.title(f'SHAP Beeswarm — High Risk Class ({best_name})',
                  color='#e8eaf6', fontweight='bold', pad=12)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close('all')

        # ── XAI 3: Waterfall Plot ─────────────────────────────────────────
        st.markdown("### 🍩 Plot 3 — SHAP Waterfall Plot (Single Patient)")
        st.markdown("""
        <div class='xai-legend'>
        <b>What it shows:</b> Step-by-step breakdown of ONE patient's prediction. Starting from the base (average) value, each feature pushes the score up or down.<br>
        <b>Red bars:</b> Push toward Higher Risk. &nbsp; <b>Blue bars:</b> Push toward Lower Risk.<br>
        <b>How to read:</b> Final prediction = base value + sum of all bars.<br>
        <b>Use case:</b> "Why was THIS patient predicted as High Risk?" — you can trace each contributing factor.
        </div>
        """, unsafe_allow_html=True)

        n_patients = len(X_test_sh)
        patient_idx = st.slider("Select test patient index", 0, n_patients-1, 0)
        true_label = RISK_LABELS[int(y_test.iloc[patient_idx])]

        exp_wf = shap.Explanation(
            values=sv_high[patient_idx],
            base_values=base_h,
            data=X_test_sh.iloc[patient_idx].values,
            feature_names=feat_names,
        )
        fig3 = plt.figure(figsize=(10, 7))
        fig3.patch.set_facecolor('#0f1117')
        shap.waterfall_plot(exp_wf, max_display=18, show=False)
        plt.title(f'Waterfall — Patient #{patient_idx} | True Risk: {true_label}',
                  color='#e8eaf6', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # SHAP table for selected patient
        wf_table = pd.DataFrame({
            'Feature': feat_names,
            'SHAP Value': sv_high[patient_idx],
            'Direction': np.where(sv_high[patient_idx] > 0, '⬆️ Raises Risk', '⬇️ Lowers Risk'),
            'Magnitude': np.abs(sv_high[patient_idx])
        }).sort_values('Magnitude', ascending=False).drop(columns='Magnitude').head(15)
        with st.expander("📋 SHAP values table for this patient"):
            st.dataframe(wf_table, use_container_width=True)

        # ── XAI 4: Dependence Plots ───────────────────────────────────────
        st.markdown("### 📉 Plot 4 — SHAP Dependence Plots")
        st.markdown("""
        <div class='xai-legend'>
        <b>What it shows:</b> Relationship between a feature's value (X-axis) and its SHAP impact (Y-axis) for each patient.<br>
        <b>Color:</b> Feature value (warm = high, cool = low). Dots above 0 = pushes High Risk up.<br>
        <b>How to read:</b> A rising curve → higher values of this feature → more High Risk. A falling curve → higher values → more Low Risk (protective).<br>
        <b>Clinical insight:</b> Bilirubin shows a sharp threshold effect. Albumin typically shows an inverse relationship (higher = protective).
        </div>
        """, unsafe_allow_html=True)

        dep_features = []
        for candidate in ['num__Bilirubin', 'num__Albumin', 'num__Stage', 'num__Prothrombin']:
            if candidate in feat_names:
                dep_features.append(candidate)
        dep_features = dep_features[:4]

        if dep_features:
            ncols = min(2, len(dep_features))
            nrows = -(-len(dep_features) // ncols)
            fig4, axes4 = plt.subplots(nrows, ncols, figsize=(12, 4.5*nrows))
            fig4.patch.set_facecolor('#0f1117')
            axes4_flat = axes4.flat if hasattr(axes4, 'flat') else [axes4]

            for ax4, feat in zip(axes4_flat, dep_features):
                ax4.set_facecolor('#1e2130')
                fi = feat_names.index(feat)
                x_vals = X_test_sh[feat].values
                y_vals = sv_high[:, fi]
                sc = ax4.scatter(x_vals, y_vals, c=x_vals, cmap='RdYlGn_r',
                                 alpha=0.7, edgecolors='white', linewidth=0.2, s=40)
                fig4.colorbar(sc, ax=ax4, label=f'{feat} value', shrink=0.8)
                ax4.axhline(0, color='#ffffff', linewidth=0.7, linestyle='--', alpha=0.5)
                ax4.set_xlabel(feat, color='#90a4ae', fontsize=9)
                ax4.set_ylabel('SHAP Value → High Risk', color='#90a4ae', fontsize=9)
                ax4.set_title(feat, color='#e8eaf6', fontweight='bold')
                ax4.tick_params(colors='#b0bec5', labelsize=8)
                for spine in ax4.spines.values(): spine.set_edgecolor('#2d3250')

            # Hide unused subplots
            for ax_extra in list(axes4_flat)[len(dep_features):]:
                ax_extra.set_visible(False)

            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        # ── XAI 5: Force Plot (static) ────────────────────────────────────
        st.markdown("### ⚡ Plot 5 — SHAP Force Plot (Static)")
        st.markdown("""
        <div class='xai-legend'>
        <b>What it shows:</b> A horizontal tug-of-war for ONE patient. Features pushing the prediction higher (red) vs lower (blue).<br>
        <b>Base value:</b> The average model output across training data.<br>
        <b>Final value (f(x)):</b> The model's output for this patient = base + all pushes.<br>
        <b>Use case:</b> At-a-glance visual for explaining a single prediction to clinicians.
        </div>
        """, unsafe_allow_html=True)

        try:
            fig5, ax5 = plt.subplots(figsize=(12, 2.5))
            fig5.patch.set_facecolor('#0f1117')
            ax5.set_facecolor('#1e2130')

            sv_p = sv_high[patient_idx]
            top_n = 8
            abs_idx = np.argsort(np.abs(sv_p))[::-1][:top_n]
            sorted_idx = sorted(abs_idx, key=lambda i: sv_p[i])

            left = base_h
            segments = []
            for i in sorted_idx:
                val = sv_p[i]
                color = '#e74c3c' if val > 0 else '#3498db'
                segments.append((left, val, color, feat_names[i]))
                left += val

            for (start, width, color, fname) in segments:
                ax5.barh(0, width, left=start, color=color, alpha=0.85,
                         height=0.6, edgecolor='#0f1117', linewidth=0.8)
                mid = start + width/2
                if abs(width) > 0.02:
                    ax5.text(mid, 0, fname.replace('num__','').replace('cat__','')[:12],
                             ha='center', va='center', fontsize=7, color='white', fontweight='bold')

            ax5.axvline(base_h, color='#ffffff', linewidth=1.2, linestyle='--', alpha=0.6, label=f'Base={base_h:.2f}')
            ax5.axvline(left, color='#f39c12', linewidth=2, label=f'f(x)={left:.2f}')
            ax5.set_yticks([])
            ax5.set_xlabel('Model Output (High Risk logit/score)', color='#90a4ae')
            ax5.set_title(f'Force Plot — Patient #{patient_idx} | True: {true_label}',
                          color='#e8eaf6', fontweight='bold')
            ax5.legend(facecolor='#1e2130', edgecolor='#2d3250', labelcolor='#e8eaf6', fontsize=8)
            ax5.tick_params(colors='#b0bec5')
            for spine in ax5.spines.values(): spine.set_edgecolor('#2d3250')
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close()
        except Exception as e:
            st.info(f"Force plot error: {e}")

        # ── XAI 6: Global Signed Importance ──────────────────────────────
        st.markdown("### 🎯 Plot 6 — Signed Mean SHAP (Risk Drivers vs Protectors)")
        st.markdown("""
        <div class='xai-legend'>
        <b>What it shows:</b> Which features systematically raise vs lower High Risk predictions across all patients.<br>
        <b>Positive (red):</b> These features on average PUSH patients toward High Risk.<br>
        <b>Negative (blue):</b> These features on average PROTECT against High Risk.<br>
        <b>Use case:</b> Target interventions — improving albumin or reducing bilirubin has the most leverage.
        </div>
        """, unsafe_allow_html=True)

        signed_mean = sv_high.mean(axis=0)
        sign_df = pd.DataFrame({'Feature': feat_names, 'Mean SHAP': signed_mean})\
                    .sort_values('Mean SHAP', ascending=True).head(20)

        fig6, ax6 = plt.subplots(figsize=(10, 6))
        fig6.patch.set_facecolor('#0f1117')
        ax6.set_facecolor('#1e2130')
        colors6 = ['#e74c3c' if v > 0 else '#3498db' for v in sign_df['Mean SHAP']]
        ax6.barh(sign_df['Feature'], sign_df['Mean SHAP'], color=colors6, edgecolor='#0f1117')
        ax6.axvline(0, color='white', linewidth=1.2, linestyle='--', alpha=0.6)
        ax6.set_title('Signed Mean SHAP — Risk Drivers vs Protectors',
                      color='#e8eaf6', fontweight='bold')
        ax6.set_xlabel('Mean SHAP Value (→ High Risk)', color='#90a4ae')
        ax6.tick_params(colors='#b0bec5', labelsize=8)
        for spine in ax6.spines.values(): spine.set_edgecolor('#2d3250')
        red_p = mpatches.Patch(color='#e74c3c', label='Risk Driver (→ High Risk)')
        blue_p = mpatches.Patch(color='#3498db', label='Risk Protector (→ Low Risk)')
        ax6.legend(handles=[red_p, blue_p], facecolor='#1e2130',
                   edgecolor='#2d3250', labelcolor='#e8eaf6')
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — XAI GUIDE
# ════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.markdown('<div class="section-header">📖 XAI Explanation Guide</div>', unsafe_allow_html=True)

    st.markdown("""
## What is SHAP?

**SHAP (SHapley Additive exPlanations)** is a unified framework for interpreting machine learning model predictions.
It assigns an *importance value* (SHAP value) to each feature for every individual prediction.

SHAP values are derived from **cooperative game theory** (Shapley values), guaranteeing:
- **Consistency** — a model that relies more on a feature will always give it a higher SHAP value
- **Local accuracy** — the sum of all SHAP values equals the prediction
- **Missingness** — features with no influence get zero SHAP value

---

## The 6 XAI Plots in This App

| Plot | Type | Scope | Best for |
|------|------|-------|----------|
| **1. Global Feature Importance (Bar)** | Bar chart | All patients | Which features matter overall |
| **2. Beeswarm / Summary** | Dot plot | All patients | Direction + magnitude across patients |
| **3. Waterfall** | Waterfall | Single patient | Step-by-step explanation of one prediction |
| **4. Dependence Plots** | Scatter | All patients | Non-linear feature → risk relationship |
| **5. Force Plot** | Horizontal bar | Single patient | Visual tug-of-war for one prediction |
| **6. Signed Mean SHAP** | Signed bar | All patients | Global risk drivers vs protectors |

---

## Clinical Interpretation Guide

### Key Features & What They Mean

| Feature | High Value Effect | Clinical Meaning |
|---------|------------------|-----------------|
| **Bilirubin** | ⬆️ Raises Risk | Liver cannot process bilirubin → severe dysfunction |
| **Stage** | ⬆️ Raises Risk | Advanced cirrhosis stage → worse prognosis |
| **Albumin** | ⬇️ Lowers Risk | Liver produces albumin → higher = better function |
| **Prothrombin** | ⬆️ Raises Risk | Longer clotting time → worse liver synthesis |
| **Copper** | ⬆️ Raises Risk | Elevated in Wilson's disease and severe cirrhosis |
| **Platelets** | ⬇️ Lowers Risk | Low platelets = portal hypertension complication |
| **Alk Phos** | ⬆️ Raises Risk | Enzyme elevated in cholestatic liver disease |
| **SGOT** | ⬆️ Raises Risk | Liver enzyme, elevated in active injury |

---

## Reading SHAP Values: Key Rules

```
SHAP Value > 0   →   Feature pushes prediction TOWARD HIGH RISK
SHAP Value < 0   →   Feature pushes prediction TOWARD LOW RISK
SHAP Value = 0   →   Feature has NO EFFECT on this prediction

Final Score = Base Value (average) + Σ(all SHAP values)
```

---

## How the Risk Labels Are Assigned

```
HIGH RISK   🔴  →  Patient died (Status = D)
                    OR alive at Stage 4

MEDIUM RISK 🟡  →  Liver transplant (Status = CL)
                    OR alive at Stage 3

LOW RISK    🟢  →  Alive (Status = C) at Stage 1 or 2
```

> **Note:** `Status` is excluded from model inputs to prevent data leakage.
> Risk is only used as the target label derived from Status + Stage.

---

## Preprocessing Pipeline

```
Raw Data (418 patients × 17 input features)
         │
         ▼
  ┌──────────────────────────────────┐
  │  Numeric Features                │
  │  → SimpleImputer (median)        │
  │  → StandardScaler                │
  └──────────────────────────────────┘
         │
  ┌──────────────────────────────────┐
  │  Categorical Features            │
  │  → SimpleImputer (most_frequent) │
  │  → OneHotEncoder                 │
  └──────────────────────────────────┘
         │
         ▼
  Transformed Feature Matrix
  → Train 9 ML Models
  → Select Best by F1 Score
  → Compute SHAP Values
```

---
*Built with Streamlit · scikit-learn · XGBoost · SHAP*
    """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:0.8rem;'>
🫀 Liver Disease Risk Predictor &nbsp;|&nbsp; Ensemble ML + SHAP XAI &nbsp;|&nbsp;
Dataset: Cirrhosis Patient Survival &nbsp;|&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
