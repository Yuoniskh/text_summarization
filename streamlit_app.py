# streamlit_app.py
import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import numpy as np
from src.summarization import ExtractiveSummarizer
from src.hybrid_deep_model import HybridDeepSummarizer
from src.evaluation import evaluate_model
from src.utils import split_sentences
import config
import json

# إضافة استيراد لـ ROUGE
from rouge_score import rouge_scorer

# إعداد الصفحة
st.set_page_config(
    page_title="📝 ملخص النصوص الذكي",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص لتحسين المظهر
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .best-model {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# تهيئة النماذج (تخزين مؤقت لتجنب إعادة التحميل)
@st.cache_resource
def load_models():
    """تحميل نماذج التلخيص (يتم مرة واحدة فقط)."""
    models = {}
    
    with st.spinner("🔄 جاري تحميل نموذج TF-IDF..."):
        models['tfidf'] = ExtractiveSummarizer(method='tfidf')
    
    with st.spinner("🔄 جاري تحميل نموذج TextRank (قد يستغرق دقيقة)..."):
        models['textrank'] = ExtractiveSummarizer(method='textrank')
    
    # محاولة تحميل نموذج Hybrid
    if os.path.exists(config.HYBRID_MODEL_PATH):
        try:
            with st.spinner("🔄 جاري تحميل نموذج Hybrid Deep Learning..."):
                models['hybrid'] = HybridDeepSummarizer.load_model(config.HYBRID_MODEL_PATH)
                models['hybrid_available'] = True
        except Exception as e:
            st.warning(f"⚠️ تعذر تحميل نموذج Hybrid: {str(e)}")
            models['hybrid_available'] = False
    else:
        models['hybrid_available'] = False
    
    return models

# تحميل النماذج
all_models = load_models()

# استخراج النماذج من القاموس بشكل آمن
tfidf_model = all_models['tfidf']
textrank_model = all_models['textrank']
hybrid_available = all_models.get('hybrid_available', False)
hybrid_model = all_models.get('hybrid', None)

# تهيئة ROUGE scorer
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# ============================================================
# دوال مساعدة
# ============================================================

def calculate_rouge_scores(text, reference_summary):
    """حساب ROUGE scores لنص معين مقابل ملخص مرجعي."""
    if not text or not reference_summary:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    try:
        scores = rouge_scorer_obj.score(reference_summary, text)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    except Exception as e:
        st.warning(f"خطأ في حساب ROUGE: {str(e)}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

def evaluate_model_on_text(model, model_name, text, reference_summary, num_sentences=3):
    """تقييم نموذج على نص معين وإرجاع ROUGE scores."""
    try:
        if model_name == "Hybrid DL" and not hybrid_available:
            return None
            
        # توليد الملخص
        if model_name == "TF-IDF":
            summary = model.summarize(text, num_sentences=num_sentences)
        elif model_name == "TextRank":
            summary = model.summarize(text, num_sentences=num_sentences)
        elif model_name == "Hybrid DL":
            summary = model.summarize(text, num_sentences=num_sentences)
        else:
            return None
            
        # حساب ROUGE scores
        scores = calculate_rouge_scores(summary, reference_summary)
        return {
            'model': model_name,
            'summary': summary,
            'rouge1': scores['rouge1'],
            'rouge2': scores['rouge2'],
            'rougeL': scores['rougeL']
        }
    except Exception as e:
        st.error(f"خطأ في تقييم {model_name}: {str(e)}")
        return None

def plot_model_comparison(results, title="مقارنة أداء نماذج التلخيص"):
    """رسم مقارنة بين النماذج باستخدام ROUGE scores."""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # أعمدة مقارنة
    x = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    x_pos = np.arange(len(x))
    width = 0.8 / len(df)
    colors = ['#667eea', '#764ba2', '#00D084', '#FF6B6B', '#FFB347']
    
    for i, (idx, row) in enumerate(df.iterrows()):
        scores = [row['rouge1'], row['rouge2'], row['rougeL']]
        bar_positions = x_pos + i * width - (len(df) - 1) * width / 2
        bars = axes[0].bar(bar_positions, scores, width, label=row['model'], 
                          color=colors[i % len(colors)])
        
        for bar, score in zip(bars, scores):
            axes[0].text(bar.get_x() + bar.get_width()/2., score + 0.01, 
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    axes[0].set_ylabel('الدرجة')
    axes[0].set_title('مقارنة ROUGE Scores')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x)
    axes[0].legend(loc='upper left')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    
    # رادار Chart
    if len(df) >= 3:
        categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(1, 2, 2, projection='polar')
        
        for i, (idx, row) in enumerate(df.iterrows()):
            values = [row['rouge1'], row['rouge2'], row['rougeL']]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=row['model'], color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('مقارنة ROUGE (رادار)', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def load_training_history():
    """تحميل تاريخ التدريب من ملف CSV."""
    history_path = config.TRAINING_HISTORY_CSV
    if os.path.exists(history_path):
        try:
            return pd.read_csv(history_path)
        except Exception as e:
            st.warning(f"⚠️ خطأ في قراءة تاريخ التدريب: {str(e)}")
            return None
    return None

def load_metrics_json():
    """تحميل مقاييس التدريب من ملف JSON."""
    metrics_path = config.METRICS_JSON
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"⚠️ خطأ في قراءة المقاييس: {str(e)}")
            return None
    return None

def plot_training_curves(history_df):
    """رسم منحنيات التدريب الكاملة."""
    if history_df is None or history_df.empty:
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Loss
    if 'train_loss' in history_df.columns and 'val_loss' in history_df.columns:
        axes[0, 0].plot(history_df['train_loss'], label='Training Loss', color='#FF6B6B', linewidth=2)
        axes[0, 0].plot(history_df['val_loss'], label='Validation Loss', color='#FFB347', linewidth=2)
        axes[0, 0].set_title('منحنى Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        best_val_loss_idx = history_df['val_loss'].idxmin()
        axes[0, 0].scatter(best_val_loss_idx, history_df['val_loss'].min(), 
                          color='red', s=100, zorder=5, 
                          label=f'Best: {history_df["val_loss"].min():.4f}')
        axes[0, 0].legend()
    
    # 2. Accuracy
    if 'train_accuracy' in history_df.columns and 'val_accuracy' in history_df.columns:
        axes[0, 1].plot(history_df['train_accuracy'], label='Training Accuracy', color='#00D084', linewidth=2)
        axes[0, 1].plot(history_df['val_accuracy'], label='Validation Accuracy', color='#667eea', linewidth=2)
        axes[0, 1].set_title('منحنى Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        best_val_acc_idx = history_df['val_accuracy'].idxmax()
        axes[0, 1].scatter(best_val_acc_idx, history_df['val_accuracy'].max(), 
                          color='red', s=100, zorder=5,
                          label=f'Best: {history_df["val_accuracy"].max():.4f}')
        axes[0, 1].legend()
    
    # 3. F1 Score
    if 'train_f1' in history_df.columns and 'val_f1' in history_df.columns:
        axes[0, 2].plot(history_df['train_f1'], label='Training F1', color='#764ba2', linewidth=2)
        axes[0, 2].plot(history_df['val_f1'], label='Validation F1', color='#667eea', linewidth=2)
        axes[0, 2].set_title('منحنى F1 Score', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        best_val_f1_idx = history_df['val_f1'].idxmax()
        axes[0, 2].scatter(best_val_f1_idx, history_df['val_f1'].max(), 
                          color='red', s=100, zorder=5,
                          label=f'Best: {history_df["val_f1"].max():.4f}')
        axes[0, 2].legend()
    
    # 4. Precision
    if 'train_precision' in history_df.columns and 'val_precision' in history_df.columns:
        axes[1, 0].plot(history_df['train_precision'], label='Training Precision', color='#FF6B6B', linewidth=2)
        axes[1, 0].plot(history_df['val_precision'], label='Validation Precision', color='#FFB347', linewidth=2)
        axes[1, 0].set_title('منحنى Precision', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Recall
    if 'train_recall' in history_df.columns and 'val_recall' in history_df.columns:
        axes[1, 1].plot(history_df['train_recall'], label='Training Recall', color='#4ECDC4', linewidth=2)
        axes[1, 1].plot(history_df['val_recall'], label='Validation Recall', color='#45B7D1', linewidth=2)
        axes[1, 1].set_title('منحنى Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. مقارنة جميع المقاييس
    if all(col in history_df.columns for col in ['val_accuracy', 'val_precision', 'val_recall', 'val_f1']):
        axes[1, 2].plot(history_df['val_accuracy'], label='Accuracy', color='#00D084', linewidth=2)
        axes[1, 2].plot(history_df['val_precision'], label='Precision', color='#FF6B6B', linewidth=2)
        axes[1, 2].plot(history_df['val_recall'], label='Recall', color='#4ECDC4', linewidth=2)
        axes[1, 2].plot(history_df['val_f1'], label='F1 Score', color='#764ba2', linewidth=2)
        axes[1, 2].set_title('مقارنة مقاييس التحقق', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('منحنيات تدريب النموذج الهجين', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_confusion_matrix_from_file():
    """عرض مصفوفة الارتباك من الملف المحفوظ."""
    cm_path = config.CONFUSION_MATRIX_IMAGE
    if os.path.exists(cm_path):
        try:
            from PIL import Image
            img = Image.open(cm_path)
            return img
        except Exception as e:
            st.warning(f"⚠️ خطأ في تحميل مصفوفة الارتباك: {str(e)}")
            return None
    return None

# ============================================================
# الشريط الجانبي
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/summarize.png", width=80)
    st.title("⚙️ الإعدادات")
    
    # اختيار النموذج
    model_options = ["🤖 TextRank (أفضل دقة)", "📊 TF-IDF (أسرع)"]
    if hybrid_available:
        model_options.append("🧠 Hybrid Deep Learning (متقدم)")
    
    model_choice = st.selectbox(
        "اختر نموذج التلخيص:",
        model_options,
        help="اختر النموذج المناسب لاحتياجاتك"
    )
    
    # عدد الجمل
    num_sentences = st.slider(
        "عدد جمل الملخص:",
        min_value=1,
        max_value=15,
        value=3,
        help="اختر عدد الجمل التي تريدها في الملخص الناتج"
    )
    
    st.divider()
    
    # معلومات عن النماذج المتاحة
    st.subheader("📋 النماذج المتاحة")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("✅ TF-IDF", "جاهز")
        st.metric("✅ TextRank", "جاهز")
    with col2:
        if hybrid_available:
            st.metric("✅ Hybrid DL", "جاهز")
        else:
            st.metric("❌ Hybrid DL", "غير متاح")
            if st.button("📚 تدريب النموذج"):
                st.info("""
                لتدريب نموذج Hybrid Deep Learning، قم بتشغيل:
                ```bash
                python train_hybrid_model.py
                ```
                بعد انتهاء التدريب، أعد تحميل الصفحة.
                """)
    
    st.divider()
    
    # عرض نتائج التقييم الأخيرة
    if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
        latest_eval = st.session_state.evaluation_results[-1]
        st.subheader("📊 آخر تقييم")
        for result in latest_eval['results']:
            col_e1, col_e2 = st.columns([2, 1])
            with col_e1:
                st.caption(f"{result['model']}")
            with col_e2:
                st.caption(f"R1: {result['rouge1']:.3f}")
    
    st.subheader("📊 إحصائيات")
    if 'history' in st.session_state and st.session_state.history:
        st.metric("عدد الملخصات المنشأة", len(st.session_state.history))
    
    st.divider()
    
    # قسم التحميل
    st.subheader("📁 تحميل ملف")
    uploaded_file = st.file_uploader(
        "اختر ملف نصي (.txt):",
        type=['txt'],
        help="يمكنك تحميل ملف نصي لتلخيصه"
    )

# ============================================================
# المحتوى الرئيسي
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>📝 ملخص النصوص الذكي</h1>
    <p>لخص مقالاتك بسهولة باستخدام الذكاء الاصطناعي - TF-IDF و TextRank و Hybrid Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# تعريف التبويبات
# ============================================================
tab1, tab2, tab3 = st.tabs(["📝 تلخيص النصوص", "📊 المقارنة والتقييم", "ℹ️ عن المشروع"])

# ============================================================
# تبويب 1: تلخيص النصوص
# ============================================================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 النص الأصلي")
        
        if uploaded_file:
            text_input = uploaded_file.read().decode('utf-8')
            st.success(f"✅ تم تحميل الملف: {uploaded_file.name}")
        else:
            text_input = st.text_area(
                "أدخل النص الذي تريد تلخيصه:",
                height=300,
                placeholder="الصق النص هنا...",
                value=""
            )
        
        if text_input:
            word_count = len(text_input.split())
            sent_count = len(split_sentences(text_input))
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("📊 عدد الكلمات", word_count)
            with col1b:
                st.metric("📝 عدد الجمل", sent_count)
    
    with col2:
        st.subheader("✨ الملخص الناتج")
        
        if st.button("🚀 توليد الملخص", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("🔄 جاري التلخيص..."):
                    start_time = time.time()
                    summary = None
                    model_used = None
                    error_msg = None
                    
                    try:
                        if "TextRank" in model_choice:
                            summary = textrank_model.summarize(text_input, num_sentences=num_sentences)
                            model_used = "TextRank"
                        elif "Hybrid" in model_choice:
                            if hybrid_model and hybrid_available:
                                summary = hybrid_model.summarize(text_input, num_sentences=num_sentences)
                                model_used = "Hybrid Deep Learning"
                            else:
                                error_msg = "نموذج Hybrid غير متاح. يرجى تدريبه أولاً."
                        else:
                            summary = tfidf_model.summarize(text_input, num_sentences=num_sentences)
                            model_used = "TF-IDF"
                    except Exception as e:
                        error_msg = f"خطأ في التلخيص: {str(e)}"
                    
                    elapsed_time = time.time() - start_time
                
                if error_msg:
                    st.error(f"❌ {error_msg}")
                elif summary:
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.markdown(f"**الملخص ({model_used}):**")
                    st.write(summary)
                    st.caption(f"⏱️ وقت التلخيص: {elapsed_time:.2f} ثانية")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        'text': text_input[:200] + "...",
                        'summary': summary,
                        'model': model_used,
                        'sentences': num_sentences,
                        'time': elapsed_time
                    })
                    
                    summary_words = len(summary.split())
                    word_count = len(text_input.split())
                    compression = (1 - summary_words / word_count) * 100 if word_count > 0 else 0
                    col2a, col2b = st.columns(2)
                    with col2a:
                        st.metric("📊 كلمات الملخص", summary_words)
                    with col2b:
                        st.metric("📉 نسبة الضغط", f"{compression:.1f}%")
                    
                    st.markdown("---")
                    st.subheader("📊 تقييم دقة ROUGE")
                    
                    reference_summary = st.text_area(
                        "أدخل ملخص مرجعي للمقارنة (اختياري):",
                        height=100,
                        placeholder="أدخل ملخص مرجعي لقياس دقة النموذج..."
                    )
                    
                    if reference_summary.strip():
                        with st.spinner("🔄 جاري حساب دقة ROUGE..."):
                            rouge_scores = calculate_rouge_scores(summary, reference_summary)
                            col_r1, col_r2, col_r3 = st.columns(3)
                            with col_r1:
                                st.metric("🎯 ROUGE-1", f"{rouge_scores['rouge1']:.4f}")
                            with col_r2:
                                st.metric("🎯 ROUGE-2", f"{rouge_scores['rouge2']:.4f}")
                            with col_r3:
                                st.metric("🎯 ROUGE-L", f"{rouge_scores['rougeL']:.4f}")
                    else:
                        st.info("💡 أدخل ملخص مرجعي لرؤية دقة ROUGE للنموذج الحالي")
                else:
                    st.warning("⚠️ لم يتم توليد ملخص")
            else:
                st.warning("⚠️ الرجاء إدخال نص للتلخيص")

# ============================================================
# تبويب 2: المقارنة والتقييم
# ============================================================
with tab2:
    st.subheader("📊 مقارنة وتقييم النماذج")
    
    # القسم 1: تقييم النماذج على نص مخصص
    st.markdown("### 🎯 تقييم النماذج على نص مخصص")
    
    eval_text = st.text_area(
        "أدخل نص للتقييم:",
        height=150,
        placeholder="أدخل نص طويل لتقييم أداء النماذج عليه...",
        value=""
    )
    
    eval_reference = st.text_area(
        "ملخص مرجعي (اختياري):",
        height=100,
        placeholder="أدخل ملخص مرجعي لقياس دقة النماذج...",
        value=""
    )
    
    eval_sentences = st.slider(
        "عدد جمل الملخص:",
        min_value=1,
        max_value=15,
        value=3,
        key="eval_sentences"
    )
    
    col_eval1, col_eval2, col_eval3 = st.columns([1, 1, 1])
    with col_eval1:
        eval_button = st.button("🚀 قيّم النماذج", type="primary", use_container_width=True)
    
    with col_eval2:
        if st.button("📊 عرض منحنيات التدريب", use_container_width=True):
            st.session_state.show_training_curves = not st.session_state.get('show_training_curves', False)
    
    with col_eval3:
        if st.button("📈 عرض مصفوفة الارتباك", use_container_width=True):
            st.session_state.show_confusion_matrix = not st.session_state.get('show_confusion_matrix', False)
    
    if eval_button:
        if eval_text.strip():
            with st.spinner("🔄 جاري تقييم النماذج..."):
                results = []
                
                tfidf_result = evaluate_model_on_text(
                    tfidf_model, "TF-IDF", eval_text, eval_reference, eval_sentences
                )
                if tfidf_result:
                    results.append(tfidf_result)
                
                textrank_result = evaluate_model_on_text(
                    textrank_model, "TextRank", eval_text, eval_reference, eval_sentences
                )
                if textrank_result:
                    results.append(textrank_result)
                
                if hybrid_available and hybrid_model:
                    hybrid_result = evaluate_model_on_text(
                        hybrid_model, "Hybrid DL", eval_text, eval_reference, eval_sentences
                    )
                    if hybrid_result:
                        results.append(hybrid_result)
                
                if results:
                    st.success("✅ تم إكمال التقييم!")
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(
                        results_df[['model', 'rouge1', 'rouge2', 'rougeL']],
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    best_model = max(results, key=lambda x: x['rouge1'])
                    st.markdown(f"🏆 **أفضل أداء:** <span class='best-model'>{best_model['model']}</span> (ROUGE-1: {best_model['rouge1']:.4f})", 
                               unsafe_allow_html=True)
                    
                    st.markdown("### 📝 الملخصات الناتجة")
                    for result in results:
                        with st.expander(f"📋 ملخص {result['model']}"):
                            st.write(result['summary'])
                            st.caption(f"ROUGE-1: {result['rouge1']:.4f} | ROUGE-2: {result['rouge2']:.4f} | ROUGE-L: {result['rougeL']:.4f}")
                    
                    st.markdown("### 📈 الرسوم البيانية للمقارنة")
                    fig = plot_model_comparison(results)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    if 'evaluation_results' not in st.session_state:
                        st.session_state.evaluation_results = []
                    st.session_state.evaluation_results.append({
                        'timestamp': time.time(),
                        'text': eval_text[:200] + "...",
                        'reference': eval_reference,
                        'results': results
                    })
                else:
                    st.error("❌ فشل في تقييم النماذج")
        else:
            st.warning("⚠️ الرجاء إدخال نص للتقييم")
    
    # القسم 2: مقاييس النموذج الهجين
    st.markdown("---")
    st.markdown("### 🧠 تقييم النموذج الهجين (من التدريب)")
    
    if hybrid_available:
        metrics = load_metrics_json()
        
        if metrics:
            st.markdown("#### 📊 مقاييس النموذج الهجين")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### 🏋️ التدريب")
                if 'training' in metrics:
                    for key, value in metrics['training'].items():
                        st.metric(label=f"Training {key.capitalize()}", value=f"{value:.4f}")
            
            with col2:
                st.markdown("##### ✅ التحقق")
                if 'validation' in metrics:
                    for key, value in metrics['validation'].items():
                        st.metric(label=f"Validation {key.capitalize()}", value=f"{value:.4f}")
            
            with col3:
                st.markdown("##### 🎯 الاختبار")
                if 'testing' in metrics:
                    for key, value in metrics['testing'].items():
                        st.metric(label=f"Test {key.capitalize()}", value=f"{value:.4f}")
            
            if 'testing' in metrics:
                st.markdown("#### 🏆 ملخص الأداء")
                test = metrics['testing']
                col_a, col_b, col_c, col_d, col_e = st.columns(5)
                with col_a:
                    st.metric("🎯 Accuracy", f"{test.get('accuracy', 0):.4f}")
                with col_b:
                    st.metric("📊 Precision", f"{test.get('precision', 0):.4f}")
                with col_c:
                    st.metric("📈 Recall", f"{test.get('recall', 0):.4f}")
                with col_d:
                    st.metric("⭐ F1 Score", f"{test.get('f1', 0):.4f}")
                with col_e:
                    st.metric("📉 Loss", f"{test.get('loss', 0):.4f}")
        else:
            st.info("ℹ️ لا توجد مقاييس تدريب. قم بتدريب النموذج الهجين أولاً.")
            st.code("python train_hybrid_model.py", language="bash")
    else:
        st.warning("⚠️ النموذج الهجين غير متاح. قم بتدريبه أولاً.")
        st.code("python train_hybrid_model.py", language="bash")
    
    # القسم 3: منحنيات التدريب
    if st.session_state.get('show_training_curves', False):
        st.markdown("---")
        st.markdown("### 📈 منحنيات تدريب النموذج الهجين")
        
        if hybrid_available:
            history_df = load_training_history()
            if history_df is not None and not history_df.empty:
                st.markdown("#### 📊 إحصائيات التدريب")
                cols = st.columns(4)
                
                if 'val_accuracy' in history_df.columns:
                    best_val_acc = history_df['val_accuracy'].max()
                    best_val_acc_epoch = history_df['val_accuracy'].idxmax() + 1
                    cols[0].metric("🏆 أفضل Validation Accuracy", 
                                  f"{best_val_acc:.4f}", delta=f"Epoch {best_val_acc_epoch}")
                
                if 'val_f1' in history_df.columns:
                    best_val_f1 = history_df['val_f1'].max()
                    best_val_f1_epoch = history_df['val_f1'].idxmax() + 1
                    cols[1].metric("⭐ أفضل Validation F1", 
                                  f"{best_val_f1:.4f}", delta=f"Epoch {best_val_f1_epoch}")
                
                if 'val_loss' in history_df.columns:
                    best_val_loss = history_df['val_loss'].min()
                    best_val_loss_epoch = history_df['val_loss'].idxmin() + 1
                    cols[2].metric("📉 أفضل Validation Loss", 
                                  f"{best_val_loss:.4f}", delta=f"Epoch {best_val_loss_epoch}")
                
                cols[3].metric("📊 عدد الـ Epochs", f"{len(history_df)}")
                
                fig = plot_training_curves(history_df)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                
                with st.expander("📋 عرض بيانات التدريب التفصيلية"):
                    st.dataframe(history_df, use_container_width=True)
            else:
                st.info("ℹ️ لا توجد بيانات تدريب. قم بتدريب النموذج الهجين أولاً.")
        else:
            st.warning("⚠️ النموذج الهجين غير متاح. قم بتدريبه أولاً.")
    # القسم 4: مصفوفة الارتباك
    if st.session_state.get('show_confusion_matrix', False):
        st.markdown("---")
        st.markdown("### 🎯 مصفوفة الارتباك (Confusion Matrix)")
        
        if hybrid_available:
            cm_img = plot_confusion_matrix_from_file()
            
            if cm_img:
                st.image(cm_img, caption="مصفوفة الارتباك للنموذج الهجين", use_container_width=True)
                
                st.markdown("#### 📖 تفسير المصفوفة")
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.info("""
                    **✅ True Positives (TP):** 
                    الجمل التي تم اختيارها بشكل صحيح
                    
                    **❌ False Negatives (FN):**
                    الجمل التي تم تجاهلها ولكن كان يجب اختيارها
                    """)
                with col_exp2:
                    st.info("""
                    **❌ False Positives (FP):**
                    الجمل التي تم اختيارها ولكن كان يجب تجاهلها
                    
                    **✅ True Negatives (TN):**
                    الجمل التي تم تجاهلها بشكل صحيح
                    """)
                
                report_path = config.CLASSIFICATION_REPORT
                if os.path.exists(report_path):
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report = f.read()
                    with st.expander("📋 عرض تقرير التصنيف الكامل"):
                        st.code(report, language='text')
            else:
                st.info("ℹ️ لم يتم العثور على مصفوفة الارتباك. تأكد من تدريب النموذج.")
        else:
            st.warning("⚠️ النموذج الهجين غير متاح. قم بتدريبه أولاً.")
    
    # القسم 5: معلومات إضافية
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 مقاييس ROUGE")
        st.info("""
        **ROUGE** هو مقياس لتقييم جودة الملخصات:
        - **ROUGE-1**: تشابه الكلمات المفردة
        - **ROUGE-2**: تشابه أزواج الكلمات
        - **ROUGE-L**: تشابه تسلسل الجمل
        """)
    
    with col2:
        st.markdown("### 💡 معلومات النماذج")
        if hybrid_available:
            st.success("""
            ✅ **الثلاثة نماذج متاحة:**
            - **TF-IDF**: استخلاص إحصائي سريع
            - **TextRank**: استخلاص قائم على الشبكات
            - **Hybrid DL**: شبكة عميقة + ميزات مختلطة
            """)
        else:
            st.warning("""
            ⚠️ **نموذج Hybrid غير متاح**
            
            لتدريب النموذج:
            ```bash
            python train_hybrid_model.py""")   
# في الجزء الذي يعرض تاريخ التقييمات (حوالي السطر 826)
# ✅ التصحيح: التحقق من وجود المتغير
    if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
        st.markdown("---")
        st.markdown("### 📋 تاريخ التقييمات")
        
        # عرض آخر 3 تقييمات مع التحقق من أن القائمة ليست فارغة
        eval_history = st.session_state.evaluation_results[-3:] if len(st.session_state.evaluation_results) >= 3 else st.session_state.evaluation_results
        
        for i, eval_item in enumerate(eval_history):
            with st.expander(f"تقييم {len(st.session_state.evaluation_results) - len(eval_history) + i + 1}: {eval_item.get('text', 'نص غير متاح')}"):
                eval_df = pd.DataFrame(eval_item.get('results', []))
                if not eval_df.empty:
                    st.dataframe(eval_df[['model', 'rouge1', 'rouge2', 'rougeL']], 
                                hide_index=True, use_container_width=True)
                else:
                    st.info("لا توجد نتائج تقييم متاحة")
    else:
        # إذا لم يكن هناك تقييمات سابقة، يمكن عرض رسالة
        pass  # أو st.info("لا توجد تقييمات سابقة")=========================================================
# تبويب 3: عن المشروع
# ============================================================
with tab3:
    st.subheader("ℹ️ عن المشروع")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📚 ملخص النصوص الذكي
        
        هذا المشروع هو نظام متكامل لتلخيص النصوص باستخدام تقنيات مختلفة:
        
        #### 🛠️ التقنيات المستخدمة:
        - **TF-IDF**: تقنية إحصائية لاستخراج الجمل المهمة
        - **TextRank**: خوارزمية مستوحاة من PageRank لتحديد أهمية الجمل
        - **Hybrid Deep Learning**: شبكة عميقة تجمع بين:
          - TF-IDF scores: درجات إحصائية للجمل
          - TextRank scores: درجات قائمة على الشبكات
          - Sentence position: موضع الجملة في النص
          - Sentence length: طول الجملة
          - Embedding features: تمثيلات عميقة للجمل
        - **Sentence Transformers**: نماذج ذكاء اصطناعي لتمثيل الجمل
         #### 🧠 الميزات المستخدمة في النموذج الهجين (13 ميزة):
        1. **TF-IDF**: درجة أهمية الكلمات
        2. **TextRank**: درجة أهمية الجملة
        3. **Position**: موضع الجملة
        4. **Length**: طول الجملة
        5. **BM25**: درجة تشابه الجملة
        6. **Centrality**: مركزية الجملة
        7. **Entropy**: إنتروبيا الجملة
        8. **NER**: الكيانات المسماة
        9. **POS**: أجزاء الكلام
        10. **Position Binary**: موضع الجملة (ثنائي)
        11. **Stopword Ratio**: نسبة كلمات التوقف
        12. **Unique Ratio**: نسبة الكلمات الفريدة
        13. **Embedding**: تضمين الجملة
        #### 📊 مميزات المشروع:
        - تلخيص استخراجي دقيق
        - دعم النصوص الطويلة
        - واجهة سهلة الاستخدام
        - إحصائيات تفصيلية
        - مقارنة بين النماذج
        - نموذج عميق قابل للتدريب والتحسن
        
        #### 🔧 ملفات المشروع الرئيسية:
        - `config.py`: إعدادات المشروع
        - `train_hybrid_model.py`: تدريب النموذج الهجين
        - `src/hybrid_deep_model.py`: نموذج التعلم العميق
        - `src/summarization.py`: نماذج التلخيص الأساسية
        - `src/evaluation.py`: أدوات التقييم
        - `src/preprocessing.py`: تنظيف البيانات
        - `src/utils.py`: دوال مساعدة
        
        #### 🔗 روابط مفيدة:
        - [GitHub Repository](https://github.com)
        - [Documentation](https://docs.streamlit.io)
        - [ROUGE Metrics](https://github.com/google-research/google-research/tree/master/rouge)
        """)
    
    with col2:
        st.markdown("""
        ### 👨‍💻 المطور
        
        **مشروع تعلم آلي لتلخيص النصوص**
        
        الإصدار: 2.0.0 (PyTorch)
        
        آخر تحديث: 2024
        
        ### 🎯 الحالة
        """)
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("📊 النماذج", "3" if hybrid_available else "2")
            st.metric("⚡ سرعة", "~0.1 ث/جملة")
        with col2b:
            st.metric("📦 حجم", "~150 MB" if hybrid_available else "~90 MB")
            st.metric("🔧 Status", "متطور")

# ============================================================
# سجل التاريخ (في الشريط الجانبي)
# ============================================================
with st.sidebar:
    st.divider()
    st.subheader("📜 سجل الملخصات")
    
    if 'history' in st.session_state and st.session_state.history:
        for i, item in enumerate(st.session_state.history[-3:]):  # آخر 3 ملخصات
            with st.expander(f"ملخص {i+1}: {item['model']}"):
                st.caption(f"النص: {item['text']}")
                st.write(f"**الملخص:** {item['summary']}")
    
    # سجل التقييمات
    if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
        st.divider()
        st.subheader("📊 سجل التقييمات")
        
        for i, eval_item in enumerate(st.session_state.evaluation_results[-2:]):  # آخر تقييمان
            with st.expander(f"تقييم {i+1}"):
                st.caption(f"النص: {eval_item['text']}")
                for result in eval_item['results']:
                    st.write(f"**{result['model']}:** R1={result['rouge1']:.3f}")
                if 'timestamp' in eval_item:
                    st.caption(f"⏱️ {time.strftime('%H:%M:%S', time.localtime(eval_item['timestamp']))}")
    else:
        st.caption("لا توجد ملخصات سابقة")

# تذييل الصفحة
st.markdown("""
<div class="footer">
    <p>🚀 تم تطويره باستخدام Streamlit | 📝 ملخص النصوص الذكي v2.0.0 (PyTorch)</p>
</div>
""", unsafe_allow_html=True)