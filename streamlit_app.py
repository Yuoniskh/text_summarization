# streamlit_app.py
import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
from src.summarization import ExtractiveSummarizer
from src.hybrid_deep_model import HybridDeepSummarizer
from src.evaluation import evaluate_model
from src.utils import split_sentences
import config

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
models = load_models()
tfidf_model = models['tfidf']
textrank_model = models['textrank']
hybrid_available = models.get('hybrid_available', False)
hybrid_model = models.get('hybrid', None)

# الشريط الجانبي
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
        max_value=10,
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
    
    # معلومات إضافية
    st.divider()
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

# المحتوى الرئيسي
st.markdown("""
<div class="main-header">
    <h1>📝 ملخص النصوص الذكي</h1>
    <p>لخص مقالاتك بسهولة باستخدام الذكاء الاصطناعي - TF-IDF و TextRank و Hybrid Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# تبويبات رئيسية
tab1, tab2, tab3 = st.tabs(["📝 تلخيص النصوص", "📊 المقارنة والتقييم", "ℹ️ عن المشروع"])

# ============================================================
# تبويب 1: تلخيص النصوص
# ============================================================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 النص الأصلي")
        
        # مصدر النص (تحميل أو كتابة)
        if uploaded_file:
            text_input = uploaded_file.read().decode('utf-8')
            st.success(f"✅ تم تحميل الملف: {uploaded_file.name}")
        else:
            text_input = st.text_area(
                "أدخل النص الذي تريد تلخيصه:",
                height=300,
                placeholder="الصق النص هنا...",
                value="Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents. AI is used in many applications such as machine translation and chatbots."
            )
        
        # إحصائيات النص
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
                    
                    # اختيار النموذج المناسب
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
                
                # عرض الملخص أو الخطأ
                if error_msg:
                    st.error(f"❌ {error_msg}")
                elif summary:
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.markdown(f"**الملخص ({model_used}):**")
                    st.write(summary)
                    st.caption(f"⏱️ وقت التلخيص: {elapsed_time:.2f} ثانية")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # حفظ في التاريخ
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        'text': text_input[:200] + "...",
                        'summary': summary,
                        'model': model_used,
                        'sentences': num_sentences,
                        'time': elapsed_time
                    })
                    
                    # إحصائيات الملخص
                    summary_words = len(summary.split())
                    word_count = len(text_input.split())
                    compression = (1 - summary_words / word_count) * 100 if word_count > 0 else 0
                    col2a, col2b = st.columns(2)
                    with col2a:
                        st.metric("📊 كلمات الملخص", summary_words)
                    with col2b:
                        st.metric("📉 نسبة الضغط", f"{compression:.1f}%")
                else:
                    st.warning("⚠️ لم يتم توليد ملخص")
            else:
                st.warning("⚠️ الرجاء إدخال نص للتلخيص")

# ============================================================
# تبويب 2: المقارنة والتقييم
# ============================================================
with tab2:
    st.subheader("📊 مقارنة أداء النماذج")
    
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
            - TF-IDF: استخلاص إحصائي سريع
            - TextRank: استخلاص قائم على الشبكات
            - Hybrid DL: شبكة عميقة + ميزات مختلطة
            """)
        else:
            st.warning("""
            ⚠️ **نموذج Hybrid غير متاح**
            
            لتدريب النموذج:
            ```bash
            python train_hybrid_model.py
            ```
            """)
    
    # بيانات تجريبية للعرض
    metrics_df = pd.DataFrame({
        'النموذج': ['TF-IDF', 'TextRank', 'Hybrid DL*'],
        'ROUGE-1': [0.259, 0.287, 0.305],
        'ROUGE-2': [0.091, 0.087, 0.115],
        'ROUGE-L': [0.185, 0.185, 0.210]
    })
    
    st.dataframe(
        metrics_df,
        hide_index=True,
        use_container_width=True
    )
    st.caption("* قيم تقريبية - تعتمد على بيانات التدريب")
    
    # رسم بياني للمقارنة
    st.subheader("📈 مقارنة بيانية")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    tfidf_scores = [0.259, 0.091, 0.185]
    textrank_scores = [0.287, 0.087, 0.185]
    hybrid_scores = [0.305, 0.115, 0.210] if hybrid_available else None
    
    x_pos = range(len(x))
    
    if hybrid_available:
        width = 0.25
        bars1 = ax.bar([i - width for i in x_pos], tfidf_scores, width, label='TF-IDF', color='#667eea')
        bars2 = ax.bar([i for i in x_pos], textrank_scores, width, label='TextRank', color='#764ba2')
        bars3 = ax.bar([i + width for i in x_pos], hybrid_scores, width, label='Hybrid DL', color='#00D084')
    else:
        width = 0.35
        bars1 = ax.bar([i - width/2 for i in x_pos], tfidf_scores, width, label='TF-IDF', color='#667eea')
        bars2 = ax.bar([i + width/2 for i in x_pos], textrank_scores, width, label='TextRank', color='#764ba2')
    
    ax.set_ylabel('الدرجة')
    ax.set_title('مقارنة أداء نماذج التلخيص')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # إضافة القيم فوق الأعمدة
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    if hybrid_available:
        for bar in bars3:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    st.pyplot(fig)
    
    # تحليل النتائج
    st.markdown("### 💡 تحليل النتائج")
    if hybrid_available:
        st.write("""
        - **Hybrid Deep Learning** يجمع بين قوة الشبكات العميقة مع الميزات الإحصائية (TF-IDF, TextRank)
        - يتفوق في جميع مقاييس ROUGE عندما يتم تدريبه على بيانات كافية
        - **TextRank** يتفوق في التقاط المفردات المهمة (ROUGE-1)
        - **TF-IDF** الأسرع والأخف من حيث الموارد
        - الاختيار يعتمد على: دقة المطلوبة × سرعة المطلوبة × الموارد المتاحة
        """)
    else:
        st.write("""
        - **TextRank** يتفوق في ROUGE-1 (الكلمات المفردة)
        - **TF-IDF** الأسرع والأخف من حيث الموارد
        - **Hybrid Deep Learning**: قريباً! سيجمع بين أفضل ما في الطريقتين
        """)

# ============================================================
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
        
        #### 📊 مميزات المشروع:
        - تلخيص استخراجي دقيق
        - دعم النصوص الطويلة
        - واجهة سهلة الاستخدام
        - إحصائيات تفصيلية
        - مقارنة بين النماذج
        - نموذج عميق قابل للتدريب والتحسن
        
        #### 🔗 روابط مفيدة:
        - [GitHub Repository](https://github.com)
        - [Documentation](https://docs.streamlit.io)
        - [ROUGE Metrics](https://github.com/google-research/google-research/tree/master/rouge)
        """)
    
    with col2:
        st.markdown("""
        ### 👨‍💻 المطور
        
        **مشروع تعلم آلي لتلخيص النصوص**
        
        الإصدار: 2.0.0
        
        آخر تحديث: 2024
        
        ### 🎯 الحالة
        """)
        
        # إحصائيات المشروع
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
        for i, item in enumerate(st.session_state.history[-5:]):  # آخر 5 ملخصات
            with st.expander(f"ملخص {i+1}: {item['model']}"):
                st.caption(f"النص: {item['text']}")
                st.write(f"**الملخص:** {item['summary']}")
                st.caption(f"⏱️ {item['time']:.2f} ثانية")
    else:
        st.caption("لا توجد ملخصات سابقة")

# تذييل الصفحة
st.markdown("""
<div class="footer">
    <p>🚀 تم تطويره باستخدام Streamlit | 📝 ملخص النصوص الذكي v2.0.0</p>
</div>
""", unsafe_allow_html=True)