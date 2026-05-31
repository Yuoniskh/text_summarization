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

# تهيئة ROUGE scorer
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# وظيفة لحساب ROUGE scores لنص معين
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

# وظيفة لتقييم نموذج على نص معين
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
                    
                    # عرض ROUGE scores إذا كان هناك ملخص مرجعي
                    st.markdown("---")
                    st.subheader("📊 تقييم دقة ROUGE")
                    
                    # إدخال ملخص مرجعي للمقارنة
                    reference_summary = st.text_area(
                        "أدخل ملخص مرجعي للمقارنة (اختياري):",
                        height=100,
                        placeholder="أدخل ملخص مرجعي لقياس دقة النموذج...",
                        help="سيتم حساب ROUGE scores بين الملخص الناتج والملخص المرجعي"
                    )
                    
                    if reference_summary.strip():
                        with st.spinner("🔄 جاري حساب دقة ROUGE..."):
                            # حساب ROUGE للملخص الحالي
                            rouge_scores = calculate_rouge_scores(summary, reference_summary)
                            
                            # عرض النتائج
                            col_r1, col_r2, col_r3 = st.columns(3)
                            with col_r1:
                                st.metric("🎯 ROUGE-1", f"{rouge_scores['rouge1']:.4f}")
                            with col_r2:
                                st.metric("🎯 ROUGE-2", f"{rouge_scores['rouge2']:.4f}")
                            with col_r3:
                                st.metric("🎯 ROUGE-L", f"{rouge_scores['rougeL']:.4f}")
                            
                            # تفسير النتائج
                            avg_score = (rouge_scores['rouge1'] + rouge_scores['rouge2'] + rouge_scores['rougeL']) / 3
                            if avg_score >= 0.7:
                                st.success("🎉 دقة ممتازة! الملخص يطابق المرجع بشكل جيد")
                            elif avg_score >= 0.5:
                                st.info("👍 دقة جيدة. الملخص يغطي معظم النقاط المهمة")
                            elif avg_score >= 0.3:
                                st.warning("⚠️ دقة متوسطة. يمكن تحسين الملخص")
                            else:
                                st.error("❌ دقة منخفضة. الملخص يحتاج مراجعة")
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
    st.subheader("📊 مقارنة أداء النماذج")
    
    # قسم تقييم النماذج على نص مخصص
    st.markdown("### 🎯 تقييم النماذج على نص مخصص")
    
    eval_text = st.text_area(
        "أدخل نص للتقييم:",
        height=150,
        placeholder="أدخل نص طويل لتقييم أداء النماذج عليه...",
        value="Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents: any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term artificial intelligence is often used to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving. As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says AI is whatever hasn't been done yet."
    )
    
    eval_reference = st.text_area(
        "ملخص مرجعي (اختياري):",
        height=100,
        placeholder="أدخل ملخص مرجعي لقياس دقة النماذج...",
        value="AI is machine intelligence that mimics human cognitive functions. The field focuses on intelligent agents that perceive their environment and take optimal actions. Tasks requiring intelligence are redefined as AI capabilities grow."
    )
    
    eval_sentences = st.slider(
        "عدد جمل الملخص:",
        min_value=1,
        max_value=5,
        value=3,
        key="eval_sentences"
    )
    
    if st.button("🚀 قيّم النماذج", type="primary", use_container_width=True):
        if eval_text.strip():
            with st.spinner("🔄 جاري تقييم النماذج..."):
                results = []
                
                # تقييم TF-IDF
                tfidf_result = evaluate_model_on_text(
                    tfidf_model, "TF-IDF", eval_text, eval_reference, eval_sentences
                )
                if tfidf_result:
                    results.append(tfidf_result)
                
                # تقييم TextRank
                textrank_result = evaluate_model_on_text(
                    textrank_model, "TextRank", eval_text, eval_reference, eval_sentences
                )
                if textrank_result:
                    results.append(textrank_result)
                
                # تقييم Hybrid إذا كان متاحاً
                if hybrid_available and hybrid_model:
                    hybrid_result = evaluate_model_on_text(
                        hybrid_model, "Hybrid DL", eval_text, eval_reference, eval_sentences
                    )
                    if hybrid_result:
                        results.append(hybrid_result)
                
                # عرض النتائج
                if results:
                    st.success("✅ تم إكمال التقييم!")
                    
                    # جدول النتائج
                    results_df = pd.DataFrame(results)
                    st.dataframe(
                        results_df[['model', 'rouge1', 'rouge2', 'rougeL']],
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # أفضل نموذج
                    best_model = max(results, key=lambda x: x['rouge1'])
                    st.info(f"🏆 **أفضل أداء:** {best_model['model']} (ROUGE-1: {best_model['rouge1']:.4f})")
                    
                    # عرض الملخصات
                    st.markdown("### 📝 الملخصات الناتجة")
                    for result in results:
                        with st.expander(f"📋 ملخص {result['model']}"):
                            st.write(result['summary'])
                            st.caption(f"ROUGE-1: {result['rouge1']:.4f} | ROUGE-2: {result['rouge2']:.4f} | ROUGE-L: {result['rougeL']:.4f}")
                    
                    # حفظ النتائج في session state
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
    
    st.divider()
    
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
    
    # بيانات تجريبية للعرض (محدثة بالنتائج الجديدة)
    if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
        # استخدام آخر نتائج تقييم
        latest_eval = st.session_state.evaluation_results[-1]
        latest_results = latest_eval['results']
        
        metrics_data = []
        for result in latest_results:
            metrics_data.append({
                'النموذج': result['model'],
                'ROUGE-1': result['rouge1'],
                'ROUGE-2': result['rouge2'],
                'ROUGE-L': result['rougeL']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.success("📊 **نتائج التقييم الحديثة** (بناءً على النص المدخل)")
    else:
        # بيانات افتراضية
        metrics_df = pd.DataFrame({
            'النموذج': ['TF-IDF', 'TextRank', 'Hybrid DL*'],
            'ROUGE-1': [0.259, 0.287, 0.333],
            'ROUGE-2': [0.091, 0.087, 0.129],
            'ROUGE-L': [0.185, 0.185, 0.213]
        })
        st.info("📊 **نتائج تجريبية** (استخدم قسم التقييم أعلاه للحصول على نتائج دقيقة)")
    
    st.dataframe(
        metrics_df,
        hide_index=True,
        use_container_width=True
    )
    
    if 'evaluation_results' not in st.session_state or not st.session_state.evaluation_results:
        st.caption("* قيم تقريبية - تعتمد على بيانات التدريب. استخدم التقييم أعلاه لنتائج دقيقة.")
    
    # رسم بياني للمقارنة
    st.subheader("📈 مقارنة بيانية")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    x_pos = range(len(x))
    
    if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
        # استخدام النتائج الحديثة
        latest_eval = st.session_state.evaluation_results[-1]
        latest_results = latest_eval['results']
        
        # إعداد البيانات للرسم
        model_data = {}
        for result in latest_results:
            model_data[result['model']] = [
                result['rouge1'], result['rouge2'], result['rougeL']
            ]
        
        # رسم الأعمدة
        width = 0.8 / len(model_data)
        colors = ['#667eea', '#764ba2', '#00D084', '#FF6B6B']
        
        bars = []
        for i, (model_name, scores) in enumerate(model_data.items()):
            bar_positions = [pos + i * width for pos in x_pos]
            bar = ax.bar(bar_positions, scores, width, label=model_name, 
                        color=colors[i % len(colors)])
            bars.append(bar)
            
            # إضافة القيم فوق الأعمدة
            for j, score in enumerate(scores):
                ax.text(bar_positions[j], score, f'{score:.3f}', 
                       ha='center', va='bottom', fontsize=8)
    else:
        # بيانات افتراضية
        tfidf_scores = [0.259, 0.091, 0.185]
        textrank_scores = [0.287, 0.087, 0.185]
        hybrid_scores = [0.333, 0.129, 0.213] if hybrid_available else None
        
        if hybrid_available:
            width = 0.25
            bars1 = ax.bar([i - width for i in x_pos], tfidf_scores, width, label='TF-IDF', color='#667eea')
            bars2 = ax.bar([i for i in x_pos], textrank_scores, width, label='TextRank', color='#764ba2')
            bars3 = ax.bar([i + width for i in x_pos], hybrid_scores, width, label='Hybrid DL', color='#00D084')
            bars = [bars1, bars2, bars3]
            
            # إضافة القيم
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            for bar in bars3:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            width = 0.35
            bars1 = ax.bar([i - width/2 for i in x_pos], tfidf_scores, width, label='TF-IDF', color='#667eea')
            bars2 = ax.bar([i + width/2 for i in x_pos], textrank_scores, width, label='TextRank', color='#764ba2')
            bars = [bars1, bars2]
            
            # إضافة القيم
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('الدرجة')
    ax.set_title('مقارنة أداء نماذج التلخيص')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
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
                st.caption(f"⏱️ {item['time']:.2f} ثانية")
    else:
        st.caption("لا توجد ملخصات سابقة")

# تذييل الصفحة
st.markdown("""
<div class="footer">
    <p>🚀 تم تطويره باستخدام Streamlit | 📝 ملخص النصوص الذكي v2.0.0</p>
</div>
""", unsafe_allow_html=True)