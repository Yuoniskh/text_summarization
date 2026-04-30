# gui_app.py - النسخة المصححة
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

from src.summarization import ExtractiveSummarizer
from src.utils import split_sentences
import config

class SummarizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📝 ملخص النصوص الذكي")
        self.root.geometry("1200x800")
        
        # تعيين الأيقونة (اختياري)
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # إعداد الألوان والأنماط
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'bg': '#f5f7fa',
            'text': '#2c3e50',
            'success': '#27ae60',
            'warning': '#f39c12',
            'error': '#e74c3c',
            'white': '#ffffff',
            'border': '#e0e0e0'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # بناء الواجهة أولاً
        self.setup_ui()
        
        # ثم تحميل النماذج (بعد إنشاء status_label)
        self.load_models()
        
        # سجل الملخصات
        self.history = []
        
    def load_models(self):
        """تحميل نماذج التلخيص."""
        self.status_label.config(text="🔄 جاري تحميل النماذج...")
        self.root.update()
        
        # تحميل في خيط منفصل حتى لا تتجمد الواجهة
        def load():
            try:
                self.tfidf_model = ExtractiveSummarizer(method='tfidf')
                self.root.after(0, lambda: self.status_label.config(text="✅ TF-IDF جاهز"))
                
                self.textrank_model = ExtractiveSummarizer(method='textrank')
                self.root.after(0, lambda: self.status_label.config(text="✅ جميع النماذج جاهزة"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"❌ خطأ: {str(e)[:50]}"))
                self.root.after(0, lambda: messagebox.showerror("خطأ", f"فشل تحميل النماذج:\n{e}"))
        
        threading.Thread(target=load, daemon=True).start()
        
    def setup_ui(self):
        """بناء واجهة المستخدم."""
        
        # ============================================================
        # الإطار العلوي - العنوان
        # ============================================================
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="📝 ملخص النصوص الذكي",
            font=('Arial', 24, 'bold'),
            bg=self.colors['primary'],
            fg=self.colors['white']
        )
        title_label.pack(pady=20)
        
        # ============================================================
        # الإطار الرئيسي
        # ============================================================
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # ============================================================
        # اللوحة اليسرى - الإدخال والإعدادات
        # ============================================================
        left_panel = tk.Frame(main_frame, bg=self.colors['white'], relief='solid', bd=1)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # عنوان اللوحة
        panel_title = tk.Label(
            left_panel,
            text="📄 النص الأصلي",
            font=('Arial', 14, 'bold'),
            bg=self.colors['white'],
            fg=self.colors['text']
        )
        panel_title.pack(pady=10)
        
        # منطقة النص
        self.text_input = scrolledtext.ScrolledText(
            left_panel,
            wrap=tk.WORD,
            font=('Arial', 11),
            bg=self.colors['white'],
            fg=self.colors['text'],
            height=20,
            relief='solid',
            bd=1
        )
        self.text_input.pack(fill='both', expand=True, padx=10, pady=5)
        
        # نص افتراضي
        default_text = """Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents. AI is used in many applications such as machine translation, sentiment analysis, chatbots, and text summarization. Extractive summarization works by selecting the most important sentences from the original text, while abstractive summarization generates new sentences."""
        self.text_input.insert('1.0', default_text)
        
        # إحصائيات النص
        stats_frame = tk.Frame(left_panel, bg=self.colors['white'])
        stats_frame.pack(fill='x', padx=10, pady=10)
        
        self.words_label = tk.Label(stats_frame, text="الكلمات: 0", bg=self.colors['white'], fg=self.colors['text'])
        self.words_label.pack(side='left', padx=5)
        
        self.sentences_label = tk.Label(stats_frame, text="الجمل: 0", bg=self.colors['white'], fg=self.colors['text'])
        self.sentences_label.pack(side='left', padx=5)
        
        # تحديث الإحصائيات عند تغيير النص
        self.text_input.bind('<KeyRelease>', self.update_stats)
        self.update_stats()
        
        # أزرار التحكم
        button_frame = tk.Frame(left_panel, bg=self.colors['white'])
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(
            button_frame,
            text="📁 تحميل ملف",
            command=self.load_file,
            bg=self.colors['secondary'],
            fg=self.colors['white'],
            font=('Arial', 10),
            relief='flat',
            padx=15,
            pady=5,
            cursor='hand2'
        ).pack(side='left', padx=5)
        
        tk.Button(
            button_frame,
            text="🗑️ مسح",
            command=self.clear_text,
            bg=self.colors['error'],
            fg=self.colors['white'],
            font=('Arial', 10),
            relief='flat',
            padx=15,
            pady=5,
            cursor='hand2'
        ).pack(side='left', padx=5)
        
        # ============================================================
        # اللوحة الوسطى - الإعدادات
        # ============================================================
        middle_panel = tk.Frame(main_frame, bg=self.colors['white'], relief='solid', bd=1, width=250)
        middle_panel.pack(side='left', fill='y', padx=10)
        middle_panel.pack_propagate(False)
        
        # عنوان
        tk.Label(
            middle_panel,
            text="⚙️ الإعدادات",
            font=('Arial', 14, 'bold'),
            bg=self.colors['white'],
            fg=self.colors['text']
        ).pack(pady=10)
        
        # اختيار النموذج
        tk.Label(
            middle_panel,
            text="اختر النموذج:",
            bg=self.colors['white'],
            fg=self.colors['text']
        ).pack(pady=(20, 5))
        
        self.model_var = tk.StringVar(value="textrank")
        
        model_frame = tk.Frame(middle_panel, bg=self.colors['white'])
        model_frame.pack()
        
        tk.Radiobutton(
            model_frame,
            text=" TextRank (أفضل)",
            variable=self.model_var,
            value="textrank",
            bg=self.colors['white'],
            font=('Arial', 10)
        ).pack(anchor='w', pady=2)
        
        tk.Radiobutton(
            model_frame,
            text="TF-IDF (أسرع)",
            variable=self.model_var,
            value="tfidf",
            bg=self.colors['white'],
            font=('Arial', 10)
        ).pack(anchor='w', pady=2)
        
        # عدد الجمل
        tk.Label(
            middle_panel,
            text="عدد جمل الملخص:",
            bg=self.colors['white'],
            fg=self.colors['text']
        ).pack(pady=(20, 5))
        
        self.sentences_spinbox = tk.Spinbox(
            middle_panel,
            from_=1,
            to=10,
            width=10,
            font=('Arial', 10),
            state='readonly'
        )
        self.sentences_spinbox.pack()
        self.sentences_spinbox.delete(0, 'end')
        self.sentences_spinbox.insert(0, '3')
        
        # زر التلخيص
        self.summarize_btn = tk.Button(
            middle_panel,
            text=" تلخيص",
            command=self.summarize,
            bg=self.colors['primary'],
            fg=self.colors['white'],
            font=('Arial', 12, 'bold'),
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.summarize_btn.pack(pady=30)
        
        # مقارنة النماذج
        tk.Button(
            middle_panel,
            text="📊 مقارنة النماذج",
            command=self.compare_models,
            bg=self.colors['secondary'],
            fg=self.colors['white'],
            font=('Arial', 10),
            relief='flat',
            padx=15,
            pady=5,
            cursor='hand2'
        ).pack(pady=10)
        
        # ============================================================
        # اللوحة اليمنى - الملخص
        # ============================================================
        right_panel = tk.Frame(main_frame, bg=self.colors['white'], relief='solid', bd=1)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        tk.Label(
            right_panel,
            text="✨ الملخص الناتج",
            font=('Arial', 14, 'bold'),
            bg=self.colors['white'],
            fg=self.colors['text']
        ).pack(pady=10)
        
        self.summary_output = scrolledtext.ScrolledText(
            right_panel,
            wrap=tk.WORD,
            font=('Arial', 11),
            bg='#f8f9fa',
            fg=self.colors['text'],
            height=20,
            relief='solid',
            bd=1
        )
        self.summary_output.pack(fill='both', expand=True, padx=10, pady=5)
        
        # إحصائيات الملخص
        summary_stats_frame = tk.Frame(right_panel, bg=self.colors['white'])
        summary_stats_frame.pack(fill='x', padx=10, pady=10)
        
        self.summary_words_label = tk.Label(summary_stats_frame, text="الكلمات: 0", bg=self.colors['white'], fg=self.colors['text'])
        self.summary_words_label.pack(side='left', padx=5)
        
        self.compression_label = tk.Label(summary_stats_frame, text="نسبة الضغط: 0%", bg=self.colors['white'], fg=self.colors['text'])
        self.compression_label.pack(side='left', padx=5)
        
        self.time_label = tk.Label(summary_stats_frame, text="الوقت: 0.00 ث", bg=self.colors['white'], fg=self.colors['text'])
        self.time_label.pack(side='left', padx=5)
        
        # أزرار الملخص
        summary_btn_frame = tk.Frame(right_panel, bg=self.colors['white'])
        summary_btn_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(
            summary_btn_frame,
            text="📋 نسخ",
            command=self.copy_summary,
            bg='#95a5a6',
            fg=self.colors['white'],
            font=('Arial', 10),
            relief='flat',
            padx=15,
            pady=5,
            cursor='hand2'
        ).pack(side='left', padx=5)
        
        tk.Button(
            summary_btn_frame,
            text="💾 حفظ",
            command=self.save_summary,
            bg=self.colors['success'],
            fg=self.colors['white'],
            font=('Arial', 10),
            relief='flat',
            padx=15,
            pady=5,
            cursor='hand2'
        ).pack(side='left', padx=5)
        
        # ============================================================
        # شريط الحالة
        # ============================================================
        status_frame = tk.Frame(self.root, bg=self.colors['border'], height=30)
        status_frame.pack(fill='x', side='bottom')
        
        self.status_label = tk.Label(
            status_frame,
            text="✅ جاهز",
            bg=self.colors['border'],
            fg=self.colors['text'],
            font=('Arial', 9)
        )
        self.status_label.pack(side='left', padx=10, pady=5)
        
    def update_stats(self, event=None):
        """تحديث إحصائيات النص المدخل."""
        text = self.text_input.get('1.0', 'end-1c')
        words = len(text.split())
        sentences = len(split_sentences(text))
        
        self.words_label.config(text=f"الكلمات: {words}")
        self.sentences_label.config(text=f"الجمل: {sentences}")
        
    def load_file(self):
        """تحميل ملف نصي."""
        file_path = filedialog.askopenfilename(
            title="اختر ملف نصي",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.text_input.delete('1.0', 'end')
                self.text_input.insert('1.0', content)
                self.update_stats()
                self.status_label.config(text=f"✅ تم تحميل: {file_path.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("خطأ", f"فشل تحميل الملف:\n{e}")
                
    def clear_text(self):
        """مسح النص المدخل."""
        self.text_input.delete('1.0', 'end')
        self.summary_output.delete('1.0', 'end')
        self.update_stats()
        self.status_label.config(text="🗑️ تم المسح")
        
    def summarize(self):
        """تلخيص النص."""
        text = self.text_input.get('1.0', 'end-1c').strip()
        
        if not text:
            messagebox.showwarning("تحذير", "الرجاء إدخال نص للتلخيص")
            return
        
        # التحقق من تحميل النماذج
        if not hasattr(self, 'tfidf_model') or not hasattr(self, 'textrank_model'):
            messagebox.showwarning("انتظر", "جاري تحميل النماذج... الرجاء الانتظار")
            return
        
        # تعطيل الزر أثناء التلخيص
        self.summarize_btn.config(state='disabled', text="⏳ جاري التلخيص...")
        self.status_label.config(text="🔄 جاري التلخيص...")
        self.root.update()
        
        # التلخيص في خيط منفصل
        def run_summarization():
            try:
                start_time = time.time()
                
                num_sentences = int(self.sentences_spinbox.get())
                model_choice = self.model_var.get()
                
                if model_choice == 'textrank':
                    summary = self.textrank_model.summarize(text, num_sentences)
                    model_name = "TextRank"
                else:
                    summary = self.tfidf_model.summarize(text, num_sentences)
                    model_name = "TF-IDF"
                
                elapsed_time = time.time() - start_time
                
                # تحديث الواجهة في الخيط الرئيسي
                self.root.after(0, self.update_summary_result, summary, text, elapsed_time, model_name)
                
            except Exception as e:
                self.root.after(0, self.show_summary_error, str(e))
        
        threading.Thread(target=run_summarization, daemon=True).start()
        
    def update_summary_result(self, summary, original_text, elapsed_time, model_name):
        """تحديث واجهة الملخص بعد الانتهاء."""
        self.summary_output.delete('1.0', 'end')
        self.summary_output.insert('1.0', summary)
        
        # تحديث الإحصائيات
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        compression = (1 - summary_words / original_words) * 100 if original_words > 0 else 0
        
        self.summary_words_label.config(text=f"الكلمات: {summary_words}")
        self.compression_label.config(text=f"نسبة الضغط: {compression:.1f}%")
        self.time_label.config(text=f"الوقت: {elapsed_time:.2f} ث")
        
        # إضافة إلى السجل
        self.history.append({
            'text': original_text[:100] + "...",
            'summary': summary,
            'model': model_name,
            'time': elapsed_time
        })
        
        # إعادة تفعيل الزر
        self.summarize_btn.config(state='normal', text="🚀 تلخيص")
        self.status_label.config(text=f"✅ اكتمل التلخيص باستخدام {model_name} في {elapsed_time:.2f} ثانية")
        
    def show_summary_error(self, error_msg):
        """عرض خطأ التلخيص."""
        self.summarize_btn.config(state='normal', text="🚀 تلخيص")
        self.status_label.config(text=f"❌ خطأ: {error_msg[:50]}")
        messagebox.showerror("خطأ في التلخيص", error_msg)
        
    def copy_summary(self):
        """نسخ الملخص إلى الحافظة."""
        summary = self.summary_output.get('1.0', 'end-1c')
        if summary.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(summary)
            self.status_label.config(text="📋 تم نسخ الملخص إلى الحافظة")
        else:
            messagebox.showwarning("تحذير", "لا يوجد ملخص لنسخه")
            
    def save_summary(self):
        """حفظ الملخص إلى ملف."""
        summary = self.summary_output.get('1.0', 'end-1c')
        if not summary.strip():
            messagebox.showwarning("تحذير", "لا يوجد ملخص لحفظه")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="حفظ الملخص",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                self.status_label.config(text=f" تم حفظ الملخص: {file_path.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("خطأ", f"فشل حفظ الملف:\n{e}")
                
    def compare_models(self):
        """فتح نافذة مقارنة النماذج."""
        text = self.text_input.get('1.0', 'end-1c').strip()
        
        if not text:
            messagebox.showwarning("تحذير", "الرجاء إدخال نص للمقارنة")
            return
        
        # التحقق من تحميل النماذج
        if not hasattr(self, 'tfidf_model') or not hasattr(self, 'textrank_model'):
            messagebox.showwarning("انتظر", "جاري تحميل النماذج... الرجاء الانتظار")
            return
            
        # فتح نافذة جديدة
        compare_window = tk.Toplevel(self.root)
        compare_window.title(" مقارنة النماذج")
        compare_window.geometry("800x600")
        compare_window.configure(bg=self.colors['bg'])
        
        # عنوان
        tk.Label(
            compare_window,
            text="TF-IDF و TextRank",
            font=('Arial', 16, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['text']
        ).pack(pady=20)
        
        # إطار للملخصات
        summaries_frame = tk.Frame(compare_window, bg=self.colors['bg'])
        summaries_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # ملخص TF-IDF
        tfidf_frame = tk.LabelFrame(summaries_frame, text="TF-IDF", bg=self.colors['white'], font=('Arial', 12, 'bold'))
        tfidf_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        tfidf_text = scrolledtext.ScrolledText(tfidf_frame, wrap=tk.WORD, height=10, font=('Arial', 10))
        tfidf_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # ملخص TextRank
        textrank_frame = tk.LabelFrame(summaries_frame, text="TextRank", bg=self.colors['white'], font=('Arial', 12, 'bold'))
        textrank_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        textrank_text = scrolledtext.ScrolledText(textrank_frame, wrap=tk.WORD, height=10, font=('Arial', 10))
        textrank_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # زر التوليد
        def generate_comparison():
            num_sentences = int(self.sentences_spinbox.get())
            
            tfidf_summary = self.tfidf_model.summarize(text, num_sentences)
            textrank_summary = self.textrank_model.summarize(text, num_sentences)
            
            tfidf_text.delete('1.0', 'end')
            tfidf_text.insert('1.0', tfidf_summary)
            
            textrank_text.delete('1.0', 'end')
            textrank_text.insert('1.0', textrank_summary)
            
        tk.Button(
            compare_window,
            text="🔄 توليد المقارنة",
            command=generate_comparison,
            bg=self.colors['primary'],
            fg=self.colors['white'],
            font=('Arial', 11),
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        ).pack(pady=10)
        
        generate_comparison()
        
        # رسم بياني
        fig, ax = plt.subplots(figsize=(8, 4))
        models = ['TF-IDF', 'TextRank']
        scores = [0.259, 0.287]
        
        bars = ax.bar(models, scores, color=[self.colors['primary'], self.colors['secondary']])
        ax.set_ylabel('ROUGE-1 Score')
        ax.set_title('compare')
        ax.set_ylim(0, 0.35)
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        canvas = FigureCanvasTkAgg(fig, master=compare_window)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

def main():
    root = tk.Tk()
    app = SummarizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()