import os
import sys
import random
import csv
import re
from collections import Counter

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QPushButton, QLabel, QFileDialog, QTabWidget,
    QFrame, QTextEdit, QProgressBar
)
from PyQt6.QtGui import QIcon, QMovie, QColor, QTextCharFormat
from PyQt6.QtCore import Qt, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.normpath(os.path.join(base_path, relative_path))

# ---------- CONFIG ----------
os.environ["QT_API"] = "PyQt6"
# In bundled mode, BASE_DIR will be in _MEIPASS. In dev mode, it's the GUI folder.
if hasattr(sys, '_MEIPASS'):
    BASE_DIR = resource_path('GUI')
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to sys.path for importing Model
sys.path.append(os.path.dirname(BASE_DIR))
from Model.clean import clean, clean_csv
import pandas as pd
import tempfile

from Model.model import load_model_and_vectorizer, preprocess_text, get_important_keywords

# Update paths to use resource_path
ASSETS_DIR = resource_path(os.path.join('GUI', 'Assets'))
ICON_PATH = os.path.join(ASSETS_DIR, 'FakeNewsIcon.png')
SPINNER_PATH = os.path.join(ASSETS_DIR, 'preview.gif')

# Add bundled NLTK data path
import nltk
NLTK_DATA_PATH = resource_path('nltk_data')
if os.path.exists(NLTK_DATA_PATH):
    nltk.data.path.append(NLTK_DATA_PATH)


# ---------- THEMES ----------
DARK_COLORS = {
    "bg": "#0A0A0B",         # Very deep dark for maximum contrast
    "text": "#FAFAFA",       # Near-pure white for readability
    "chart_bg": "#0A0A0B",   # Seamless with background
    "pie_real": "#00E676",   # Vivid green
    "pie_fake": "#FF5252",   # Vivid red
}

LIGHT_COLORS = {
    "bg": "#F5F6F7",         # Clean light grey/white
    "text": "#1A1A1A",       # Dark grey/black
    "chart_bg": "#F5F6F7",
    "pie_real": "#2E7D32",   # Forest green
    "pie_fake": "#D32F2F",   # Deep red
}


# Model prediction helper
def real_predict(text: str, model, vectorizer, important_keywords):
    if not text.strip():
        return "Unknown", (0.0, 0.0), []

    # 1. Clean
    cleaned_text = clean(text)
    
    # 2. Preprocess
    preprocessed = preprocess_text([cleaned_text])
    
    # 3. Vectorize
    vec = vectorizer.transform(preprocessed)
    
    # 4. Predict
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    class_map = {c: i for i, c in enumerate(classes)}
    
    # Assuming 0 is Fake, 1 is Real
    fake_prob = probs[class_map.get(0, 0)]
    real_prob = probs[class_map.get(1, 1)]
    
    label = "Real" if real_prob > fake_prob else "Fake"

    # Find important keywords present in the text
    words_in_text = [w.lower() for w in re.sub(r'[^\w\s]', '', text).split()]
    word_counts = Counter(words_in_text)
    matches = [(w, word_counts[w]) for w in important_keywords if w in word_counts]
    matches.sort(key=lambda x: x[1], reverse=True)

    return label, (real_prob, fake_prob), matches[:10]


# ---------- SUMMARY GENERATOR ----------
def summarize_text(text: str):
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return "Not enough content to summarize."

    summary = sentences[:3]
    return ". ".join(summary) + "."


# ---------- EXTRA ANALYTICS ----------
def compute_sentiment(text: str) -> float:
    positive_words = ["good", "great", "positive", "benefit", "safe", "trust"]
    negative_words = ["bad", "terrible", "danger", "risk", "hoax", "fake"]

    t = text.lower()
    pos = sum(t.count(w) for w in positive_words)
    neg = sum(t.count(w) for w in negative_words)
    total = pos + neg
    if total == 0:
        return 0.5
    return max(0.0, min(1.0, (pos + 0.5) / (total + 1.0)))


def compute_readability(text: str) -> float:
    sentences = [s for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    words = text.split()
    if not sentences or not words:
        return 0.0
    avg_words_per_sentence = len(words) / len(sentences)
    score = max(0.0, min(1.0, 1.5 - (avg_words_per_sentence / 30.0)))
    return score


def explain_prediction(label: str, real_prob: float, fake_prob: float, word_counts):
    if label == "Real":
        base = "The article is likely real because the overall language and structure appear consistent and measured."
        prob_part = f" The model estimates a {real_prob:.1%} chance of being real versus {fake_prob:.1%} fake."
    else:
        base = "The article is likely fake because it contains patterns often seen in misleading or sensational content."
        prob_part = f" The model estimates a {fake_prob:.1%} chance of being fake versus {real_prob:.1%} real."

    if word_counts:
        top_words = ", ".join([w for w, _ in word_counts])
        words_part = f" Notably, the following words appeared frequently: {top_words}."
    else:
        words_part = " No particularly dominant keywords were detected."

    return base + prob_part + words_part


# ---------- MAIN WINDOW ----------
class FakeNewsDashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fake News Detector")
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))
        self.resize(1200, 750)

        self.dark_mode = True
        self.colors = DARK_COLORS

        central = QWidget()
        self.setCentralWidget(central)

        self.main_layout = QHBoxLayout(central)

        # Sidebar
        self.sidebar = self.build_sidebar()
        self.main_layout.addWidget(self.sidebar)

        # Tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs, stretch=1)

        self.dashboard_tab = QWidget()
        self.details_tab = QWidget()
        self.about_tab = QWidget()

        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.details_tab, "Details")
        self.tabs.addTab(self.about_tab, "About")

        self.build_dashboard_tab()
        self.build_details_tab()
        self.build_about_tab()

        self.apply_theme()

        # Load Model
        self.status_label.setText("Loading model...")
        QTimer.singleShot(100, self.load_model_async)

    def load_model_async(self):
        try:
            self.model, self.vectorizer = load_model_and_vectorizer()
            self.important_keywords = get_important_keywords(self.model, self.vectorizer)
            self.status_label.setText("Ready")
        except Exception as e:
            self.status_label.setText(f"Error loading model: {e}")
            self.model, self.vectorizer = None, None

    # ---------- SIDEBAR ----------
    def build_sidebar(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Fake News Detector")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)

        layout.addSpacing(10)

        btn_dash = QPushButton("Dashboard")
        btn_dash.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        layout.addWidget(btn_dash)

        btn_details = QPushButton("Details")
        btn_details.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        layout.addWidget(btn_details)

        btn_about = QPushButton("About")
        btn_about.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        layout.addWidget(btn_about)

        layout.addSpacing(20)

        btn_theme = QPushButton("Toggle Dark/Light")
        btn_theme.clicked.connect(self.toggle_theme)
        layout.addWidget(btn_theme)

        layout.addStretch()

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        return frame

    # ---------- DASHBOARD TAB ----------
    def build_dashboard_tab(self):
        layout = QGridLayout(self.dashboard_tab)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # PIE CHART (bigger area)
        self.pie_canvas, self.pie_ax, self.pie_fig = self.create_pie_chart(0.5, 0.5)
        layout.addWidget(self.pie_canvas, 0, 0, 2, 2)

        # BAR CHART FOR TOP WORDS (next to pie)
        self.bar_canvas, self.bar_ax, self.bar_fig = self.create_bar_chart([])
        layout.addWidget(self.bar_canvas, 0, 2, 2, 2)

        # TEXT INPUT
        self.text_box = QPlainTextEdit()
        self.text_box.setPlaceholderText("Paste news article text here...")
        layout.addWidget(self.text_box, 2, 0, 1, 4)

        # BUTTONS
        btn_row = QHBoxLayout()

        btn_import = QPushButton("Import Text File")
        btn_import.clicked.connect(self.import_text_file)
        btn_row.addWidget(btn_import)

        btn_csv = QPushButton("Upload CSV")
        btn_csv.clicked.connect(self.upload_csv)
        btn_row.addWidget(btn_csv)

        btn_row.addStretch()

        btn_detect = QPushButton("Detect")
        btn_detect.clicked.connect(self.run_detection)
        btn_row.addWidget(btn_detect)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.text_box.clear)
        btn_row.addWidget(btn_clear)

        layout.addLayout(btn_row, 3, 0, 1, 4)

        # SENTIMENT METER
        self.sentiment_bar = QProgressBar()
        self.sentiment_bar.setRange(0, 100)
        self.sentiment_bar.setFormat("Sentiment: %p% (higher = more positive)")
        layout.addWidget(self.sentiment_bar, 4, 0, 1, 4)

        # LOADING SPINNER
        self.spinner_label = QLabel()
        self.spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.exists(SPINNER_PATH):
            self.spinner_movie = QMovie(SPINNER_PATH)
            self.spinner_label.setMovie(self.spinner_movie)
        else:
            self.spinner_movie = None
            self.spinner_label.setText("Processing...")

        self.spinner_label.setVisible(False)
        layout.addWidget(self.spinner_label, 5, 0, 1, 4)

        # CSV SUMMARY LABEL (Bottom Left)
        self.csv_summary_label = QLabel("")
        self.csv_summary_label.setStyleSheet("font-weight: bold; color: #2ECC71;")
        layout.addWidget(self.csv_summary_label, 6, 0, 1, 2)

    # ---------- DETAILS TAB ----------
    def build_details_tab(self):
        layout = QVBoxLayout(self.details_tab)
        layout.setContentsMargins(30, 30, 30, 30)

        header = QLabel("Prediction Details")
        header.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(header)

        self.details_label = QLabel("Run a detection to see details.")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

        # WORD COUNT + READABILITY
        self.word_count_label = QLabel("Word count: 0")
        layout.addWidget(self.word_count_label)

        self.readability_label = QLabel("Readability: N/A")
        layout.addWidget(self.readability_label)

        layout.addSpacing(20)

        summary_header = QLabel("Summary")
        summary_header.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(summary_header)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        layout.addWidget(self.summary_box)

        explanation_header = QLabel("AI Explanation")
        explanation_header.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(explanation_header)

        self.explanation_box = QTextEdit()
        self.explanation_box.setReadOnly(True)
        layout.addWidget(self.explanation_box)

        layout.addStretch()

    # ---------- ABOUT TAB ----------
    def build_about_tab(self):
        layout = QVBoxLayout(self.about_tab)
        layout.setContentsMargins(30, 30, 30, 30)

        header = QLabel("About This Project")
        header.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(header)

        layout.addSpacing(10)

        description = QLabel(
            "This Fake News Detector is a machine learning-based application designed to classify "
            "news articles as either Real or Fake. By analyzing patterns in text, assessing readability, "
            "and evaluating sentiment, our tool provides valuable insights to help users identify potentially "
            "misleading or sensational content online."
        )
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 16px;")
        layout.addWidget(description)

        layout.addSpacing(20)

        team_header = QLabel("Project Team:")
        team_header.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(team_header)

        layout.addSpacing(5)

        team = QLabel("• Carter Wilson\n• Tyler Wills\n• Gabriel Caldwell\n• Peyton Hollis")
        team.setStyleSheet("font-size: 16px;")
        layout.addWidget(team)

        layout.addSpacing(20)

        source_header = QLabel("Source / Inspiration:")
        source_header.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(source_header)

        layout.addSpacing(5)

        source_link = QLabel(
            "Source: <a href='https://www.geeksforgeeks.org/machine-learning/fake-news-detection-using-machine-learning/'>"
            "https://www.geeksforgeeks.org/machine-learning/fake-news-detection-using-machine-learning/</a>"
        )
        source_link.setOpenExternalLinks(True)
        source_link.setStyleSheet("font-size: 16px;")
        layout.addWidget(source_link)

        layout.addStretch()

    # ---------- PIE (DONUT) CHART ----------
    def create_pie_chart(self, real_prob, fake_prob):
        fig = Figure(facecolor=self.colors["chart_bg"])
        ax = fig.add_subplot(111)

        labels = ["Real", "Fake"]
        values = [real_prob, fake_prob]
        colors = [self.colors["pie_real"], self.colors["pie_fake"]]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            pctdistance=0.75,
            labeldistance=1.1,
            textprops={"color": self.colors["text"], "fontsize": 14, "fontweight": "bold"},
            wedgeprops={"linewidth": 2, "edgecolor": self.colors["bg"], "width": 0.4},  # Donut style
        )

        ax.set_title("Prediction Confidence", color=self.colors["text"], fontsize=15, fontweight='bold', pad=15)

        # Enhance contrast for percentages
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(13)
            autotext.set_fontweight('bold')

        fig.tight_layout()
        canvas = FigureCanvasQTAgg(fig)
        return canvas, ax, fig

    def update_pie_chart(self, real_prob, fake_prob):
        self.pie_ax.clear()

        labels = ["Real", "Fake"]
        values = [real_prob, fake_prob]
        colors = [self.colors["pie_real"], self.colors["pie_fake"]]

        self.pie_fig.set_facecolor(self.colors["chart_bg"])

        wedges, texts, autotexts = self.pie_ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            pctdistance=0.75,
            labeldistance=1.1,
            textprops={"color": self.colors["text"], "fontsize": 14, "fontweight": "bold"},
            wedgeprops={"linewidth": 2, "edgecolor": self.colors["bg"], "width": 0.4},  # Donut style
        )

        self.pie_ax.set_title("Prediction Confidence", color=self.colors["text"], fontsize=15, fontweight='bold', pad=15)

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(13)
            autotext.set_fontweight('bold')

        self.pie_fig.tight_layout()
        self.pie_canvas.draw()

    # ---------- BAR CHART FOR TOP WORDS ----------
    def create_bar_chart(self, word_counts):
        fig = Figure(facecolor=self.colors["chart_bg"])
        ax = fig.add_subplot(111)

        words = [w for w, _ in word_counts]
        counts = [c for _, c in word_counts]

        ax.bar(words, counts, color=self.colors["pie_fake"])
        ax.set_title("Top Words", color=self.colors["text"], fontsize=14, pad=10)
        ax.tick_params(axis="x", labelrotation=30, labelcolor=self.colors["text"])
        ax.tick_params(axis="y", labelcolor=self.colors["text"])
        fig.tight_layout()

        canvas = FigureCanvasQTAgg(fig)
        return canvas, ax, fig

    def update_bar_chart(self, word_counts):
        self.bar_ax.clear()

        words = [w for w, _ in word_counts]
        counts = [c for _, c in word_counts]

        self.bar_fig.set_facecolor(self.colors["chart_bg"])
        self.bar_ax.bar(words, counts, color=self.colors["pie_fake"])
        self.bar_ax.set_title("Top Words", color=self.colors["text"], fontsize=14, pad=10)
        self.bar_ax.tick_params(axis="x", labelrotation=30, labelcolor=self.colors["text"])
        self.bar_ax.tick_params(axis="y", labelcolor=self.colors["text"])

        self.bar_fig.tight_layout()
        self.bar_canvas.draw()

    # ---------- KEYWORD HIGHLIGHTING ----------
    def highlight_keywords(self, word_counts):
        cursor = self.text_box.textCursor()
        cursor.select(cursor.SelectionType.Document)
        cursor.setCharFormat(QTextCharFormat())

        fmt = QTextCharFormat()
        fmt.setBackground(QColor("#FFF3B0"))

        text = self.text_box.toPlainText()
        lower_text = text.lower()
        for word, _ in word_counts:
            if not word:
                continue
            start = 0
            w = word.lower()
            while True:
                idx = lower_text.find(w, start)
                if idx == -1:
                    break
                cursor.setPosition(idx)
                cursor.movePosition(cursor.MoveOperation.Right, cursor.MoveMode.KeepAnchor, len(w))
                cursor.mergeCharFormat(fmt)
                start = idx + len(w)

    # ---------- DETECTION ----------
    def run_detection(self):
        text = self.text_box.toPlainText()
        if not text.strip():
            self.status_label.setText("Please enter text first.")
            return
        
        if not hasattr(self, 'model') or self.model is None:
            self.status_label.setText("Model not loaded yet.")
            return

        self.show_spinner(True)
        QTimer.singleShot(100, lambda: self.finish_detection(text))

    def finish_detection(self, text):
        label, (real_prob, fake_prob), word_counts = real_predict(
            text, self.model, self.vectorizer, self.important_keywords
        )

        # update pie chart
        self.update_pie_chart(real_prob, fake_prob)

        # update bar chart
        self.update_bar_chart(word_counts)

        # keyword highlighting
        self.highlight_keywords(word_counts)

        # update details tab
        details = [
            f"<b>Prediction:</b> {label}",
            f"<b>Real:</b> {real_prob:.2%}",
            f"<b>Fake:</b> {fake_prob:.2%}",
        ]
        self.details_label.setText("<br>".join(details))

        # word count
        words = [w for w in text.split() if w.strip()]
        self.word_count_label.setText(f"Word count: {len(words)}")

        # readability
        readability = compute_readability(text)
        self.readability_label.setText(f"Readability (0–1): {readability:.2f}")

        # summary
        summary = summarize_text(text)
        self.summary_box.setPlainText(summary)

        # AI explanation
        explanation = explain_prediction(label, real_prob, fake_prob, word_counts)
        self.explanation_box.setPlainText(explanation)

        # sentiment meter
        sentiment = compute_sentiment(text)
        self.sentiment_bar.setValue(int(sentiment * 100))

        self.status_label.setText(f"Prediction complete: {label}")
        self.show_spinner(False)

    # ---------- FILE IMPORT ----------
    def import_text_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt)")
        if path:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                self.text_box.setPlainText(f.read())
            self.status_label.setText(f"Loaded: {os.path.basename(path)}")

    def upload_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Upload CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        self.status_label.setText("Processing CSV...")
        self.show_spinner(True)
        
        # Run in a separate shot to avoid UI freeze
        QTimer.singleShot(100, lambda: self.process_csv_batch(path))

    def process_csv_batch(self, path):
        try:
            # 1. Create a temp file for cleaning
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp_path = tmp.name
            
            # 2. Clean use Model.clean logic
            if not clean_csv(path, tmp_path):
                self.status_label.setText("Error cleaning CSV.")
                self.show_spinner(False)
                return

            # 3. Load cleaned data
            df = pd.read_csv(tmp_path)
            if 'text' not in df.columns:
                self.status_label.setText("CSV missing 'text' column after cleaning.")
                self.show_spinner(False)
                return

            # 4. Predict in batch
            if not self.model or not self.vectorizer:
                self.status_label.setText("Model not loaded.")
                self.show_spinner(False)
                return

            texts = df['text'].astype(str).tolist()
            # Batch preprocess (it expects a list)
            preprocessed = preprocess_text(texts)
            # Batch vectorize
            vec = self.vectorizer.transform(preprocessed)
            # Batch predict
            probs = self.model.predict_proba(vec)
            classes = self.model.classes_
            class_map = {c: i for i, c in enumerate(classes)}
            
            fake_idx = class_map.get(0, 0)
            real_idx = class_map.get(1, 1)
            
            results = []
            real_count = 0
            fake_count = 0
            
            for p in probs:
                f_p = p[fake_idx]
                r_p = p[real_idx]
                label = "Real" if r_p > f_p else "Fake"
                results.append(label)
                if label == "Real":
                    real_count += 1
                else:
                    fake_count += 1

            df['prediction'] = results
            df['real_prob'] = probs[:, real_idx]
            df['fake_prob'] = probs[:, fake_idx]

            # 5. Update UI (Bottom Left)
            summary = (
                f"<b>Last CSV:</b> {os.path.basename(path)}<br>"
                f"<b>Total Rows:</b> {len(df)}<br>"
                f"<b>Real:</b> {real_count} | <b>Fake:</b> {fake_count}"
            )
            self.csv_summary_label.setText(summary)
            self.status_label.setText(f"Batch processing complete: {len(df)} rows")

            # 6. Prompt to save results
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Prediction Results", 
                f"results_{os.path.basename(path)}", 
                "CSV Files (*.csv)"
            )
            if save_path:
                df.to_csv(save_path, index=False)
                self.status_label.setText(f"Results saved to {os.path.basename(save_path)}")

            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        except Exception as e:
            self.status_label.setText(f"Batch Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.show_spinner(False)

    # ---------- SPINNER ----------
    def show_spinner(self, show):
        self.spinner_label.setVisible(show)
        if self.spinner_movie:
            if show:
                self.spinner_movie.start()
            else:
                self.spinner_movie.stop()

    # ---------- THEME ----------
    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.colors = DARK_COLORS if self.dark_mode else LIGHT_COLORS
        self.apply_theme()
        self.update_pie_chart(0.5, 0.5)
        self.update_bar_chart([])

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QWidget { background-color: #0A0A0B; color: #FAFAFA; font-family: 'Segoe UI', Arial; }
                QPlainTextEdit { background-color: #161618; color: #FAFAFA; border: 1px solid #333; border-radius: 8px; font-size: 14px; }
                QTextEdit { background-color: #161618; color: #FAFAFA; border: 1px solid #333; border-radius: 8px; }
                QPushButton { background-color: #6C5CE7; color: #FFFFFF; border-radius: 8px; padding: 10px 15px; font-weight: bold; border: none; }
                QPushButton:hover { background-color: #8C7CFF; }
                QPushButton:pressed { background-color: #5A4EBF; }
                QProgressBar {
                    background-color: #161618;
                    color: #FFFFFF;
                    border: 2px solid #333;
                    border-radius: 10px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #00E676;
                    border-radius: 8px;
                }
                QLabel { font-size: 14px; }
                QTabWidget::pane { border: 1px solid #333; background: #0A0A0B; }
                QTabBar::tab { background: #161618; color: #888; padding: 10px 20px; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px; }
                QTabBar::tab:selected { background: #6C5CE7; color: white; }
            """)
        else:
            self.setStyleSheet("""
                QWidget { background-color: #F5F6F7; color: #1A1A1A; font-family: 'Segoe UI', Arial; }
                QPlainTextEdit { background-color: #FFFFFF; color: #1A1A1A; border: 1px solid #DDD; border-radius: 8px; font-size: 14px; }
                QTextEdit { background-color: #FFFFFF; color: #1A1A1A; border: 1px solid #DDD; border-radius: 8px; }
                QPushButton { background-color: #0984E3; color: #FFFFFF; border-radius: 8px; padding: 10px 15px; font-weight: bold; border: none; }
                QPushButton:hover { background-color: #29AAFF; }
                QProgressBar {
                    background-color: #E0E0E0;
                    color: #000000;
                    border: 1px solid #CCC;
                    border-radius: 10px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #2E7D32;
                    border-radius: 10px;
                }
                QLabel { font-size: 14px; }
            """)


# ---------- RUN ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FakeNewsDashboard()
    window.show()
    sys.exit(app.exec())
