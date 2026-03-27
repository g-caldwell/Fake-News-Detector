import os
import sys
import random
import csv
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


# ---------- CONFIG ----------
os.environ["QT_API"] = "PyQt6"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(BASE_DIR, "FakeNewsIcon.png")
SPINNER_PATH = os.path.join(BASE_DIR, "preview.gif")


# ---------- THEMES ----------
DARK_COLORS = {
    "bg": "#121212",
    "text": "#FFFFFF",
    "chart_bg": "#1E1E1E",
    "pie_real": "#2ECC71",   # green = real
    "pie_fake": "#E74C3C",   # red = fake
}

LIGHT_COLORS = {
    "bg": "#FFFFFF",
    "text": "#000000",
    "chart_bg": "#FFFFFF",
    "pie_real": "#27AE60",
    "pie_fake": "#C0392B",
}


# ---------- FAKE MODEL ----------
def fake_predict(text: str):
    if not text.strip():
        return "Unknown", (0.0, 0.0), Counter()

    fake_keywords = ["conspiracy", "hoax", "clickbait", "shocking", "exposed"]
    score = sum(text.lower().count(k) for k in fake_keywords) + random.random()

    fake_prob = min(0.95, 0.3 + score * 0.1)
    real_prob = 1.0 - fake_prob
    label = "Fake" if fake_prob >= 0.5 else "Real"

    words = [w.strip(".,!?;:()[]\"'").lower() for w in text.split()]
    words = [w for w in words if len(w) > 3]
    word_counts = Counter(words).most_common(5)

    return label, (real_prob, fake_prob), word_counts


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

        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.details_tab, "Details")

        self.build_dashboard_tab()
        self.build_details_tab()

        self.apply_theme()

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

    # ---------- PIE CHART ----------
    def create_pie_chart(self, real_prob, fake_prob):
        fig = Figure(facecolor=self.colors["chart_bg"])
        ax = fig.add_subplot(111)

        labels = ["Real", "Fake"]
        values = [real_prob, fake_prob]
        colors = [self.colors["pie_real"], self.colors["pie_fake"]]

        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            pctdistance=0.7,
            labeldistance=1.05,
            textprops={"color": self.colors["text"], "fontsize": 12, "fontweight": "bold"},
            wedgeprops={"linewidth": 1.5, "edgecolor": self.colors["bg"]},
        )

        ax.set_title("Prediction Confidence", color=self.colors["text"], fontsize=14, pad=10)

        canvas = FigureCanvasQTAgg(fig)
        return canvas, ax, fig

    def update_pie_chart(self, real_prob, fake_prob):
        self.pie_ax.clear()

        labels = ["Real", "Fake"]
        values = [real_prob, fake_prob]
        colors = [self.colors["pie_real"], self.colors["pie_fake"]]

        self.pie_fig.set_facecolor(self.colors["chart_bg"])

        self.pie_ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            pctdistance=0.7,
            labeldistance=1.05,
            textprops={"color": self.colors["text"], "fontsize": 12, "fontweight": "bold"},
            wedgeprops={"linewidth": 1.5, "edgecolor": self.colors["bg"]},
        )

        self.pie_ax.set_title("Prediction Confidence", color=self.colors["text"], fontsize=14, pad=10)

        for text in self.pie_ax.texts:
            text.set_color(self.colors["text"])

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

        self.show_spinner(True)
        QTimer.singleShot(900, lambda: self.finish_detection(text))

    def finish_detection(self, text):
        label, (real_prob, fake_prob), word_counts = fake_predict(text)

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
        if path:
            real_count = 0
            fake_count = 0
            total = 0
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    if "text" not in reader.fieldnames:
                        self.status_label.setText("CSV must have a 'text' column.")
                        return
                    for row in reader:
                        t = row["text"]
                        label, _, _ = fake_predict(t)
                        total += 1
                        if label == "Real":
                            real_count += 1
                        else:
                            fake_count += 1
                self.status_label.setText(
                    f"CSV processed: {total} rows — Real: {real_count}, Fake: {fake_count}"
                )
            except Exception as e:
                self.status_label.setText(f"Error reading CSV: {e}")

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
                QWidget { background-color: #121212; color: #FFFFFF; }
                QPlainTextEdit { background-color: #1E1E1E; color: #FFFFFF; }
                QTextEdit { background-color: #1E1E1E; color: #FFFFFF; }
                QPushButton { background-color: #6C5CE7; color: #FFFFFF; border-radius: 6px; padding: 6px 10px; }
                QPushButton:hover { background-color: #A29BFE; }
                QProgressBar {
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                    border: 1px solid #333333;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #2ECC71;
                    border-radius: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget { background-color: #FFFFFF; color: #000000; }
                QPlainTextEdit { background-color: #FFFFFF; color: #000000; }
                QTextEdit { background-color: #FFFFFF; color: #000000; }
                QPushButton { background-color: #0984E3; color: #FFFFFF; border-radius: 6px; padding: 6px 10px; }
                QPushButton:hover { background-color: #74B9FF; }
                QProgressBar {
                    background-color: #F0F0F0;
                    color: #000000;
                    border: 1px solid #CCCCCC;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #27AE60;
                    border-radius: 5px;
                }
            """)


# ---------- RUN ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FakeNewsDashboard()
    window.show()
    sys.exit(app.exec())