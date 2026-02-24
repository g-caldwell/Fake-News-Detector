import os
import sys
import random
from collections import Counter

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QPushButton, QLabel, QSizePolicy, QFileDialog, QTabWidget,
    QFrame, QSpacerItem, QTextEdit
)
from PyQt6.QtGui import QIcon, QMovie
from PyQt6.QtCore import Qt, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ---------- CONFIG ----------
os.environ["QT_API"] = "PyQt6"
ICON_PATH = "FakeNewsIcon.png"
SPINNER_PATH = "preview.gif"


# ---------- THEMES ----------
DARK_COLORS = {
    "bg": "#121212",
    "text": "#FFFFFF",
    "chart_bg": "#1E1E1E",
    "pie_real": "#74B9FF",
    "pie_fake": "#FF7675",
}

LIGHT_COLORS = {
    "bg": "#FFFFFF",
    "text": "#000000",
    "chart_bg": "#FFFFFF",
    "pie_real": "#0984E3",
    "pie_fake": "#D63031",
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

        main_layout = QHBoxLayout(central)

        # Sidebar
        self.sidebar = self.build_sidebar()
        main_layout.addWidget(self.sidebar)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

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

        # PIE CHART
        self.pie_canvas, self.pie_ax, self.pie_fig = self.create_pie_chart(0.5, 0.5)
        layout.addWidget(self.pie_canvas, 0, 0, 1, 2)

        # TOP WORDS PANEL
        self.top_words_box = QTextEdit()
        self.top_words_box.setReadOnly(True)
        self.top_words_box.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.top_words_box, 0, 2, 1, 2)

        # TEXT INPUT
        self.text_box = QPlainTextEdit()
        self.text_box.setPlaceholderText("Paste news article text here...")
        layout.addWidget(self.text_box, 1, 0, 1, 4)

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

        layout.addLayout(btn_row, 2, 0, 1, 4)

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
        layout.addWidget(self.spinner_label, 3, 0, 1, 4)

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

        layout.addSpacing(20)

        summary_header = QLabel("Summary")
        summary_header.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(summary_header)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        layout.addWidget(self.summary_box)

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

        # match figure background to theme
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

        # ensure all text (labels + percentages) matches theme
        for text in self.pie_ax.texts:
            text.set_color(self.colors["text"])

        self.pie_canvas.draw()

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

        # update top words panel
        if word_counts:
            lines = [f"<b>{w}</b> — {c} times" for w, c in word_counts]
            self.top_words_box.setHtml("<br>".join(lines))
        else:
            self.top_words_box.setHtml("No significant words found.")

        # update details tab
        details = [
            f"<b>Prediction:</b> {label}",
            f"<b>Real:</b> {real_prob:.2%}",
            f"<b>Fake:</b> {fake_prob:.2%}",
        ]
        self.details_label.setText("<br>".join(details))

        # summary
        summary = summarize_text(text)
        self.summary_box.setPlainText(summary)

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
            self.status_label.setText(f"CSV selected: {os.path.basename(path)}")

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
        # refresh chart with neutral values so theme applies cleanly
        self.update_pie_chart(0.5, 0.5)

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QWidget { background-color: #121212; color: #FFFFFF; }
                QPlainTextEdit { background-color: #1E1E1E; color: #FFFFFF; }
                QTextEdit { background-color: #1E1E1E; color: #FFFFFF; }
                QPushButton { background-color: #6C5CE7; color: #FFFFFF; }
                QPushButton:hover { background-color: #A29BFE; }
            """)
        else:
            self.setStyleSheet("""
                QWidget { background-color: #FFFFFF; color: #000000; }
                QPlainTextEdit { background-color: #FFFFFF; color: #000000; }
                QTextEdit { background-color: #FFFFFF; color: #000000; }
                QPushButton { background-color: #0984E3; color: #FFFFFF; }
                QPushButton:hover { background-color: #74B9FF; }
            """)


# ---------- RUN ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FakeNewsDashboard()
    window.show()
    sys.exit(app.exec())