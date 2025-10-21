# -*- coding: utf-8 -*-
"""
Floodzy Professional PDF Report Generator (Final, Layout-Safe Version)

✅ Fitur Unggulan:
- Struktur Laporan Akademik: Cover, Daftar Isi, Bab, Sub-bab.
- Header & Footer Otomatis: Judul laporan dan nomor halaman di setiap halaman.
- Desain Profesional: Styling konsisten untuk teks, judul, dan gambar.
- Layout Gambar Aman: Otomatis menyesuaikan ukuran gambar agar tidak 'overflow' (tanpa preserveAspectRatio).
- Parser Teks Andal: Regex untuk format bold/italic.
- Daftar Isi Otomatis & Stabil: via afterFlowable + bookmarkKey.

Author: Gemini AI (Refactored for Matt Yudha)
Date: 21 Oktober 2025
"""

import os
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Image, PageBreak,
    ListFlowable, ListItem, NextPageTemplate
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib import colors

# --- 1️⃣ Konfigurasi Global & Path yang Andal ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Asumsi script ini di folder 'ml'
except NameError:
    SCRIPT_DIR = os.getcwd()
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "feature_importance")
os.makedirs(REPORT_DIR, exist_ok=True)

PDF_PATH = os.path.join(REPORT_DIR, "Laporan_Analisis_Fitur_Floodzy_Ujian.pdf")
LOGO_PATH = os.path.join(PROJECT_ROOT, "logo_universitas.png")
XGB_IMG_PATH = os.path.join(REPORT_DIR, "xgb_feature_importance_top10.png")
SHAP_IMG_PATH = os.path.join(REPORT_DIR, "shap_summary_plot.png")


class ProfessionalReport:
    """
    Builder laporan PDF profesional dengan struktur akademik.
    """
    def __init__(self, path, title, author, nim, department, university):
        self.path = path
        self.title = title
        self.author = author
        self.nim = nim
        self.department = department
        self.university = university

        self.story = []
        self.toc = TableOfContents()
        self.setup_styles()

        self.doc = BaseDocTemplate(
            path,
            pagesize=A4,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
        )

        # Frames & Templates
        main_frame = Frame(
            self.doc.leftMargin,
            self.doc.bottomMargin,
            self.doc.width,
            self.doc.height,
            id="main_frame",
        )
        main_template = PageTemplate(
            id="main", frames=[main_frame], onPage=self._header_footer
        )

        cover_frame = Frame(
            self.doc.leftMargin,
            self.doc.bottomMargin,
            self.doc.width,
            self.doc.height,
            id="cover_frame",
        )
        cover_template = PageTemplate(id="cover", frames=[cover_frame])

        self.doc.addPageTemplates([cover_template, main_template])

        # Hook untuk TOC yang stabil (auto)
        self.doc.afterFlowable = self._after_flowable

    # ---------- Styles ----------
    def setup_styles(self):
        styles = getSampleStyleSheet()
        self.styles = {
            # Note: pakai 'Heading1/Heading2' bawaan agar aman
            "Title": ParagraphStyle(
                name="Title",
                parent=styles["Title"],
                fontName="Helvetica-Bold",
                fontSize=24,
                alignment=TA_CENTER,
                spaceAfter=1 * cm,
            ),
            "SubTitle": ParagraphStyle(
                name="SubTitle",
                parent=styles["Heading2"],
                fontName="Helvetica",
                fontSize=16,
                alignment=TA_CENTER,
                spaceAfter=0.5 * cm,
            ),
            "Author": ParagraphStyle(
                name="Author",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=12,
                alignment=TA_CENTER,
                spaceAfter=0.6 * cm,
            ),
            "AcademicInfo": ParagraphStyle(
                name="AcademicInfo",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=12,
                alignment=TA_CENTER,
                spaceAfter=0.5 * cm,
            ),
            "H1": ParagraphStyle(
                name="H1",
                parent=styles["Heading1"],
                fontName="Helvetica-Bold",
                fontSize=16,
                spaceBefore=18,
                spaceAfter=10,
                keepWithNext=1,
                leading=18,
            ),
            "H2": ParagraphStyle(
                name="H2",
                parent=styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=14,
                spaceBefore=14,
                spaceAfter=8,
                keepWithNext=1,
                leading=16,
            ),
            "Body": ParagraphStyle(
                name="Body",
                parent=styles["Normal"],
                fontName="Times-Roman",
                fontSize=12,
                leading=16,
                alignment=TA_JUSTIFY,
                spaceAfter=5,
            ),
            "Caption": ParagraphStyle(
                name="Caption",
                parent=styles["Normal"],
                fontName="Times-Italic",
                fontSize=10,
                alignment=TA_CENTER,
                spaceBefore=5,
                textColor=colors.grey,
            ),
        }

        # TOC styles
        self.toc.levelStyles = [
            ParagraphStyle(
                fontName="Helvetica-Bold",
                fontSize=14,
                name="TOCLevel1",
                leftIndent=20,
                firstLineIndent=-20,
                spaceBefore=10,
                leading=16,
            ),
            ParagraphStyle(
                fontName="Helvetica",
                fontSize=12,
                name="TOCLevel2",
                leftIndent=40,
                firstLineIndent=-20,
                spaceBefore=5,
                leading=12,
            ),
        ]

    # ---------- Header & Footer ----------
    def _header_footer(self, canvas, doc):
        canvas.saveState()

        # Header text
        header_para = Paragraph(
            "Laporan Analisis Fitur — Proyek Floodzy", self.styles["Body"]
        )
        w, h = header_para.wrap(doc.width, doc.topMargin)
        header_para.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin - h)
        canvas.line(
            doc.leftMargin,
            doc.height + doc.topMargin - h - 2,
            doc.leftMargin + doc.width,
            doc.height + doc.topMargin - h - 2,
        )

        # Footer page number (kanan)
        page_num_text = f"Halaman {doc.page}"
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(
            doc.leftMargin + doc.width, doc.bottomMargin - 0.5 * cm, page_num_text
        )

        canvas.restoreState()

    # ---------- TOC Hook ----------
    def _after_flowable(self, flowable):
        """
        Otomatis mendeteksi Heading dan menambah entry TOC + outline.
        """
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name in ("H1", "H2"):
                level = 0 if style_name == "H1" else 1
                text = flowable.getPlainText()
                key = getattr(flowable, "bookmarkKey", None) or self._make_key(text, level)

                # Bookmark page dan outline di PDF
                self.doc.canv.bookmarkPage(key)
                try:
                    # Outline (PDF sidebar)
                    self.doc.canv.addOutlineEntry(text, key, level=level, closed=False)
                except Exception:
                    pass  # amanin kalau viewer nggak support outline

                # Tambah ke TOC
                self.toc.addEntry(level, text, self.doc.page, key=key)

    @staticmethod
    def _make_key(text, level):
        clean = re.sub(r"<.*?>", "", text).strip()
        clean = re.sub(r"\s+", "-", clean).lower()
        return f"toc-{level}-{clean}"

    # ---------- Cover ----------
    def add_cover_page(self):
        # Logo (opsional)
        if os.path.exists(LOGO_PATH):
            logo = Image(LOGO_PATH, width=4 * cm, height=4 * cm)
            logo.hAlign = "CENTER"
            self.story.append(Spacer(1, 1.0 * cm))
            self.story.append(logo)
            self.story.append(Spacer(1, 0.5 * cm))

        self.story.append(Paragraph("LAPORAN PENELITIAN", self.styles["SubTitle"]))
        self.story.append(Spacer(1, 0.5 * cm))
        self.story.append(Paragraph(self.title, self.styles["Title"]))
        self.story.append(
            Paragraph(
                "Analisis <i>Feature Importance</i> Menggunakan XGBoost dan SHAP",
                self.styles["SubTitle"],
            )
        )
        self.story.append(Spacer(1, 2.0 * cm))

        self.story.append(Paragraph("Oleh:", self.styles["Author"]))
        self.story.append(
            Paragraph(f"{self.author.upper()}<br/>NIM: {self.nim}", self.styles["AcademicInfo"])
        )
        self.story.append(Spacer(1, 2.0 * cm))

        self.story.append(Paragraph(self.department, self.styles["AcademicInfo"]))
        self.story.append(Paragraph(self.university, self.styles["AcademicInfo"]))
        self.story.append(Paragraph(f"{datetime.now().year}", self.styles["AcademicInfo"]))

        # Next pages pakai template utama (dengan header/footer)
        self.story.append(NextPageTemplate("main"))
        self.story.append(PageBreak())

    # ---------- TOC ----------
    def add_table_of_contents(self):
        self.story.append(Paragraph("Daftar Isi", self.styles["H1"]))
        self.story.append(self.toc)
        self.story.append(PageBreak())

    # ---------- Chapters ----------
    def add_chapter(self, title, content_list, level=0):
        """
        Tambahkan bab atau sub-bab. level=0 => H1, level=1 => H2.
        Dukungan markup sederhana: *bold* dan _italic_.
        """
        style = self.styles["H1"] if level == 0 else self.styles["H2"]

        # Sanitize + markup
        clean_title = re.sub(r"<.*?>", "", title)
        formatted_title = re.sub(r"\*(.*?)\*", r"<b>\1</b>", title)
        formatted_title = re.sub(r"_(.*?)_", r"<i>\1</i>", formatted_title)

        # Buat paragraph heading
        p = Paragraph(formatted_title, style)
        p.bookmarkText = clean_title
        p.keepWithNext = 1
        p.bookmarkKey = self._make_key(clean_title, level)
        self.story.append(p)

        # Isi kontennya
        for item in content_list:
            if isinstance(item, str):
                s = re.sub(r"\*(.*?)\*", r"<b>\1</b>", item)
                s = re.sub(r"_(.*?)_", r"<i>\1</i>", s)
                self.story.append(Paragraph(s, self.styles["Body"]))
            else:
                self.story.append(item)

        self.story.append(Spacer(1, 0.5 * cm))

    # ---------- Images (Layout-Safe, keep aspect ratio) ----------
    def add_image_with_caption(self, img_path, caption, max_width=15 * cm, max_height=18 * cm):
        """
        Menambahkan gambar yang otomatis discale biar muat frame:
        - Preserve aspect ratio tanpa 'preserveAspectRatio' (yang memang tidak ada di ReportLab).
        - Batas aman default: width 15cm, height 18cm.
        """
        if not os.path.exists(img_path):
            self.story.append(Paragraph(f"<i>[Gambar tidak ditemukan di: {img_path}]</i>", self.styles["Caption"]))
            return

        img = Image(img_path)  # biarkan ambil dimensi asli
        # Set ukuran gambar yang ditarik (drawWidth/Height) dengan skala aman
        # 1) coba fit by width
        img.drawWidth = max_width
        img.drawHeight = img.imageHeight * (max_width / float(img.imageWidth))
        # 2) bila masih ketinggian, fit by height
        if img.drawHeight > max_height:
            img.drawHeight = max_height
            img.drawWidth = img.imageWidth * (max_height / float(img.imageHeight))

        img.hAlign = "CENTER"
        self.story.append(img)
        self.story.append(Paragraph(caption, self.styles["Caption"]))

    # ---------- Build ----------
    def build_pdf(self):
        self.doc.multiBuild(self.story)
        print(f"✅ Laporan profesional berhasil dibuat di: {self.path}")

        try:
            os.startfile(self.path)  # Windows only
            print("📂 PDF otomatis dibuka.")
        except Exception:
            print("ℹ️ Tidak bisa auto-open PDF di sistem ini. Silakan buka manual.")


# --- 2️⃣ Konten Laporan ---
REPORT_TITLE = "🌊 Floodzy: Prediksi Risiko Banjir Berbasis Machine Learning"
AUTHOR_NAME = "Matt Yudha"
AUTHOR_NIM = "123456789"
DEPARTMENT = "PROGRAM STUDI TEKNIK INFORMATIKA"
UNIVERSITY = "UNIVERSITAS TEKNOLOGI KREATIF"

konten_abstrak = [
    "Sistem peringatan dini banjir merupakan komponen krusial dalam mitigasi bencana. Proyek Floodzy mengembangkan model *machine learning* menggunakan algoritma XGBoost untuk memprediksi ketinggian air sungai berdasarkan data sensorik historis. Laporan ini berfokus pada analisis *feature importance* untuk mengidentifikasi variabel paling berpengaruh. Dengan menggunakan metode *built-in* dari XGBoost dan SHAP (*SHapley Additive exPlanations*), penelitian ini menunjukkan bahwa fitur seperti _river_level_cm_, _rain_mm_, dan _soil_moisture_ secara konsisten menjadi prediktor utama. Hasil analisis ini tidak hanya memvalidasi pilihan model tetapi juga memberikan wawasan mendalam tentang faktor-faktor hidrologis yang mendorong terjadinya banjir, yang dapat digunakan untuk menyempurnakan strategi penempatan sensor dan kebijakan penanggulangan bencana."
]

konten_pendahuluan = [
    "Banjir adalah salah satu bencana alam yang paling sering terjadi dan merusak di Indonesia. Kerugian yang ditimbulkan tidak hanya bersifat material, tetapi juga dapat mengancam keselamatan jiwa. Oleh karena itu, pengembangan sistem peringatan dini (EWS - *Early Warning System*) yang andal menjadi sebuah keharusan.",
    "Proyek Floodzy bertujuan untuk membangun EWS berbasis data dengan memanfaatkan kemajuan dalam bidang *machine learning*. Dengan menganalisis data historis dari berbagai sensor (curah hujan, ketinggian air, kelembapan tanah), model diharapkan dapat memberikan prediksi akurat beberapa jam sebelum kejadian. Keberhasilan model ini sangat bergantung pada pemilihan fitur yang relevan. Laporan ini akan membahas secara mendalam proses analisis dan interpretasi fitur-fitur yang digunakan dalam model prediksi Floodzy."
]

konten_metodologi = [
    "Metodologi penelitian dibagi menjadi beberapa tahapan utama:",
    ListFlowable([
        ListItem(Paragraph("<b>Pengumpulan Data:</b> Data dikumpulkan dari sensor IoT yang tersebar di beberapa titik DAS (Daerah Aliran Sungai), mencakup variabel seperti curah hujan (mm), ketinggian air sungai (cm), kelembapan tanah (%), suhu (°C), dan kecepatan angin (km/j) selama periode 24 bulan.", style=getSampleStyleSheet()['Normal'])),
        ListItem(Paragraph("<b>Model Machine Learning:</b> Algoritma yang digunakan adalah XGBoost (eXtreme Gradient Boosting), yang dikenal karena performa tinggi dan kemampuannya menangani data tabular kompleks.", style=getSampleStyleSheet()['Normal'])),
        ListItem(Paragraph("<b>Analisis Feature Importance:</b> Dua metode digunakan untuk mengukur pentingnya fitur: <br/>1. <i>Importance Type 'weight'</i> dari XGBoost. <br/>2. <i>SHAP Summary Plot</i> untuk dampak global tiap fitur terhadap prediksi.", style=getSampleStyleSheet()['Normal'])),
    ], bulletType='1')
]

konten_hasil = [
    "Setelah model XGBoost dilatih, analisis *feature importance* dilakukan untuk menginterpretasi faktor-faktor yang paling mempengaruhi prediksi ketinggian air. Hasilnya disajikan dalam dua visualisasi utama."
]

konten_kesimpulan = [
    "Berdasarkan analisis yang telah dilakukan, dapat disimpulkan bahwa fitur-fitur hidrologis utama seperti ketinggian air sebelumnya (_river_level_cm_), curah hujan (_rain_mm_), dan kelembapan tanah (_soil_moisture_) merupakan prediktor paling signifikan dalam model Floodzy. Visualisasi dari XGBoost dan SHAP secara konsisten menempatkan fitur-fitur ini di peringkat teratas.",
    "Temuan ini mengonfirmasi bahwa model XGBoost tidak hanya berkinerja baik secara prediktif, tetapi juga dapat diinterpretasikan dan sejalan dengan pemahaman domain hidrologi. Hasil ini memberikan dasar yang kuat untuk implementasi model Floodzy dalam sistem peringatan dini banjir yang andal dan dapat dipertanggungjawabkan."
]


# --- 3️⃣ Main Execution ---
if __name__ == "__main__":
    report = ProfessionalReport(PDF_PATH, REPORT_TITLE, AUTHOR_NAME, AUTHOR_NIM, DEPARTMENT, UNIVERSITY)

    report.add_cover_page()
    report.add_table_of_contents()

    report.add_chapter("Abstrak", konten_abstrak)
    report.add_chapter("BAB I: PENDAHULUAN", konten_pendahuluan, level=0)
    report.add_chapter("BAB II: METODOLOGI PENELITIAN", konten_metodologi, level=0)

    report.add_chapter("BAB III: HASIL DAN ANALISIS", konten_hasil, level=0)
    report.add_image_with_caption(
        XGB_IMG_PATH,
        caption="Gambar 1: Peringkat 10 Fitur Teratas Berdasarkan XGBoost ('weight')",
    )
    report.story.append(Spacer(1, 0.5 * cm))
    report.story.append(Paragraph(
        "Gambar 1 menunjukkan frekuensi penggunaan setiap fitur untuk membuat keputusan di dalam model. "
        "Fitur <i>river_level_cm</i> menjadi yang paling sering digunakan, mengindikasikan bahwa kondisi sungai saat ini "
        "adalah prediktor terkuat untuk kondisi di masa depan.",
        report.styles["Body"],
    ))
    report.story.append(Spacer(1, 1 * cm))

    report.add_image_with_caption(
        SHAP_IMG_PATH,
        caption="Gambar 2: SHAP Summary Plot untuk Menganalisis Dampak Fitur",
    )
    report.story.append(Spacer(1, 0.5 * cm))
    report.story.append(Paragraph(
        "Gambar 2 memberikan detail lebih kaya. Titik-titik merah menunjukkan nilai fitur yang tinggi, sedangkan biru menunjukkan nilai rendah. "
        "Terlihat jelas bahwa nilai <i>river_level_cm</i> yang tinggi (merah) memiliki dampak SHAP positif yang besar, "
        "artinya sangat mendorong prediksi banjir. Hal yang sama berlaku untuk <i>rain_mm</i>.",
        report.styles["Body"],
    ))

    report.add_chapter("BAB IV: KESIMPULAN", konten_kesimpulan, level=0)

    report.build_pdf()
