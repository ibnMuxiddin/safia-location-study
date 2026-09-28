"""Step 12: one-page A4 report in Uzbek and Russian (reportlab).

Every number is read from the outputs of earlier steps (facts.json, model_metrics.json, top10.csv,
coverage5.csv, candidate_checks.csv), never typed by hand. A static map is drawn with matplotlib.
Checks: each PDF has exactly one page; a PNG preview is rendered for visual inspection.
"""
import html
import json
import os
import sys

import geopandas as gpd
import h3
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pypdfium2 as pdfium
from matplotlib.patches import Polygon as MplPolygon
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import DATA_INTERIM, DATA_PROCESSED, DATA_RAW, OUTPUTS_FIGURES, REPORTS

AUTHOR = "Azizbek Sunnatov"
REPO_URL = "github.com/ibnMuxiddin/safia-location-study"
MAP_PNG = OUTPUTS_FIGURES / "12_report_map.png"
RED, PURPLE, GREY = "#c0392b", "#8e44ad", "#555555"
SCORE_COLORS = ["#f7f7f7", "#fde0c5", "#f9a870", "#e8603c", "#b2182b"]
RU_DISTRICTS = {"Olmazor": "Алмазар", "Yunusobod": "Юнусабад", "Shayxontohur": "Шайхантахур",
                "Mirzo Ulug'bek": "Мирзо-Улугбек", "Uchtepa": "Учтепа", "Yashnobod": "Яшнабад",
                "Chilonzor": "Чиланзар", "Yangihayot": "Янгихаёт", "Sergeli": "Сергели", "Mirobod": "Мирабад",
                "Yakkasaroy": "Яккасарай", "Bektemir": "Бектемир"}

font_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(font_dir, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(font_dir, "DejaVuSans-Bold.ttf")))
# Without a registered family, <b> inside Paragraphs silently stays regular
registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold")


def num(x: float, d: int = 1) -> str:
    """Number with a decimal comma (uz/ru convention)."""
    return f"{x:.{d}f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def esc(text) -> str:
    """Data strings go into ReportLab's XML markup: '&' or '<' in a name would break the Paragraph."""
    return html.escape(str(text), quote=False)


def district(name: str, lang: str) -> str:
    return esc(RU_DISTRICTS.get(name, name) if lang == "ru" else name)


def fit(text: str, width_pt: float, size: float) -> str:
    """Shorten at word boundaries until the text fits the column (Cyrillic is wider than Latin)."""
    if pdfmetrics.stringWidth(text, "DejaVu", size) <= width_pt:
        return text
    words = text.split(" ")
    while len(words) > 1:
        words.pop()
        cand = " ".join(words).rstrip(" ,-") + "…"
        if pdfmetrics.stringWidth(cand, "DejaVu", size) <= width_pt:
            return cand
    return words[0]


def landmark(text: str, lang: str) -> str:
    name, kind = text.rsplit(" (", 1)
    kind = kind.rstrip(")")
    if kind == "metro":
        return f"{name} metro" if lang == "uz" else f"м. {name}"
    if kind == "mall":
        return f"{name} SM" if lang == "uz" else f"ТЦ {name}"
    return name


def draw_map(top: pd.DataFrame, cov: pd.DataFrame) -> None:
    cells = pd.read_parquet(DATA_PROCESSED / "hex_features.parquet")[["h3"]].merge(
        pd.read_parquet(DATA_PROCESSED / "oof_scores.parquet")[["h3", "score"]], on="h3")
    safia = pd.read_csv(DATA_INTERIM / "safia_tashkent.csv")
    safia = safia[safia["special_site"].isna()]
    comp = pd.read_csv(DATA_INTERIM / "competitors_geo.csv")
    districts = gpd.read_file(DATA_RAW / "tashkent_districts.geojson")
    districts = districts[districts["district"] != "Yangi Toshkent"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("score", SCORE_COLORS)

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    for r in cells.itertuples():
        ax.add_patch(MplPolygon([(lon, lat) for lat, lon in h3.cell_to_boundary(r.h3)],
                                facecolor=cmap(r.score), edgecolor="none", alpha=0.75))
    districts.boundary.plot(ax=ax, color="#777", linewidth=0.5)
    ax.scatter(comp["lon"], comp["lat"], s=5, color="#2c3e50", zorder=3, label="Bon! / Cake Lab / Breadly")
    ax.scatter(safia["lon"], safia["lat"], s=6, color="#e74c3c", edgecolor="white", linewidth=0.3, zorder=4, label="Safia")
    for frame, color in [(top, RED), (cov, PURPLE)]:
        for r in frame.itertuples():
            ax.annotate(str(r.rank), (r.lon, r.lat), ha="center", va="center", fontsize=6.5, color="white",
                        fontweight="bold", zorder=5,
                        bbox={"boxstyle": "round,pad=0.25", "fc": color, "ec": "white", "lw": 0.5})
    ax.legend(loc="lower right", fontsize=6, frameon=True, markerscale=2)
    ax.set_aspect(1 / 0.75)  # ~cos(41.3 deg)
    ax.set_axis_off()
    b = districts.total_bounds
    ax.set_xlim(b[0] - 0.005, b[2] + 0.005)
    ax.set_ylim(b[1] - 0.005, b[3] + 0.005)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(MAP_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def texts(lang: str, f: dict, m: dict, checks: pd.DataFrame) -> dict:
    res = {r["model"]: r for r in m["results"]}[m["chosen_model"]]
    comp = f["competitors_near_safia_by_brand"]
    n_checked, n_rej = len(checks), int((checks["verdict"] == "rejected").sum())
    low = min(f["per_100k"], key=f["per_100k"].get)
    high = max(f["per_100k"], key=f["per_100k"].get)
    far = sorted(f["far_share"].items(), key=lambda kv: -kv[1])[:3]
    if lang == "uz":
        return {
            "title": "Safia: Toshkentda yangi filiallar uchun hududlar",
            "sub": f"Ochiq ma'lumot asosidagi tadqiqot · {AUTHOR} · ML Researcher vakansiyasi uchun · 2026-yil sentyabr",
            "facts_h": "Tarmoq haqida 3 fakt",
            "map_note": "<b>Xaritani o'qish:</b> kataklar rangi — o'xshashlik bali (0–1). "
                        "<font color='#c0392b'><b>1–10</b></font> — markazdagi nomzodlar, "
                        "<font color='#8e44ad'><b>C1–C5</b></font> — janubdagi nomzodlar. "
                        "Qizil nuqtalar — Safia, to'q nuqtalar — raqobatchilar. Yangi Toshkent tumani kiritilmagan "
                        "(asosan qurilish, rasmiy aholi ma'lumoti yo'q).",
            "f1": f"<b>Aholiga nisbatan qamrov {num(f['per_100k_ratio_max_min'])} barobar farq qiladi.</b> 100 ming aholiga: "
                  f"{district(high, 'uz')} {num(f['per_100k'][high])}, shahar o'rtachasi {num(f['per_100k_city'])}, "
                  f"<b>{district(low, 'uz')} {num(f['per_100k'][low])}</b> ({f['population_by_district'][low] // 1000} ming aholiga "
                  f"{f['branches_by_district'][low]} filial).",
            "f2": f"<b>Turar-joy hududlarining {pct(f['far_share_city'])} i eng yaqin Safia'dan {f['cover_m'] // 1000} km dan uzoqda</b> "
                  f"(~{num(f['residents_far_est'] / 1e6, 2)} mln aholi): "
                  + ", ".join(f"{district(d, 'uz')} {pct(v)}" for d, v in far) + ".",
            "f3": f"<b>Raqobatchilarning {pct(f['competitors_near_safia_share'])} i Safia'dan {f['overlap_m']} m ichida</b> "
                  f"(Bon! {pct(comp['Bon!'])}, Cake Lab {pct(comp['Cake Lab'])}): raqobat markazda, janub bo'sh.",
            "model_h": "Model",
            "model": f"Har bir ~0,9 km² katak uchun <b>hozirgi Safia joylariga o'xshashlik</b> bahosi. "
                     f"{len(m['features'])} ta ochiq belgi: kafe, maktab, bekat, supermarket, OTM, aholi zichligi, sanoat ulushi… "
                     f"Logistik regressiya, fazoviy kross-validatsiya (shahar qismlari bo'yicha): "
                     f"ROC-AUC {num(res['spatial_roc_auc'], 2)}, PR-AUC {num(res['spatial_pr_auc'], 2)} "
                     f"(tasodifiy — {num(m['prevalence'], 2)}). LightGBM yaxshiroq chiqmadi.",
            "cols": ["#", "Tuman", "Mo'ljal", "Ball"],
            "top_h": "1–10 · O'xshashlik: markazdagi bo'shliqlar",
            "cov_h": "C1–C5 · Qamrov: kam xizmat ko'rsatilgan janub",
            "concl_h": "Xulosa: ikki strategiya",
            "concl": "<b>O'xshashlik</b> (1–10) — mavjud muvaffaqiyatli joylarga o'xshash, xavfi past. "
                     "<b>Qamrov</b> (C1–C5) — aholi ko'p, filial kam, lekin model ularni baholay olmaydi: "
                     "janubda o'rganish uchun misol kam. Bu yerda qaror uchun <b>ichki savdo ma'lumoti</b> kerak.",
            "lim_h": "Cheklovlar",
            "lim": ["Model filial <b>borligini</b> o'rganadi, sotuvni emas.",
                    f"OSM yangi qurilishlardan orqada qoladi (Qo'yliqdagi yangi FoodCity yo'q). Nomzodlar sun'iy yo'ldosh "
                    f"xaritasida tekshirildi: {n_checked} ta, {n_rej} tasi rad etildi (ekinzor).",
                    "Global aholi xaritalari (WorldPop, GHSL) Toshkent janubini 4–7 barobar oshiradi — "
                    "rasmiy tuman statistikasi (2023) ishlatildi.",
                    "Bon! va Breadly ro'yxati to'liq bo'lmasligi mumkin."],
            "next_h": "Ichki ma'lumot bilan nima qilaman",
            "next": ["Target «filial bor» → <b>filial sotuvi</b> (iiko): yangi joy uchun sotuv prognozi.",
                     f"<b>Kannibalizatsiya:</b> {f['safia_close_pairs']} filialning {f['overlap_m']} m ichida boshqa Safia bor — "
                     "sotuvga ta'sirini o'lchash.",
                     "<b>Janub (C1–C5):</b> mavjud janubiy filiallar sotuvi bo'yicha talabni baholash."],
            "foot": f"Kod, interaktiv xarita (uz/ru) va ma'lumotlar: <b>{REPO_URL}</b>. "
                    "Manbalar: safiabakery.uz, cakelab.uz, OpenStreetMap, Toshkent shahar statistika boshqarmasi (2023).",
        }
    return {
        "title": "Safia: зоны для новых филиалов в Ташкенте",
        "sub": f"Исследование на открытых данных · {AUTHOR} · для вакансии ML Researcher · сентябрь 2026",
        "facts_h": "3 факта о сети",
        "map_note": "<b>Как читать карту:</b> цвет ячеек — балл сходства (0–1). "
                    "<font color='#c0392b'><b>1–10</b></font> — кандидаты в центре, "
                    "<font color='#8e44ad'><b>C1–C5</b></font> — кандидаты на юге. "
                    "Красные точки — Safia, тёмные — конкуренты. Район Янги Ташкент не включён "
                    "(в основном стройка, нет официальных данных о населении).",
        "f1": f"<b>Охват на жителя различается в {num(f['per_100k_ratio_max_min'])} раза.</b> На 100 тыс. жителей: "
              f"{district(high, 'ru')} {num(f['per_100k'][high])}, в среднем по городу {num(f['per_100k_city'])}, "
              f"<b>{district(low, 'ru')} {num(f['per_100k'][low])}</b> ({f['branches_by_district'][low]} филиала на "
              f"{f['population_by_district'][low] // 1000} тыс. жителей).",
        "f2": f"<b>{pct(f['far_share_city'])} жилых территорий дальше {f['cover_m'] // 1000} км от ближайшей Safia</b> "
              f"(~{num(f['residents_far_est'] / 1e6, 2)} млн жителей): "
              + ", ".join(f"{district(d, 'ru')} {pct(v)}" for d, v in far) + ".",
        "f3": f"<b>{pct(f['competitors_near_safia_share'])} филиалов конкурентов — в {f['overlap_m']} м от Safia</b> "
              f"(Bon! {pct(comp['Bon!'])}, Cake Lab {pct(comp['Cake Lab'])}): конкуренция в центре, юг свободен.",
        "model_h": "Модель",
        "model": f"Для каждой ячейки ~0,9 км² — оценка <b>сходства с текущими местами Safia</b>. "
                 f"{len(m['features'])} открытых признаков: кафе, школы, остановки, супермаркеты, вузы, плотность "
                 f"населения, доля промзон… Логистическая регрессия, пространственная кросс-валидация (по частям города): "
                 f"ROC-AUC {num(res['spatial_roc_auc'], 2)}, PR-AUC {num(res['spatial_pr_auc'], 2)} "
                 f"(случайная — {num(m['prevalence'], 2)}). LightGBM не оказался лучше.",
        "cols": ["#", "Район", "Ориентир", "Балл"],
        "top_h": "1–10 · Сходство: пробелы в центре",
        "cov_h": "C1–C5 · Охват: недообслуженный юг",
        "concl_h": "Вывод: две стратегии",
        "concl": "<b>Сходство</b> (1–10) — похоже на действующие успешные точки, риск ниже. "
                 "<b>Охват</b> (C1–C5) — много жителей, мало филиалов, но модель не может их оценить: "
                 "на юге мало примеров для обучения. Для решения здесь нужны <b>внутренние данные о продажах</b>.",
        "lim_h": "Ограничения",
        "lim": ["Модель учится на <b>наличии</b> филиала, а не на продажах.",
                f"OSM отстаёт от новой застройки (нет нового FoodCity у Куйлюка). Кандидаты проверены по спутниковым "
                f"снимкам: {n_checked}, отклонён {n_rej} (посевное поле).",
                "Глобальные карты населения (WorldPop, GHSL) завышают юг Ташкента в 4–7 раз — "
                "использована официальная статистика по районам (2023).",
                "Список Bon! и Breadly может быть неполным."],
        "next_h": "Что сделаю с внутренними данными",
        "next": ["Цель «филиал есть» → <b>продажи филиала</b> (iiko): прогноз продаж для новой точки.",
                 f"<b>Каннибализация:</b> у {f['safia_close_pairs']} филиалов другая Safia в пределах {f['overlap_m']} м — "
                 "измерить влияние на продажи.",
                 "<b>Юг (C1–C5):</b> оценить спрос по продажам действующих южных филиалов."],
        "foot": f"Код, интерактивная карта (uz/ru) и данные: <b>{REPO_URL}</b>. "
                "Источники: safiabakery.uz, cakelab.uz, OpenStreetMap, Управление статистики г. Ташкента (2023).",
    }


def build(lang: str, top: pd.DataFrame, cov: pd.DataFrame, f: dict, m: dict, checks: pd.DataFrame) -> None:
    t = texts(lang, f, m, checks)
    out = REPORTS / f"safia_location_{lang}.pdf"
    base = ParagraphStyle("b", fontName="DejaVu", fontSize=9.0, leading=11.8, alignment=TA_LEFT)
    small = ParagraphStyle("s", parent=base, fontSize=7.9, leading=9.9)
    h = ParagraphStyle("h", parent=base, fontName="DejaVu-Bold", fontSize=10.2, leading=12.5, spaceBefore=5, spaceAfter=2,
                       textColor=colors.HexColor(RED))
    title = ParagraphStyle("t", parent=base, fontName="DejaVu-Bold", fontSize=18, leading=22)
    sub = ParagraphStyle("st", parent=base, fontSize=8.5, textColor=colors.HexColor(GREY))

    def cand_table(frame, header, color):
        rows = [[Paragraph(f"<b>{c}</b>", small) for c in t["cols"]]]
        for r in frame.itertuples():
            rows.append([Paragraph(esc(r.rank), small), Paragraph(district(r.district, lang), small),
                         Paragraph(esc(fit(landmark(r.landmark, lang), 44 * mm - 10, small.fontSize)), small),
                         Paragraph(num(r.score, 2), small)])
        tbl = Table(rows, colWidths=[7 * mm, 27 * mm, 44 * mm, 12 * mm])
        tbl.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(color)),
                                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f6f6")]),
                                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 0.6),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
        hs = ParagraphStyle("th", parent=h, textColor=colors.HexColor(color), fontSize=8.8)
        return [Paragraph(header, hs), tbl]

    right = [Paragraph(t["facts_h"], h)]
    for k in ("f1", "f2", "f3"):
        right += [Paragraph("• " + t[k], base), Spacer(1, 2.5)]
    right += [Paragraph(t["model_h"], h), Paragraph(t["model"], base)]
    right += cand_table(top, t["top_h"], RED)
    right += cand_table(cov, t["cov_h"], PURPLE)

    left = [Image(str(MAP_PNG), width=92 * mm, height=92 * mm * 5.6 / 5.2, kind="proportional"),
            Spacer(1, 3), Paragraph(t["map_note"], small)]
    top_row = Table([[left, right]], colWidths=[94 * mm, 92 * mm])
    top_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))

    def block(head, items):
        body = [Paragraph(head, h)]
        body += [Paragraph(items, base)] if isinstance(items, str) else [Paragraph("• " + x, base) for x in items]
        return body

    bottom = Table([[block(t["concl_h"], t["concl"]), block(t["lim_h"], t["lim"]), block(t["next_h"], t["next"])]],
                   colWidths=[58 * mm, 66 * mm, 62 * mm])
    bottom.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#fbeeee"))]))

    story = [Paragraph(t["title"], title), Paragraph(t["sub"], sub), Spacer(1, 4), top_row, Spacer(1, 3), bottom,
             Spacer(1, 4), Paragraph(t["foot"], ParagraphStyle("f", parent=small, textColor=colors.HexColor(GREY)))]
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm, topMargin=11 * mm,
                            bottomMargin=10 * mm, title=t["title"], author=AUTHOR)
    doc.build(story)

    pages = len(PdfReader(str(out)).pages)
    png = OUTPUTS_FIGURES / f"12_report_preview_{lang}.png"
    pdfium.PdfDocument(str(out))[0].render(scale=1.6).to_pil().save(png)
    print(f"{out.name}: {pages} page(s), {out.stat().st_size / 1e3:.0f} KB | preview {png.name}")
    if pages != 1:
        raise SystemExit(f"{out.name} has {pages} pages; it must fit on one page")


def main() -> int:
    top = pd.read_csv(DATA_PROCESSED / "top10.csv")
    cov = pd.read_csv(DATA_PROCESSED / "coverage5.csv")
    facts = json.loads((DATA_PROCESSED / "facts.json").read_text(encoding="utf-8"))
    metrics = json.loads((DATA_PROCESSED / "model_metrics.json").read_text(encoding="utf-8"))
    checks = pd.read_csv(DATA_INTERIM / "candidate_checks.csv")
    draw_map(top, cov)
    for lang in ("uz", "ru"):
        build(lang, top, cov, facts, metrics, checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
