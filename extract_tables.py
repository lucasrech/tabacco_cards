from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from rapidocr_onnxruntime import RapidOCR


SHEET_RE = re.compile(r"^\d{4}$")
INDEX_RE = re.compile(r"\b0/\d+-[A-Za-z0-9\-()]+\b|\b0/2-[A-Za-z0-9\-()]+\b")


@dataclass
class OCRToken:
    text: str
    conf: float
    left: float
    top: float
    right: float
    bottom: float

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class TableStructure:
    table_box: tuple[float, float, float, float]
    row_boxes: list[tuple[float, float, float, float]]
    col_boxes: list[tuple[float, float, float, float]]


# --------------------------
# Basic text helpers
# --------------------------


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_cell(value: str) -> str:
    value = normalize_text(value)
    value = value.replace("“", '"').replace("”", '"')
    return value


def is_ditto_mark(value: str) -> bool:
    compact = normalize_text(value).replace(" ", "")
    if not compact:
        return False
    if compact in {"〃", "々"}:
        return True
    return re.fullmatch(r"""["'`´“”]+""", compact) is not None


def is_title_ditto_marker(value: str) -> bool:
    compact = normalize_text(value).replace(" ", "").replace('"', "")
    if is_ditto_mark(value):
        return True
    # Common OCR confusion for ditto marks in old monospaced tables.
    return compact in {"11", "I", "II", "H", "l1", "1l", "||"}


# --------------------------
# OCR
# --------------------------


def parse_ocr_tokens(ocr_result: list, min_confidence: float) -> list[OCRToken]:
    tokens: list[OCRToken] = []
    if not ocr_result:
        return tokens

    for item in ocr_result:
        if len(item) != 3:
            continue
        points, text, conf = item
        text = normalize_text(str(text))
        conf = float(conf)
        if not text or conf < min_confidence:
            continue

        pts = np.array(points, dtype=float)
        tokens.append(
            OCRToken(
                text=text,
                conf=conf,
                left=float(np.min(pts[:, 0])),
                top=float(np.min(pts[:, 1])),
                right=float(np.max(pts[:, 0])),
                bottom=float(np.max(pts[:, 1])),
            )
        )

    return tokens


# --------------------------
# AI structure (Table Transformer)
# --------------------------


@lru_cache(maxsize=1)
def load_tatr_models():
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"

    det_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection", use_fast=False)
    det_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection").to(device)
    det_model.eval()

    str_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition", use_fast=False)
    str_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition").to(device)
    str_model.eval()

    return {
        "device": device,
        "det_processor": det_processor,
        "det_model": det_model,
        "str_processor": str_processor,
        "str_model": str_model,
    }


def run_object_detection(processor, model, image: Image.Image, threshold: float) -> list[dict]:
    import torch

    model_device = next(model.parameters()).device
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=model_device)
    results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    id2label = model.config.id2label
    dets: list[dict] = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x0, y0, x1, y1 = [float(x) for x in box.tolist()]
        dets.append(
            {
                "score": float(score.item()),
                "label": id2label[int(label.item())].lower(),
                "box": (x0, y0, x1, y1),
            }
        )
    return dets


def area(box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def inside_box(cx: float, cy: float, box: tuple[float, float, float, float], margin: float = 0.0) -> bool:
    x0, y0, x1, y1 = box
    return (x0 - margin) <= cx <= (x1 + margin) and (y0 - margin) <= cy <= (y1 + margin)


def merge_near_boxes(
    boxes: list[tuple[float, float, float, float]],
    axis: str,
    overlap_threshold: float = 0.65,
) -> list[tuple[float, float, float, float]]:
    if not boxes:
        return []

    if axis == "x":
        boxes = sorted(boxes, key=lambda b: b[0])
    else:
        boxes = sorted(boxes, key=lambda b: b[1])

    merged: list[list[float]] = [list(boxes[0])]

    for box in boxes[1:]:
        cur = merged[-1]
        if axis == "x":
            inter = max(0.0, min(cur[2], box[2]) - max(cur[0], box[0]))
            base = max(1.0, min(cur[2] - cur[0], box[2] - box[0]))
        else:
            inter = max(0.0, min(cur[3], box[3]) - max(cur[1], box[1]))
            base = max(1.0, min(cur[3] - cur[1], box[3] - box[1]))

        if (inter / base) >= overlap_threshold:
            cur[0] = min(cur[0], box[0])
            cur[1] = min(cur[1], box[1])
            cur[2] = max(cur[2], box[2])
            cur[3] = max(cur[3], box[3])
        else:
            merged.append(list(box))

    return [tuple(b) for b in merged]


def detect_table_structure_ai(image: Image.Image) -> TableStructure | None:
    models = load_tatr_models()

    dets = run_object_detection(models["det_processor"], models["det_model"], image, threshold=0.55)
    table_candidates = [d for d in dets if d["label"] == "table"]
    if not table_candidates:
        return None

    table = max(table_candidates, key=lambda d: d["score"] * max(1.0, area(d["box"])))
    tx0, ty0, tx1, ty1 = table["box"]

    crop = image.crop((tx0, ty0, tx1, ty1))
    structs = run_object_detection(models["str_processor"], models["str_model"], crop, threshold=0.5)

    row_boxes_local = [d["box"] for d in structs if d["label"] == "table row"]
    col_boxes_local = [d["box"] for d in structs if d["label"] == "table column"]

    if len(col_boxes_local) < 2:
        return None

    row_boxes = [(x0 + tx0, y0 + ty0, x1 + tx0, y1 + ty0) for (x0, y0, x1, y1) in row_boxes_local]
    col_boxes = [(x0 + tx0, y0 + ty0, x1 + tx0, y1 + ty0) for (x0, y0, x1, y1) in col_boxes_local]

    row_boxes = merge_near_boxes(row_boxes, axis="y")
    col_boxes = merge_near_boxes(col_boxes, axis="x")
    row_boxes = sorted(row_boxes, key=lambda b: b[1])
    col_boxes = sorted(col_boxes, key=lambda b: b[0])

    return TableStructure(table_box=(tx0, ty0, tx1, ty1), row_boxes=row_boxes, col_boxes=col_boxes)


# --------------------------
# Build matrix from AI structure
# --------------------------


def assign_to_nearest(value: float, spans: list[tuple[float, float]]) -> int:
    if not spans:
        return 0
    for i, (s0, s1) in enumerate(spans):
        if s0 <= value <= s1:
            return i
    centers = [((s0 + s1) / 2.0) for s0, s1 in spans]
    return int(np.argmin(np.abs(np.array(centers) - value)))


def group_tokens_by_y(tokens: list[OCRToken]) -> list[list[OCRToken]]:
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda t: t.cy)
    heights = [max(1.0, t.bottom - t.top) for t in ordered]
    tol = max(8.0, float(np.median(heights)) * 0.60)

    rows: list[list[OCRToken]] = [[ordered[0]]]
    row_centers = [ordered[0].cy]

    for tok in ordered[1:]:
        if abs(tok.cy - row_centers[-1]) <= tol:
            rows[-1].append(tok)
            row_centers[-1] = float(np.mean([t.cy for t in rows[-1]]))
        else:
            rows.append([tok])
            row_centers.append(tok.cy)

    for row in rows:
        row.sort(key=lambda t: t.left)
    return rows


def matrix_from_structure(tokens: list[OCRToken], structure: TableStructure) -> list[list[str]]:
    ncols = len(structure.col_boxes)
    if ncols == 0:
        return []

    table_margin = 8.0
    table_tokens = [t for t in tokens if inside_box(t.cx, t.cy, structure.table_box, margin=table_margin)]
    if not table_tokens:
        return []

    col_spans = [(b[0], b[2]) for b in structure.col_boxes]

    # Build logical rows from OCR y-clusters to avoid accidental row merges.
    token_rows = group_tokens_by_y(table_tokens)
    matrix: list[list[str]] = []
    for row_tokens in token_rows:
        cells: list[list[OCRToken]] = [[] for _ in range(ncols)]
        for tok in row_tokens:
            ci = assign_to_nearest(tok.cx, col_spans)
            cells[ci].append(tok)

        row_values: list[str] = []
        for cell_tokens in cells:
            ordered = sorted(cell_tokens, key=lambda t: (t.cy, t.left))
            row_values.append(clean_cell(" ".join(t.text for t in ordered)))
        matrix.append(row_values)

    return matrix


def normalize_ditto_semantics(matrix: list[list[str]]) -> list[list[str]]:
    if not matrix:
        return matrix

    col_count = len(matrix[0])
    if col_count not in {4, 5}:
        return matrix

    title_idx = 1
    index_idx = 2
    card_idx = 3 if col_count == 5 else None

    prev_title = ""
    prev_base_title = ""
    prev_index = ""
    prev_card = ""

    out: list[list[str]] = []
    for row in matrix:
        cur = row[:]
        title_raw = clean_cell(cur[title_idx])
        index_raw = clean_cell(cur[index_idx])
        card_raw = clean_cell(cur[card_idx]) if card_idx is not None else ""

        title = title_raw.replace('"', "").strip()
        title = re.sub(r"\s+", " ", title)
        index = index_raw.replace('"', "").strip()
        card = card_raw

        title_range = re.search(r"\(\d+/\d+\)", title)
        index_range = re.search(r"\(\d+/\d+\)", index)
        title_is_range_only = bool(re.fullmatch(r"\(\d+/\d+\)", title))
        explicit_title_ditto = is_title_ditto_marker(title_raw)

        if explicit_title_ditto:
            title = prev_title
        elif title_is_range_only and prev_base_title:
            title = f"{prev_base_title} {title}"
        elif not title and index_range and prev_base_title:
            title = f"{prev_base_title} {index_range.group(0)}"
        elif title and title_range and prev_base_title and not re.search(r"[A-Za-z].*[A-Za-z]", title):
            # Handles rows where OCR captured only range or very short token.
            title = f"{prev_base_title} {title_range.group(0)}"

        if is_ditto_mark(index_raw):
            index = prev_index
        elif not index and prev_index and (title_is_range_only or explicit_title_ditto):
            index = prev_index

        if card_idx is not None:
            if is_ditto_mark(card_raw):
                card = prev_card
            if card:
                prev_card = card
            cur[card_idx] = card

        if title:
            prev_title = title
            base = re.sub(r"\s*\(\d+/\d+\)\s*$", "", title).strip()
            if base:
                prev_base_title = base

        if index:
            prev_index = index

        cur[title_idx] = title
        cur[index_idx] = index
        out.append(cur)

    return out


def collapse_columns_by_syntax(matrix: list[list[str]], forced_columns: int | None) -> list[list[str]]:
    if not matrix:
        return matrix
    ncols = len(matrix[0])
    if ncols <= 5 and forced_columns is None:
        return matrix

    def non_empty_col_values(j: int) -> list[str]:
        return [clean_cell(row[j]) for row in matrix if clean_cell(row[j])]

    def score_sheet(values: list[str]) -> float:
        if not values:
            return 0.0
        return sum(bool(re.fullmatch(r"\d{4}(?:\s+\d{4})*", v)) for v in values) / len(values)

    def score_index(values: list[str]) -> float:
        if not values:
            return 0.0
        return sum(bool(INDEX_RE.search(v)) for v in values) / len(values)

    def score_count(values: list[str]) -> float:
        if not values:
            return 0.0
        return sum(bool(re.fullmatch(r"[0-9\]\)\}jJ\s]+", v) and "/" not in v) for v in values) / len(values)

    def score_cardnos(values: list[str]) -> float:
        if not values:
            return 0.0
        return sum(bool("/" in v and not INDEX_RE.search(v)) for v in values) / len(values)

    stats = []
    for j in range(ncols):
        vals = non_empty_col_values(j)
        rightness = 0.0 if ncols == 1 else j / (ncols - 1)
        stats.append(
            {
                "j": j,
                "values": vals,
                "rightness": rightness,
                "sheet": score_sheet(vals),
                "index": score_index(vals),
                "count": score_count(vals),
                "card": score_cardnos(vals),
            }
        )

    sheet_col = max(stats, key=lambda s: s["sheet"] + (1.0 - s["rightness"]) * 0.6 - s["count"] * 0.2)["j"]
    count_col = max(
        (s for s in stats if s["j"] != sheet_col),
        key=lambda s: s["count"] + s["rightness"] * 0.7,
    )["j"]

    index_candidates = [s for s in stats if s["j"] not in {sheet_col, count_col}]
    if index_candidates:
        index_col = max(index_candidates, key=lambda s: s["index"] + s["rightness"] * 0.2)["j"]
    else:
        index_col = min(ncols - 1, sheet_col + 1)

    between = [s for s in stats if index_col < s["j"] < count_col]
    has_card_nos = any(s["card"] > 0.2 for s in between)

    if forced_columns is not None:
        target_cols = forced_columns
    else:
        target_cols = 5 if has_card_nos else 4

    def join_cells(row: list[str], cols: list[int]) -> str:
        return clean_cell(" ".join(clean_cell(row[c]) for c in cols if clean_cell(row[c])))

    collapsed: list[list[str]] = []
    for row in matrix:
        if target_cols == 5:
            title_cols = [j for j in range(ncols) if sheet_col < j < index_col]
            card_cols = [j for j in range(ncols) if index_col < j < count_col]
            if not title_cols:
                title_cols = [j for j in range(ncols) if j not in {sheet_col, index_col, count_col} and j < count_col]
            if not card_cols:
                card_cols = [j for j in range(index_col + 1, count_col)]

            new_row = [
                clean_cell(row[sheet_col]),
                join_cells(row, title_cols),
                clean_cell(row[index_col]),
                join_cells(row, card_cols),
                clean_cell(row[count_col]),
            ]
        elif target_cols == 4:
            title_cols = [j for j in range(ncols) if sheet_col < j < index_col]
            if not title_cols:
                title_cols = [j for j in range(ncols) if j not in {sheet_col, index_col, count_col} and j < count_col]
            count_cols = [j for j in range(index_col + 1, ncols) if j == count_col or stats[j]["count"] > 0.5]
            if not count_cols:
                count_cols = [count_col]

            new_row = [
                clean_cell(row[sheet_col]),
                join_cells(row, title_cols),
                clean_cell(row[index_col]),
                join_cells(row, count_cols),
            ]
        else:
            # generic: keep left-to-right best columns
            selected = sorted({sheet_col, index_col, count_col})
            extra = [j for j in range(ncols) if j not in selected]
            selected = (selected + extra)[:target_cols]
            new_row = [clean_cell(row[j]) for j in selected]

        collapsed.append(new_row)

    return collapsed


def trim_and_normalize_matrix(matrix: list[list[str]]) -> list[list[str]]:
    if not matrix:
        return matrix

    # Drop fully empty rows.
    matrix = [row for row in matrix if any(clean_cell(c) for c in row)]
    if not matrix:
        return matrix

    # Remove leading header rows until first 4-digit sheet-like row.
    start = 0
    for i, row in enumerate(matrix):
        row_join = " ".join(row[:2])
        if re.search(r"\b\d{4}\b", row_join):
            start = i
            break
    matrix = matrix[start:]

    # Remove empty columns.
    if matrix:
        keep_cols = [j for j in range(len(matrix[0])) if any(clean_cell(r[j]) for r in matrix)]
        matrix = [[row[j] for j in keep_cols] for row in matrix]

    # Normalize OCR artifacts.
    for row in matrix:
        for j, cell in enumerate(row):
            value = clean_cell(cell)
            value = value.replace("0/2-18-11", "0/2-18-II")
            value = value.replace("0/2-18-111", "0/2-18-III")
            value = value.replace("Ass.Football", "Association Football")
            row[j] = value

    matrix = normalize_ditto_semantics(matrix)

    # Ditto propagation per column.
    for i in range(1, len(matrix)):
        for j in range(len(matrix[i])):
            if is_ditto_mark(matrix[i][j]):
                matrix[i][j] = matrix[i - 1][j]

    return matrix


# --------------------------
# Fallback (OCR-only, generic)
# --------------------------


def group_rows(tokens: list[OCRToken]) -> list[list[OCRToken]]:
    if not tokens:
        return []
    tokens_sorted = sorted(tokens, key=lambda t: t.cy)
    heights = [max(1.0, t.bottom - t.top) for t in tokens_sorted]
    tol = max(10.0, float(np.median(heights)) * 0.65)

    rows: list[list[OCRToken]] = []
    centers: list[float] = []
    for token in tokens_sorted:
        if not rows:
            rows.append([token])
            centers.append(token.cy)
            continue
        if abs(token.cy - centers[-1]) <= tol:
            rows[-1].append(token)
            centers[-1] = float(np.mean([t.cy for t in rows[-1]]))
        else:
            rows.append([token])
            centers.append(token.cy)

    for row in rows:
        row.sort(key=lambda t: t.left)
    return rows


def kmeans_1d(values: np.ndarray, k: int, max_iter: int = 40) -> np.ndarray:
    if len(values) < k:
        return np.linspace(values.min(), values.max(), num=k)

    centers = np.quantile(values, np.linspace(0.1, 0.9, k))
    for _ in range(max_iter):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        new_centers = np.array(
            [values[labels == i].mean() if np.any(labels == i) else centers[i] for i in range(k)],
            dtype=float,
        )
        if np.allclose(new_centers, centers, atol=1e-3):
            break
        centers = new_centers
    return np.sort(centers)


def fallback_matrix_from_ocr(tokens: list[OCRToken]) -> list[list[str]]:
    rows = group_rows(tokens)
    data_rows = [r for r in rows if len(r) >= 2]
    if not data_rows:
        return []

    all_x = np.array([t.cx for row in data_rows for t in row], dtype=float)
    n = len(all_x)
    best_k = 4
    best_bic = float("inf")

    for k in range(2, 8):
        centers = kmeans_1d(all_x, k)
        dist = np.min(np.abs(all_x[:, None] - centers[None, :]), axis=1)
        sse = float(np.sum(dist**2)) + 1e-6
        bic = n * math.log(sse / n) + (2 * k) * math.log(n)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    centers = kmeans_1d(all_x, best_k)
    matrix: list[list[str]] = []

    for row in data_rows:
        cells: list[list[OCRToken]] = [[] for _ in range(best_k)]
        for token in row:
            ci = int(np.argmin(np.abs(centers - token.cx)))
            cells[ci].append(token)
        text_row = []
        for cell_tokens in cells:
            cell_tokens = sorted(cell_tokens, key=lambda t: (t.cy, t.left))
            text_row.append(clean_cell(" ".join(t.text for t in cell_tokens)))
        if sum(1 for c in text_row if c) >= 2:
            matrix.append(text_row)

    return matrix


# --------------------------
# Main extraction flow
# --------------------------


def load_image_rgb(image_path: Path) -> np.ndarray:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.array(image)


def extract_rows_from_image(
    image_path: Path,
    ocr_engine: RapidOCR,
    min_confidence: float,
    forced_columns: int | None,
) -> tuple[list[list[str]], int]:
    image_rgb = load_image_rgb(image_path)
    image_pil = Image.fromarray(image_rgb)

    ocr_result, _ = ocr_engine(image_rgb)
    tokens = parse_ocr_tokens(ocr_result, min_confidence=min_confidence)

    rows: list[list[str]] = []
    detected_cols = 0

    try:
        structure = detect_table_structure_ai(image_pil)
        if structure is not None:
            rows = matrix_from_structure(tokens, structure)
            rows = collapse_columns_by_syntax(rows, forced_columns=forced_columns)
            rows = trim_and_normalize_matrix(rows)
            if rows:
                detected_cols = len(rows[0])
    except Exception as exc:
        print(f"[warn] IA de estrutura falhou em {image_path.name}: {exc}")

    if not rows:
        rows = fallback_matrix_from_ocr(tokens)
        rows = collapse_columns_by_syntax(rows, forced_columns=forced_columns)
        rows = trim_and_normalize_matrix(rows)
        detected_cols = len(rows[0]) if rows else 0

    if forced_columns is not None and rows:
        target = forced_columns
        adjusted: list[list[str]] = []
        for row in rows:
            if len(row) > target:
                adjusted.append(row[:target])
            elif len(row) < target:
                adjusted.append(row + [""] * (target - len(row)))
            else:
                adjusted.append(row)
        rows = adjusted
        detected_cols = target

    return rows, detected_cols


# --------------------------
# IO and CLI
# --------------------------


def save_image_csv(rows: list[list[str]], columns: int, source_image: str, output_path: Path) -> pd.DataFrame:
    if columns == 4:
        col_names = ["Sheet", "Series Title", "World Index Number", "No of Cards"]
    elif columns == 5:
        col_names = ["Sheet", "Series Title", "World Index Number", "Card Nos.", "No of Cards"]
    else:
        col_names = [f"col_{i}" for i in range(1, columns + 1)]

    df = pd.DataFrame(rows, columns=col_names)
    df.insert(0, "row_index", range(1, len(df) + 1))
    df.insert(0, "source_image", source_image)
    df.to_csv(output_path, index=False)
    return df


def collect_image_paths(input_dir: Path) -> list[Path]:
    allowed = {".heic", ".jpg", ".jpeg", ".png"}
    ext_priority = {".heic": 0, ".jpg": 1, ".jpeg": 1, ".png": 2}

    candidates = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed]
    candidates.sort(key=lambda p: (p.stem.lower(), ext_priority.get(p.suffix.lower(), 9), p.name.lower()))

    picked: dict[str, Path] = {}
    for path in candidates:
        key = path.stem.lower()
        if key not in picked:
            picked[key] = path

    return sorted(picked.values(), key=lambda p: p.name.lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrai tabelas de fotos e salva em CSV.")
    parser.add_argument("--input-dir", default="img", help="Pasta com as imagens.")
    parser.add_argument("--output-dir", default="output", help="Pasta de saida dos CSVs.")
    parser.add_argument(
        "--columns",
        default="auto",
        help='Quantidade de colunas: "auto" ou um numero inteiro (ex: 5).',
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Confianca minima do OCR (0 a 1).",
    )
    return parser.parse_args()


def parse_forced_columns(raw_value: str) -> int | None:
    value = raw_value.strip().lower()
    if value == "auto":
        return None
    columns = int(value)
    if columns <= 0:
        raise ValueError("--columns deve ser maior que zero")
    return columns


def main() -> None:
    args = parse_args()
    register_heif_opener()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta de entrada nao encontrada: {input_dir}")

    forced_columns = parse_forced_columns(args.columns)
    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {input_dir}")

    ocr_engine = RapidOCR()
    all_frames: list[pd.DataFrame] = []

    for image_path in image_paths:
        rows, detected_cols = extract_rows_from_image(
            image_path=image_path,
            ocr_engine=ocr_engine,
            min_confidence=args.min_confidence,
            forced_columns=forced_columns,
        )

        out_file = output_dir / f"{image_path.stem}.csv"
        frame = save_image_csv(rows, detected_cols, image_path.name, out_file)
        all_frames.append(frame)
        print(f"[ok] {image_path.name}: {len(frame)} linhas, {detected_cols} colunas -> {out_file}")

    combined = pd.concat(all_frames, ignore_index=True).fillna("")
    combined_out = output_dir / "tabelas_extraidas.csv"
    combined.to_csv(combined_out, index=False)
    print(f"[ok] CSV consolidado: {combined_out}")


if __name__ == "__main__":
    main()
