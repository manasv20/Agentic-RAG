#!/usr/bin/env python3
"""
Generate the Agentic RAG workflow diagram to match the reference layout:
- App_Core -> Sidebar at top; four grey-shaded section blocks below
- Branching flows inside each section (e.g. OpenCV | CLIP merging into ChromaDB)
- Colors: White (steps), Light Blue (agents), Orange (tools), Pink (DB/core)
"""
from PIL import Image, ImageDraw, ImageFont
import os

W = 2000
H = 1100
MARGIN = 32
SECTION_PAD = 16
BOX_W = 128
BOX_H = 32
ARROW_W = 24
FONT_SIZE = 10
TITLE_FONT_SIZE = 13
SECTION_GRAY = (248, 248, 248)
BORDER_GRAY = (200, 200, 200)
WHITE = (255, 255, 255)
LIGHT_BLUE = (173, 216, 230)
ORANGE = (255, 165, 0)
PINK = (255, 182, 193)
BLACK = (0, 0, 0)
DARK_GRAY = (70, 70, 70)

def get_font(size):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()

def draw_box(draw, x, y, w, h, fill, text, font):
    draw.rectangle([x, y, x + w, y + h], fill=fill, outline=DARK_GRAY, width=1)
    b = draw.textbbox((0, 0), text, font=font)
    tw, th = b[2] - b[0], b[3] - b[1]
    draw.text((x + (w - tw) // 2, y + (h - th) // 2), text, fill=BLACK, font=font)

def arrow_down(draw, x, y, length=20):
    draw.line([x, y, x, y + length - 6], fill=BLACK, width=2)
    draw.polygon([(x, y + length), (x - 4, y + length - 8), (x + 4, y + length - 8)], fill=BLACK)

def arrow_right(draw, x, y, length=20):
    draw.line([x, y, x + length - 6, y], fill=BLACK, width=2)
    draw.polygon([(x + length, y), (x + length - 8, y - 4), (x + length - 8, y + 4)], fill=BLACK)

def main():
    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)
    font = get_font(FONT_SIZE)
    title_font = get_font(TITLE_FONT_SIZE)

    y = MARGIN
    # Top: App_Core (pink) -> Sidebar (white)
    draw_box(draw, MARGIN, y, 110, BOX_H, PINK, "App_Core", font)
    arrow_right(draw, MARGIN + 115, y + BOX_H // 2, ARROW_W)
    draw_box(draw, MARGIN + 115 + ARROW_W, y, 72, BOX_H, WHITE, "Sidebar", font)
    y += BOX_H + 24

    # Four section columns (grey blocks with title)
    col_w = (W - 2 * MARGIN - 3 * SECTION_PAD) // 4
    section_h = H - y - MARGIN - 40  # leave room for legend
    sections = [
        "Video Processing",
        "Audio Processing",
        "Document Processing",
        "Query & Inference",
    ]
    for i, title in enumerate(sections):
        sx = MARGIN + i * (col_w + SECTION_PAD)
        draw.rectangle([sx, y, sx + col_w, y + section_h], fill=SECTION_GRAY, outline=BORDER_GRAY, width=1)
        draw.text((sx + 10, y + 8), title, font=title_font, fill=DARK_GRAY)
    y += 36
    start_y = y
    content_h = section_h - 36
    box_gap = 8
    arrow_len = 18

    def col_x(c):
        return MARGIN + c * (col_w + SECTION_PAD) + 12

    def center_x(c):
        return col_x(c) + (col_w - 24) // 2 - BOX_W // 2

    # ---- Video Processing: upload -> Frame Extraction -> [OpenCV->V_CV | CLIP Image Embed->CLIP Text Embed] -> ChromaDB Collections -> DB ----
    cx = center_x(0)
    cy = start_y
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "Video upload", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "Frame Extraction", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    # Branch: two columns
    left_x, right_x = cx - 8, cx + BOX_W + 8
    draw_box(draw, left_x, cy, BOX_W - 20, BOX_H, WHITE, "OpenCV", font)
    draw_box(draw, right_x, cy, BOX_W - 20, BOX_H, WHITE, "CLIP Image Embed", font)
    cy += BOX_H + box_gap
    arrow_down(draw, left_x + (BOX_W - 20) // 2, cy - box_gap, arrow_len)
    arrow_down(draw, right_x + (BOX_W - 20) // 2, cy - box_gap, arrow_len)
    draw_box(draw, left_x, cy, BOX_W - 20, BOX_H, ORANGE, "V_CV", font)
    draw_box(draw, right_x, cy, BOX_W - 20, BOX_H, ORANGE, "CLIP Text Embed", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, PINK, "ChromaDB Collections", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, PINK, "DB", font)

    # ---- Audio: upload -> Transcription -> [Agent sample_chars, Agent chunk_params] -> Chunking -> A_Emb -> Ollama Embeddings -> DB ----
    cx = center_x(1)
    cy = start_y
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "Audio upload", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "Transcription", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx - 24, cy, BOX_W - 16, BOX_H, LIGHT_BLUE, "Agent: sample_chars", font)
    draw_box(draw, cx + 20, cy, BOX_W - 16, BOX_H, LIGHT_BLUE, "Agent: chunk_params", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "Chunking", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, ORANGE, "A_Emb", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, ORANGE, "Ollama Embeddings", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, PINK, "DB", font)

    # ---- Document: PDF upload -> PyPDF2 -> [Agent sample_pages, Agent chunk_params] -> Recursive/Table/Section Chunking -> [Optional Metadata, D_Ext] -> Ollama Embeddings -> DB ----
    cx = center_x(2)
    cy = start_y
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "PDF upload", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "PyPDF2 Text Extraction", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx - 24, cy, BOX_W - 16, BOX_H, LIGHT_BLUE, "Agent: sample_pages", font)
    draw_box(draw, cx + 20, cy, BOX_W - 16, BOX_H, LIGHT_BLUE, "Agent: chunk_params", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W + 10, BOX_H, WHITE, "Recursive/Table/Section Chunking", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx - 28, cy, BOX_W - 8, BOX_H, WHITE, "Optional Metadata Extraction", font)
    draw_box(draw, cx + 22, cy, BOX_W - 30, BOX_H, ORANGE, "D_Ext", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, ORANGE, "Ollama Embeddings", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, PINK, "DB", font)

    # ---- Query & Inference: User Question -> System prompt -> Select collection -> Query ChromaDB -> [Retrieve context, search_documents] -> Ollama Chat Agent -> Evaluator Agent -> Validation -> Final Answer ----
    cx = center_x(3)
    cy = start_y
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "User Question", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "System prompt", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "Select collection", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, PINK, "Query ChromaDB", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx - 24, cy, BOX_W - 16, BOX_H, WHITE, "Retrieve context", font)
    draw_box(draw, cx + 20, cy, BOX_W - 16, BOX_H, ORANGE, "search_documents", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, LIGHT_BLUE, "Ollama Chat Agent", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, LIGHT_BLUE, "Evaluator Agent", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "Validation", font)
    cy += BOX_H + box_gap
    arrow_down(draw, cx + BOX_W // 2, cy - box_gap, arrow_len)
    draw_box(draw, cx, cy, BOX_W, BOX_H, WHITE, "Final Answer", font)

    # Legend
    leg_y = H - 48
    draw.text((MARGIN, leg_y - 4), "Legend:", font=title_font, fill=DARK_GRAY)
    draw_box(draw, MARGIN, leg_y + 2, 88, 24, WHITE, "Step", font)
    draw_box(draw, MARGIN + 100, leg_y + 2, 88, 24, LIGHT_BLUE, "Agent", font)
    draw_box(draw, MARGIN + 200, leg_y + 2, 88, 24, ORANGE, "Tool / Embed", font)
    draw_box(draw, MARGIN + 300, leg_y + 2, 88, 24, PINK, "DB / Core", font)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "agentic-rag-workflow.png")
    img.save(out_path, "PNG")
    print(f"Saved: {os.path.abspath(out_path)}")
    return out_path

if __name__ == "__main__":
    main()
