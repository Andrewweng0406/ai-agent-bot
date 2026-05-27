"""
WengStock AI — LINE Rich Menu 一鍵建立腳本
執行後自動：生成圖片 → 上傳 → 建立選單 → 設為預設
"""
import os, json, sys
import httpx
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
if not TOKEN:
    sys.exit("❌ LINE_CHANNEL_ACCESS_TOKEN 未設定")

BASE   = "https://api.line.me/v2/bot"
HEADS  = {"Authorization": f"Bearer {TOKEN}"}
IMG_W, IMG_H = 2500, 1686
CELL_W       = IMG_W // 3
CELL_H       = IMG_H // 2

# ─── 顏色 ───────────────────────────────────────
BG       = "#0D1117"
DIVIDER  = "#2A2A3A"
HOVER    = "#141C26"
GREEN    = "#00E676"
YELLOW   = "#FFD600"
TEAL     = "#00BFA5"
BLUE     = "#82B1FF"
WHITE    = "#FFFFFF"
GRAY     = "#808080"

# ─── 6 格定義 ────────────────────────────────────
CELLS = [
    # row 0
    {"label": "掃描美股",   "sub": "/scan 美股",   "color": GREEN,  "cmd": "/scan 美股"},
    {"label": "猩猩掃描",   "sub": "scan gorilla", "color": YELLOW, "cmd": "scan gorilla"},
    {"label": "盤前簡報",   "sub": "/morning",     "color": TEAL,   "cmd": "/morning"},
    # row 1
    {"label": "我的自選",   "sub": "/mywatchlist", "color": BLUE,   "cmd": "/mywatchlist"},
    {"label": "猩猩持倉",   "sub": "/gpositions",  "color": YELLOW, "cmd": "/gpositions"},
    {"label": "大盤風向",   "sub": "/gmarket",     "color": GREEN,  "cmd": "/gmarket"},
]


def hex2rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """載入支援中文的字型，依序嘗試 macOS / Ubuntu / fallback。"""
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",                        # macOS
        "/Library/Fonts/Arial Unicode.ttf",                                # macOS (backup)
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",          # Ubuntu
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_image() -> str:
    img  = Image.new("RGB", (IMG_W, IMG_H), hex2rgb(BG))
    draw = ImageDraw.Draw(img)

    # 字型尺寸：在 2500px 寬圖裡，200px ≈ 手機上 30px，很清晰
    font_main = _load_font(185)
    font_sub  = _load_font(120)

    for i, cell in enumerate(CELLS):
        row, col = divmod(i, 3)
        x0 = col * CELL_W
        y0 = row * CELL_H
        x1 = x0 + CELL_W
        y1 = y0 + CELL_H

        # 底色交錯（讓格子有層次感）
        fill = hex2rgb(HOVER) if (row + col) % 2 == 0 else hex2rgb(BG)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill)

        # 頂部彩色橫條（16px 粗，視覺分類用）
        draw.rectangle([x0, y0, x1 - 1, y0 + 16], fill=hex2rgb(cell["color"]))

        cx = x0 + CELL_W // 2

        # 指令提示（小字，上方）
        cmd_y = y0 + int(CELL_H * 0.38)
        draw.text((cx, cmd_y), cell["sub"], font=font_sub,
                  fill=hex2rgb(cell["color"]), anchor="mm")

        # 主標籤（大字，置中偏下）
        label_y = y0 + int(CELL_H * 0.65)
        draw.text((cx, label_y), cell["label"], font=font_main,
                  fill=hex2rgb(WHITE), anchor="mm")

    # 格線
    for col in range(1, 3):
        x = col * CELL_W
        draw.line([(x, 0), (x, IMG_H)], fill=hex2rgb(DIVIDER), width=6)
    draw.line([(0, CELL_H), (IMG_W, CELL_H)], fill=hex2rgb(DIVIDER), width=6)

    path = "/tmp/rich_menu.png"
    img.save(path, "PNG")
    print(f"✅ 圖片生成：{path}  ({IMG_W}×{IMG_H})")
    return path


def create_menu() -> str:
    menu = {
        "size":        {"width": IMG_W, "height": IMG_H},
        "selected":    True,
        "name":        "WengStock AI Menu",
        "chatBarText": "📊 WengStock 選單",
        "areas": [
            {
                "bounds": {
                    "x": (i % 3) * CELL_W,
                    "y": (i // 3) * CELL_H,
                    "width":  CELL_W,
                    "height": CELL_H,
                },
                "action": {"type": "message", "text": cell["cmd"]},
            }
            for i, cell in enumerate(CELLS)
        ],
    }
    resp = httpx.post(f"{BASE}/richmenu", headers={**HEADS, "Content-Type": "application/json"},
                      content=json.dumps(menu))
    resp.raise_for_status()
    menu_id = resp.json()["richMenuId"]
    print(f"✅ Rich Menu 建立：{menu_id}")
    return menu_id


def upload_image(menu_id: str, img_path: str) -> None:
    with open(img_path, "rb") as f:
        data = f.read()
    resp = httpx.post(
        f"https://api-data.line.me/v2/bot/richmenu/{menu_id}/content",
        headers={**HEADS, "Content-Type": "image/png"},
        content=data,
        timeout=60,
    )
    resp.raise_for_status()
    print("✅ 圖片上傳完成")


def set_default(menu_id: str) -> None:
    resp = httpx.post(f"{BASE}/user/all/richmenu/{menu_id}", headers=HEADS)
    resp.raise_for_status()
    print(f"✅ 已設為預設選單：{menu_id}")


def delete_old_menus() -> None:
    resp = httpx.get(f"{BASE}/richmenu/list", headers=HEADS)
    if resp.status_code != 200:
        return
    menus = resp.json().get("richmenus", [])
    for m in menus:
        mid = m["richMenuId"]
        httpx.delete(f"{BASE}/richmenu/{mid}", headers=HEADS)
        print(f"🗑️  刪除舊選單：{mid}")


if __name__ == "__main__":
    print("🚀 WengStock AI Rich Menu 建立中...\n")
    delete_old_menus()
    img_path = build_image()
    menu_id  = create_menu()
    upload_image(menu_id, img_path)
    set_default(menu_id)
    print("\n🎉 完成！重新開啟 LINE 對話就能看到選單。")
