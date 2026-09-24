"""Draw the README architecture diagrams into docs/images/."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path("docs/images")

NAVY = (15, 23, 42)
SLATE = (51, 65, 85)
BLUE = (37, 99, 235)
GREEN = (22, 163, 74)
AMBER = (180, 83, 9)
RED = (185, 28, 28)
PURPLE = (109, 40, 217)
WHITE = (255, 255, 255)
INK = (15, 23, 42)
MUTED = (71, 85, 105)
LINE = (203, 213, 225)
BG = (248, 250, 252)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates += [
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ]
    candidates += [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=fnt) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int], outline: tuple[int, int, int] | None = None) -> None:
    draw.rounded_rectangle(box, radius=16, fill=fill, outline=outline or fill, width=2)


def label(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str, fill: tuple[int, int, int]) -> None:
    rounded(draw, box, fill)
    x0, y0, x1, _y1 = box
    pad = 18
    tf = font(22, bold=True)
    bf = font(16)
    draw.text((x0 + pad, y0 + 14), title, font=tf, fill=WHITE)
    y = y0 + 48
    for line in wrap(draw, body, bf, (x1 - x0) - pad * 2):
        draw.text((x0 + pad, y), line, font=bf, fill=(241, 245, 249))
        y += 22


def arrow(draw: ImageDraw.ImageDraw, x: int, y0: int, y1: int) -> None:
    draw.line((x, y0, x, y1 - 8), fill=SLATE, width=3)
    draw.polygon([(x - 7, y1 - 10), (x + 7, y1 - 10), (x, y1)], fill=SLATE)


def canvas(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), BG)
    return img, ImageDraw.Draw(img)


def system_context() -> None:
    img, draw = canvas(1100, 640)
    title = font(28, bold=True)
    draw.text((40, 28), "Glasses and spoken guidance", font=title, fill=INK)
    sub = font(16)
    draw.text((40, 68), "The glasses see the path. The service decides. The wearer hears the command.", font=sub, fill=MUTED)
    label(draw, (80, 130, 1020, 250), "Smart glasses", "Forward camera, position, and heading. Any camera can stand in while the glasses build is in progress.", BLUE)
    arrow(draw, 550, 250, 310)
    label(draw, (80, 310, 1020, 430), "Navigation service", "ADE20K SegFormer, depth, obstacle check, and a walking route. Returns one short command.", GREEN)
    arrow(draw, 550, 430, 490)
    label(draw, (80, 490, 1020, 600), "Voice in the ear", "Stop, move left, move right, or go forward. Hands stay free.", PURPLE)
    img.save(OUT / "01-system-context.png")


def pipeline() -> None:
    img, draw = canvas(1100, 980)
    draw.text((40, 24), "Frame pipeline", font=font(28, bold=True), fill=INK)
    steps = [
        (BLUE, "Capture", "Forward camera frame, position, and heading from the glasses."),
        (GREEN, "Segmentation", "ADE20K SegFormer. ONNX INT8 in the cloud, transformers on a laptop."),
        (GREEN, "Depth", "Metric depth from the glasses when present. Otherwise a segmentation proxy."),
        (AMBER, "CARE", "Hazard check from obstacle pixels and depth. Stop overrides the route."),
        (AMBER, "Spatial reasoner", "Picks stop, turn, or go forward. Optional Llama 3.1 can rephrase."),
        (PURPLE, "Phrase and validator", "config/phrases.yaml plus cooldown so the same line is not repeated."),
        (PURPLE, "Speech", "Spoken guidance in the ear."),
    ]
    y = 80
    for color, title, body in steps:
        label(draw, (160, y, 940, y + 90), title, body, color)
        if title != "Speech":
            arrow(draw, 550, y + 90, y + 118)
        y += 118
    img.save(OUT / "02-pipeline-architecture.png")


def decision() -> None:
    img, draw = canvas(1100, 760)
    draw.text((40, 24), "Decision priority", font=font(28, bold=True), fill=INK)
    draw.text((40, 64), "The spatial reasoner checks these in order. The first match wins.", font=font(16), fill=MUTED)
    rows = [
        (RED, "1. Vision stop", "Hazard or obstacle in the walking band. Stop is spoken immediately."),
        (RED, "2. Destination", "Route cue says stop, within about 15 m of the destination."),
        (BLUE, "3. Route", "Forward, left, or right from Google Directions or OSRM, if that side is walkable."),
        (AMBER, "4. Blocked path", "No walkable lane. Slow down."),
        (GREEN, "5. CARE direction", "Turn toward the clearer side, otherwise go forward."),
    ]
    y = 110
    for color, title, body in rows:
        label(draw, (80, y, 1020, y + 100), title, body, color)
        y += 120
    img.save(OUT / "05-decision-priority.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    system_context()
    pipeline()
    decision()
    for path in sorted(OUT.glob("*.png")):
        print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
