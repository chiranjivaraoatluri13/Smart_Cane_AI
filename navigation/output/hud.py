"""On-screen navigation command overlay."""

from __future__ import annotations

import platform
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from navigation.models import NavigationCommand

_COMMAND_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    NavigationCommand.STOP.value: (0, 0, 255),
    NavigationCommand.SLOW_DOWN.value: (0, 140, 255),
    NavigationCommand.GO_FORWARD.value: (0, 220, 0),
    NavigationCommand.MOVE_LEFT.value: (0, 255, 255),
    NavigationCommand.MOVE_RIGHT.value: (0, 255, 255),
}

_ARROW: dict[str, str] = {
    NavigationCommand.STOP.value: "■ STOP",
    NavigationCommand.SLOW_DOWN.value: "▼ SLOW",
    NavigationCommand.GO_FORWARD.value: "▲ FORWARD",
    NavigationCommand.MOVE_LEFT.value: "◀ LEFT",
    NavigationCommand.MOVE_RIGHT.value: "▶ RIGHT",
}

# phone_client.html — #command / #details / #route-status
_LIVE_COMMAND_RGB = (255, 255, 255)
_LIVE_DETAILS_RGB = (221, 221, 221)
_LIVE_SHADOW_RGBA = (0, 0, 0, 204)
_LIVE_ROUTE_PILL_RGBA = (0, 0, 0, 166)  # rgba(0,0,0,0.65)
_LIVE_DETAILS_MAX_WIDTH_PX = 320


def _windows_font_candidates(*, bold: bool) -> list[Path]:
    fonts_dir = Path("C:/Windows/Fonts")
    names = (
        ["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"]
        if bold
        else ["segoeui.ttf", "arial.ttf", "calibri.ttf"]
    )
    return [fonts_dir / name for name in names if (fonts_dir / name).is_file()]


def _unix_font_candidates(*, bold: bool) -> list[Path]:
    roots = (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/System/Library/Fonts/Supplemental"),
    )
    names = (
        ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"]
        if bold
        else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"]
    )
    out: list[Path] = []
    for root in roots:
        for name in names:
            p = root / name
            if p.is_file():
                out.append(p)
    return out


@lru_cache(maxsize=16)
def _load_phone_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    candidates: list[Path] = []
    if platform.system() == "Windows":
        candidates = _windows_font_candidates(bold=bold)
    else:
        candidates = _unix_font_candidates(bold=bold)

    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_live_detail_text(
    *,
    rationale: str = "",
    facts: dict | None = None,
) -> tuple[str, str]:
    """Mirror ``phone_client.html`` details + route-status strip."""
    detail_text = rationale or ""
    route_status = ""
    rc = (facts or {}).get("route_cue")
    if rc:
        turn = rc.get("turn")
        meters = float(rc.get("meters_to_turn") or 0.0)
        feet = int(round(meters * 3.281))
        if turn == "loading":
            detail_text = "⏳ Loading walking route…"
        elif turn == "failed":
            detail_text = (
                "✗ Walking route unavailable — check connection or set destination again"
            )
        elif turn == "stop":
            detail_text = f"📍 Destination in {feet} ft"
        elif turn == "left":
            detail_text = f"↰ Turn left in {feet} ft"
        elif turn == "right":
            detail_text = f"↱ Turn right in {feet} ft"
        elif turn == "forward":
            detail_text = f"↑ Continue {feet} ft"
        route_status = detail_text
    return detail_text, route_status


def _wrap_text(text: str, font, max_width: int, draw) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_text_shadow(
    draw,
    xy: tuple[int, int],
    text: str,
    *,
    font,
    fill: tuple[int, int, int],
    shadow_rgba: tuple[int, int, int, int] = _LIVE_SHADOW_RGBA,
) -> None:
    x, y = xy
    for dx, dy in ((0, 2), (0, 1), (1, 2), (-1, 2), (0, 3)):
        draw.text((x + dx, y + dy), text, font=font, fill=shadow_rgba)
    draw.text((x, y), text, font=font, fill=fill)


def _rounded_rect(draw, box: tuple[int, int, int, int], radius: int, fill) -> None:
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=radius, fill=fill)
        return
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=fill)


def _scale_px(reference: int, h: int, ref_h: int = 800) -> int:
    return max(10, int(reference * h / ref_h))


def draw_navigation_hud(
    bgr: np.ndarray,
    *,
    phrase: str,
    command: str,
    speak: bool,
    confidence: float,
    rationale: str = "",
    stale: bool = False,
) -> np.ndarray:
    """Draw a large banner so the user can read the current instruction.

    ``stale=True`` dims the banner. Use it on frames where the segmenter
    didn't actually run (skipped under ``process_every_n_frames``); the HUD
    still shows the last command but visibly indicates it's not fresh.
    """
    out = bgr.copy()
    h, w = out.shape[:2]
    banner_h = min(100, max(70, h // 4))

    color = _COMMAND_COLORS_BGR.get(command, (255, 255, 255))
    if not speak:
        color = (160, 160, 160)
    if stale:
        # Desaturate toward gray to flag "not from this frame".
        color = tuple(int(c * 0.55 + 90) for c in color)  # type: ignore[assignment]

    cv2.rectangle(out, (0, 0), (w, banner_h), (20, 20, 20), thickness=-1)
    cv2.rectangle(out, (0, 0), (w, banner_h), color, thickness=2)

    label = phrase.upper() if phrase else command.replace("_", " ").upper()
    arrow = _ARROW.get(command, "")
    title = f"{arrow}  {label}" if arrow else label

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(1.4, w / 500)
    thick = 2
    cv2.putText(out, title, (12, int(banner_h * 0.45)), font, scale, color, thick, cv2.LINE_AA)

    sub = f"cmd={command}  conf={confidence:.0%}"
    if not speak:
        sub += "  (silent — cooldown)"
    if stale:
        sub += "  (cached)"
    cv2.putText(
        out,
        sub,
        (12, int(banner_h * 0.78)),
        font,
        scale * 0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    if rationale:
        short = rationale[:60] + ("…" if len(rationale) > 60 else "")
        cv2.putText(
            out,
            short,
            (12, h - 12),
            font,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def draw_sparse_navigation_hud(
    bgr: np.ndarray,
    *,
    phrase: str,
    visible: bool,
    command: str = "",
    rationale: str = "",
    facts: dict | None = None,
    speak: bool = True,
) -> np.ndarray:
    """Live phone HUD: centered uppercase phrase, details, optional route pill.

    Matches ``phone_client.html`` (#command-area, #details, #route-status).
    Uses PIL with Segoe UI (or closest system sans) for web-like typography.
    """
    if not visible or not phrase:
        return bgr

    from PIL import Image, ImageDraw

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    command_text = phrase.strip().upper()
    command_size = _scale_px(48, h)
    details_size = _scale_px(16, h)
    route_size = _scale_px(15, h)
    command_font = _load_phone_font(command_size, bold=True)
    details_font = _load_phone_font(details_size, bold=False)
    route_font = _load_phone_font(route_size, bold=False)

    cmd_bbox = draw.textbbox((0, 0), command_text, font=command_font)
    cmd_w = cmd_bbox[2] - cmd_bbox[0]
    cmd_h = cmd_bbox[3] - cmd_bbox[1]
    cmd_x = max(0, (w - cmd_w) // 2)

    details_text, route_status = build_live_detail_text(
        rationale=rationale, facts=facts
    )
    details_max_w = min(w - 48, int(_LIVE_DETAILS_MAX_WIDTH_PX * w / 390))
    detail_lines = _wrap_text(details_text, details_font, details_max_w, draw)

    details_block_h = 0
    line_heights: list[int] = []
    if detail_lines:
        gap = max(8, _scale_px(12, h))
        for line in detail_lines:
            bb = draw.textbbox((0, 0), line, font=details_font)
            lh = bb[3] - bb[1]
            line_heights.append(lh)
            details_block_h += lh
        details_block_h += gap * max(0, len(detail_lines) - 1)

    route_pill_h = 0
    route_pill_pad_y = max(8, _scale_px(8, h))
    route_pill_pad_x = max(16, _scale_px(20, h))
    bottom_safe = max(16, _scale_px(32, h))
    if route_status:
        rs_bbox = draw.textbbox((0, 0), route_status, font=route_font)
        route_pill_h = (rs_bbox[3] - rs_bbox[1]) + route_pill_pad_y * 2

    block_h = cmd_h + (max(8, _scale_px(12, h)) if detail_lines else 0) + details_block_h
    center_y = h // 2
    if route_status:
        center_y = int((h - route_pill_h - bottom_safe) * 0.48)
    cmd_y = center_y - block_h // 2

    _draw_text_shadow(
        draw,
        (cmd_x, cmd_y),
        command_text,
        font=command_font,
        fill=_LIVE_COMMAND_RGB,
    )

    if detail_lines:
        y = cmd_y + cmd_h + max(8, _scale_px(12, h))
        for i, line in enumerate(detail_lines):
            bb = draw.textbbox((0, 0), line, font=details_font)
            lw = bb[2] - bb[0]
            lx = max(0, (w - lw) // 2)
            _draw_text_shadow(
                draw,
                (lx, y),
                line,
                font=details_font,
                fill=_LIVE_DETAILS_RGB,
                shadow_rgba=(0, 0, 0, 230),
            )
            y += line_heights[i] + max(6, _scale_px(8, h))

    if route_status:
        rs_bbox = draw.textbbox((0, 0), route_status, font=route_font)
        rs_w = rs_bbox[2] - rs_bbox[0]
        rs_h = rs_bbox[3] - rs_bbox[1]
        pill_w = min(w - 40, rs_w + route_pill_pad_x * 2)
        pill_h = rs_h + route_pill_pad_y * 2
        pill_x0 = (w - pill_w) // 2
        pill_y0 = h - bottom_safe - pill_h
        _rounded_rect(
            draw,
            (pill_x0, pill_y0, pill_x0 + pill_w, pill_y0 + pill_h),
            radius=min(20, pill_h // 2),
            fill=_LIVE_ROUTE_PILL_RGBA,
        )
        text_x = pill_x0 + (pill_w - rs_w) // 2
        text_y = pill_y0 + route_pill_pad_y
        draw.text((text_x, text_y), route_status, font=route_font, fill=_LIVE_COMMAND_RGB)

    composed = Image.alpha_composite(base, overlay)
    out_rgb = np.asarray(composed.convert("RGB"))
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def format_command_banner(record: dict) -> str:
    """Human-readable line for the terminal."""
    phrase = record.get("phrase") or str(record.get("command", "")).replace("_", " ")
    speak = record.get("speak", True)
    cmd = record.get("command", "")
    conf = record.get("confidence", 0)
    silent = "" if speak else " [silent — cooldown]"
    return f">>> {phrase.upper()}  ({cmd}, {conf:.0%}){silent}"
