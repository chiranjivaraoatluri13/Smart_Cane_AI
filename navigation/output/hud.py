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


def _horizontal_padding(w: int, *, ratio: float = 0.10) -> int:
    return max(12, int(w * ratio))


def _safe_text_width(w: int, *, ratio: float = 0.10) -> int:
    pad = _horizontal_padding(w, ratio=ratio)
    return max(32, w - 2 * pad)


def _line_metrics(draw, line: str, font) -> tuple[int, int]:
    bb = draw.textbbox((0, 0), line, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def _wrapped_block_metrics(
    draw, lines: list[str], font, *, line_gap: int
) -> tuple[int, int, list[int], list[int]]:
    line_widths: list[int] = []
    line_heights: list[int] = []
    for line in lines:
        lw, lh = _line_metrics(draw, line, font)
        line_widths.append(lw)
        line_heights.append(lh)
    block_w = max(line_widths, default=0)
    block_h = 0
    if line_heights:
        block_h = sum(line_heights) + line_gap * max(0, len(line_heights) - 1)
    return block_w, block_h, line_widths, line_heights


def _fit_wrapped_text(
    text: str,
    draw,
    *,
    start_size: int,
    min_size: int,
    max_width: int,
    bold: bool,
    line_gap: int,
    max_block_height: int | None = None,
) -> tuple[object, list[str], list[int], list[int], int]:
    """Shrink font until wrapped text fits width (and optional height)."""
    size = start_size
    step = max(1, start_size // 12)
    font = _load_phone_font(size, bold=bold)
    lines: list[str] = []
    line_widths: list[int] = []
    line_heights: list[int] = []
    block_h = 0

    while size >= min_size:
        font = _load_phone_font(size, bold=bold)
        lines = _wrap_text(text, font, max_width, draw)
        block_w, block_h, line_widths, line_heights = _wrapped_block_metrics(
            draw, lines, font, line_gap=line_gap
        )
        fits_width = block_w <= max_width
        fits_height = max_block_height is None or block_h <= max_block_height
        if fits_width and fits_height:
            return font, lines, line_widths, line_heights, block_h
        size -= step

    font = _load_phone_font(min_size, bold=bold)
    lines = _wrap_text(text, font, max_width, draw)
    _, block_h, line_widths, line_heights = _wrapped_block_metrics(
        draw, lines, font, line_gap=line_gap
    )
    return font, lines, line_widths, line_heights, block_h


def _draw_wrapped_shadow_block(
    draw,
    *,
    lines: list[str],
    line_heights: list[int],
    line_widths: list[int],
    font,
    fill: tuple[int, int, int],
    shadow_rgba: tuple[int, int, int, int],
    frame_w: int,
    start_y: int,
    line_gap: int,
) -> int:
    """Draw centered wrapped lines; return y after last line."""
    y = start_y
    for i, line in enumerate(lines):
        lx = max(0, (frame_w - line_widths[i]) // 2)
        _draw_text_shadow(
            draw,
            (lx, y),
            line,
            font=font,
            fill=fill,
            shadow_rgba=shadow_rgba,
        )
        y += line_heights[i]
        if i < len(lines) - 1:
            y += line_gap
    return y


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

    safe_w = _safe_text_width(w)
    top_safe = max(16, _scale_px(24, h))
    bottom_safe = max(16, _scale_px(32, h))
    cmd_line_gap = max(4, _scale_px(6, h))
    detail_line_gap = max(6, _scale_px(8, h))
    cmd_detail_gap = max(8, _scale_px(12, h))

    command_text = phrase.strip().upper()
    command_start = _scale_px(48, h)
    command_min = max(18, _scale_px(24, h))
    details_start = _scale_px(16, h)
    details_min = max(11, _scale_px(12, h))
    route_start = _scale_px(15, h)
    route_min = max(10, _scale_px(11, h))
    route_pill_pad_y = max(8, _scale_px(8, h))
    route_pill_pad_x = max(16, _scale_px(20, h))

    details_text, route_status = build_live_detail_text(
        rationale=rationale, facts=facts
    )

    route_pill_h = 0
    route_font = _load_phone_font(route_start, bold=False)
    route_lines: list[str] = []
    route_line_heights: list[int] = []
    route_line_widths: list[int] = []
    route_block_h = 0
    route_inner_w = safe_w - route_pill_pad_x * 2
    if route_status:
        route_font, route_lines, route_line_widths, route_line_heights, route_block_h = (
            _fit_wrapped_text(
                route_status,
                draw,
                start_size=route_start,
                min_size=route_min,
                max_width=max(32, route_inner_w),
                bold=False,
                line_gap=max(4, _scale_px(4, h)),
            )
        )
        route_pill_h = route_block_h + route_pill_pad_y * 2

    available_h = h - top_safe - bottom_safe - route_pill_h
    cmd_max_h = available_h
    if details_text:
        cmd_max_h = int(available_h * 0.72)

    command_font, cmd_lines, cmd_line_widths, cmd_line_heights, cmd_block_h = (
        _fit_wrapped_text(
            command_text,
            draw,
            start_size=command_start,
            min_size=command_min,
            max_width=safe_w,
            bold=True,
            line_gap=cmd_line_gap,
            max_block_height=cmd_max_h,
        )
    )

    detail_lines: list[str] = []
    detail_line_widths: list[int] = []
    detail_line_heights: list[int] = []
    details_block_h = 0
    details_font = _load_phone_font(details_start, bold=False)
    if details_text:
        details_font, detail_lines, detail_line_widths, detail_line_heights, details_block_h = (
            _fit_wrapped_text(
                details_text,
                draw,
                start_size=details_start,
                min_size=details_min,
                max_width=safe_w,
                bold=False,
                line_gap=detail_line_gap,
                max_block_height=max(0, available_h - cmd_block_h - cmd_detail_gap),
            )
        )

    block_h = cmd_block_h
    if detail_lines:
        block_h += cmd_detail_gap + details_block_h

    center_y = top_safe + available_h // 2
    if route_status:
        center_y = top_safe + int(available_h * 0.46)
    block_top = max(top_safe, center_y - block_h // 2)
    if block_top + block_h > h - bottom_safe - route_pill_h:
        block_top = max(top_safe, h - bottom_safe - route_pill_h - block_h)

    y = block_top
    y = _draw_wrapped_shadow_block(
        draw,
        lines=cmd_lines,
        line_heights=cmd_line_heights,
        line_widths=cmd_line_widths,
        font=command_font,
        fill=_LIVE_COMMAND_RGB,
        shadow_rgba=_LIVE_SHADOW_RGBA,
        frame_w=w,
        start_y=y,
        line_gap=cmd_line_gap,
    )

    if detail_lines:
        y += cmd_detail_gap
        _draw_wrapped_shadow_block(
            draw,
            lines=detail_lines,
            line_heights=detail_line_heights,
            line_widths=detail_line_widths,
            font=details_font,
            fill=_LIVE_DETAILS_RGB,
            shadow_rgba=(0, 0, 0, 230),
            frame_w=w,
            start_y=y,
            line_gap=detail_line_gap,
        )

    if route_status and route_lines:
        pill_w = min(safe_w, max(route_line_widths) + route_pill_pad_x * 2)
        pill_h = route_block_h + route_pill_pad_y * 2
        pill_x0 = max(_horizontal_padding(w), (w - pill_w) // 2)
        pill_y0 = h - bottom_safe - pill_h
        _rounded_rect(
            draw,
            (pill_x0, pill_y0, pill_x0 + pill_w, pill_y0 + pill_h),
            radius=min(20, pill_h // 2),
            fill=_LIVE_ROUTE_PILL_RGBA,
        )
        text_y = pill_y0 + route_pill_pad_y
        route_line_gap = max(4, _scale_px(4, h))
        for i, line in enumerate(route_lines):
            text_x = pill_x0 + (pill_w - route_line_widths[i]) // 2
            draw.text(
                (text_x, text_y),
                line,
                font=route_font,
                fill=_LIVE_COMMAND_RGB,
            )
            text_y += route_line_heights[i] + route_line_gap

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
