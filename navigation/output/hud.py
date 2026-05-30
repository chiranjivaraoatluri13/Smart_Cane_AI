"""On-screen navigation command overlay."""

from __future__ import annotations

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


def format_command_banner(record: dict) -> str:
    """Human-readable line for the terminal."""
    phrase = record.get("phrase") or str(record.get("command", "")).replace("_", " ")
    speak = record.get("speak", True)
    cmd = record.get("command", "")
    conf = record.get("confidence", 0)
    silent = "" if speak else " [silent — cooldown]"
    return f">>> {phrase.upper()}  ({cmd}, {conf:.0%}){silent}"
