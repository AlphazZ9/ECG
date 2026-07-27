# -*- coding: utf-8 -*-
"""
ecg.ui.sidebar
--------------
Reusable components:
  CollapsibleSection  -- collapsible labelled section (left sidebar + right
                          accordion panel)
"""
from __future__ import annotations

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.ui.theme import (
    BORDER, BORDER2, TEXT,
    FONT_SIDEBAR_HDR,
    SPACE_S, SPACE_M,
)


class CollapsibleSection:
    """A collapsible section with a clickable arrow header.

    Shared by the left sidebar and the right accordion panel. The outer
    ``_wrapper`` frame is always packed (preserving its position in the
    parent's pack order).  Only the inner ``frame`` is shown/hidden, so
    re-opening a section never moves it to the bottom of the list.

    Usage::
        sec = CollapsibleSection(parent, "FILTERS", initially_open=True)
        ctk.CTkLabel(sec.frame, text="HP cut").pack(...)
    """

    def __init__(self, parent, title: str, initially_open: bool = True) -> None:
        self._title = title
        self._open  = initially_open

        # Outer wrapper — ALWAYS packed; never removed from layout
        self._wrapper = ctk.CTkFrame(parent, fg_color="transparent")
        self._wrapper.pack(fill="x")

        # ── Header row ──────────────────────────────────────────────
        # Compact + token-based spacing (was hardcoded padx=10, pady=(8,0),
        # height=28 — inconsistent with the SPACE_* scale used everywhere
        # else in the sidebar, and taller than needed across 5 stacked
        # sections). text_color raised from MUTED to TEXT so section titles
        # read clearly at a glance instead of blending into the background.
        hdr = ctk.CTkFrame(self._wrapper, fg_color=BORDER, corner_radius=6, height=26)
        hdr.pack(fill="x", padx=SPACE_M, pady=(SPACE_S, 0))
        hdr.pack_propagate(False)

        arrow = "▾" if initially_open else "▸"
        self._btn = ctk.CTkButton(
            hdr, text=f"  {arrow}  {title}",
            font=FONT_SIDEBAR_HDR, text_color=TEXT,
            fg_color="transparent",
            hover_color=BORDER2,
            anchor="w", height=26, corner_radius=6,
            command=self._toggle,
        )
        self._btn.pack(fill="x")
        # ── Content frame — inside the wrapper, shown or hidden ──────
        self.frame = ctk.CTkFrame(self._wrapper, fg_color="transparent")
        if initially_open:
            self.frame.pack(fill="x")

    def _toggle(self) -> None:
        self._open = not self._open
        arrow = "▾" if self._open else "▸"
        self._btn.configure(text=f"  {arrow}  {self._title}")
        if self._open:
            self.frame.pack(fill="x")
        else:
            self.frame.pack_forget()

    def open(self) -> None:
        if not self._open:
            self._toggle()

    def close(self) -> None:
        if self._open:
            self._toggle()
