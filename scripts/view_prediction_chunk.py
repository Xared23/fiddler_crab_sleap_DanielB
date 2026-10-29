#!/usr/bin/env python3
"""Serve a lightweight browser viewer for one SLEAP prediction chunk."""

from __future__ import annotations

import argparse
import json
import math
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import sleap_io as sio


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SLEAP Chunk Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #181818;
      --panel: #222;
      --panel-2: #292929;
      --text: #ededed;
      --muted: #aaa;
      --border: #3a3a3a;
      --accent: #6ea8fe;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    #chunk-name {
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-columns: minmax(250px, 310px) minmax(0, 1fr);
      height: calc(100vh - 51px);
    }
    aside {
      overflow-y: auto;
      padding: 14px;
      border-right: 1px solid var(--border);
      background: var(--panel);
    }
    section + section { margin-top: 18px; }
    h2 {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    button, input {
      accent-color: var(--accent);
      font: inherit;
    }
    button {
      min-height: 30px;
      padding: 4px 9px;
      border: 1px solid var(--border);
      border-radius: 5px;
      background: var(--panel-2);
      color: var(--text);
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button:disabled { cursor: default; opacity: .45; }
    .row { display: flex; align-items: center; gap: 7px; }
    .row + .row { margin-top: 8px; }
    .row.wrap { flex-wrap: wrap; }
    .grow { flex: 1; min-width: 0; }
    label {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
    }
    input[type="range"] { width: 100%; }
    input[type="number"] {
      width: 96px;
      padding: 5px 7px;
      border: 1px solid var(--border);
      border-radius: 4px;
      background: var(--bg);
      color: var(--text);
    }
    input[type="checkbox"] { margin: 0; }
    .value {
      min-width: 42px;
      color: var(--text);
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .metric {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 5px 0;
      border-bottom: 1px solid var(--border);
    }
    .metric span:first-child { color: var(--muted); }
    .metric span:last-child {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    #legend { display: grid; gap: 5px; }
    .legend-item { display: flex; align-items: center; gap: 7px; }
    .swatch { width: 10px; height: 10px; border-radius: 50%; }
    #stage {
      position: relative;
      overflow: auto;
      min-width: 0;
      background: #0c0c0c;
    }
    #frame-image {
      display: block;
      width: 100%;
      max-width: none;
      height: auto;
      image-rendering: auto;
    }
    #status {
      position: sticky;
      left: 12px;
      bottom: 12px;
      display: inline-block;
      margin: 12px;
      padding: 5px 8px;
      border: 1px solid var(--border);
      border-radius: 4px;
      background: rgba(20, 20, 20, .88);
      color: var(--muted);
      pointer-events: none;
    }
    .hint { color: var(--muted); font-size: 12px; }
    .error { color: #ff8c8c !important; }
    @media (max-width: 760px) {
      main { display: block; height: auto; }
      aside { border-right: 0; border-bottom: 1px solid var(--border); }
      #stage { min-height: 55vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>SLEAP 40-frame chunk viewer</h1>
    <div id="chunk-name">Loading…</div>
  </header>
  <main>
    <aside>
      <section>
        <h2>Frame</h2>
        <div class="row">
          <button id="previous" title="Previous frame (←)">←</button>
          <button id="play" title="Play/pause (Space)">Play</button>
          <button id="next" title="Next frame (→)">→</button>
          <button id="random" class="grow" title="Random frame (R)">Random</button>
        </div>
        <div class="row">
          <input id="position" class="grow" type="range" min="0" max="39" value="0">
          <input id="frame-number" type="number">
        </div>
      </section>

      <section>
        <h2>Confidence filters</h2>
        <label>
          <span>Whole crab</span>
          <span id="instance-value" class="value">0.00</span>
        </label>
        <input id="instance-threshold" type="range" min="0" max="1" step="0.01" value="0">
        <label>
          <span>Individual point</span>
          <span id="point-value" class="value">0.20</span>
        </label>
        <input id="point-threshold" type="range" min="0" max="1" step="0.01" value="0.2">
      </section>

      <section>
        <h2>Overlay</h2>
        <div class="row wrap">
          <label><input id="show-points" type="checkbox" checked> Points</label>
          <label><input id="show-edges" type="checkbox" checked> Edges</label>
          <label><input id="show-labels" type="checkbox"> Names</label>
        </div>
        <div id="legend" style="margin-top: 10px"></div>
      </section>

      <section>
        <h2>Zoom</h2>
        <div class="row">
          <button id="zoom-out">−</button>
          <button id="zoom-fit" class="grow">Fit</button>
          <button id="zoom-in">+</button>
          <span id="zoom-value" class="value">100%</span>
        </div>
      </section>

      <section>
        <h2>Current frame</h2>
        <div class="metric"><span>Frame</span><span id="current-frame">—</span></div>
        <div class="metric"><span>All detections</span><span id="total-count">—</span></div>
        <div class="metric"><span>After filter</span><span id="visible-count">—</span></div>
      </section>

      <section>
        <h2>Chunk</h2>
        <div class="metric"><span>Frames</span><span id="frame-range">—</span></div>
        <div class="metric"><span>Video</span><span id="video-name">—</span></div>
        <div class="metric"><span>Video shape</span><span id="video-shape">—</span></div>
        <div class="hint" style="margin-top: 10px">
          Shortcuts: ←/→ frames, Space play, R random, +/− zoom.
        </div>
      </section>
    </aside>

    <div id="stage">
      <img id="frame-image" alt="Annotated prediction frame">
      <div id="status">Loading chunk…</div>
    </div>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const elements = {
      chunkName: $("chunk-name"),
      previous: $("previous"),
      play: $("play"),
      next: $("next"),
      random: $("random"),
      position: $("position"),
      frameNumber: $("frame-number"),
      instanceThreshold: $("instance-threshold"),
      instanceValue: $("instance-value"),
      pointThreshold: $("point-threshold"),
      pointValue: $("point-value"),
      showPoints: $("show-points"),
      showEdges: $("show-edges"),
      showLabels: $("show-labels"),
      legend: $("legend"),
      zoomOut: $("zoom-out"),
      zoomFit: $("zoom-fit"),
      zoomIn: $("zoom-in"),
      zoomValue: $("zoom-value"),
      currentFrame: $("current-frame"),
      totalCount: $("total-count"),
      visibleCount: $("visible-count"),
      frameRange: $("frame-range"),
      videoName: $("video-name"),
      videoShape: $("video-shape"),
      image: $("frame-image"),
      stage: $("stage"),
      status: $("status"),
    };

    let info = null;
    let position = 0;
    let zoom = 1;
    let playTimer = null;
    let isPlaying = false;
    let imageUrl = null;
    let requestNumber = 0;
    let debounceTimer = null;

    function fileName(path) {
      return path.split(/[\\/]/).pop();
    }

    function currentFrame() {
      return info.frames[position];
    }

    function setStatus(message, isError = false) {
      elements.status.textContent = message;
      elements.status.classList.toggle("error", isError);
    }

    function updateFrameControls() {
      const frame = currentFrame();
      elements.position.value = String(position);
      elements.frameNumber.value = String(frame);
      elements.currentFrame.textContent = frame.toLocaleString();
      elements.previous.disabled = position === 0;
      elements.next.disabled = position === info.frames.length - 1;
    }

    async function setPosition(nextPosition) {
      position = Math.max(0, Math.min(info.frames.length - 1, nextPosition));
      updateFrameControls();
      await loadFrame();
    }

    function setZoom(nextZoom) {
      zoom = Math.max(0.25, Math.min(4, nextZoom));
      elements.image.style.width = `${zoom * 100}%`;
      elements.zoomValue.textContent = `${Math.round(zoom * 100)}%`;
    }

    function stopPlayback() {
      isPlaying = false;
      if (playTimer !== null) {
        clearTimeout(playTimer);
        playTimer = null;
      }
      elements.play.textContent = "Play";
    }

    async function advancePlayback() {
      if (!isPlaying) return;
      if (position >= info.frames.length - 1) {
        stopPlayback();
        return;
      }
      await setPosition(position + 1);
      if (isPlaying) {
        playTimer = setTimeout(advancePlayback, 100);
      }
    }

    function togglePlayback() {
      if (isPlaying) {
        stopPlayback();
        return;
      }
      isPlaying = true;
      elements.play.textContent = "Pause";
      advancePlayback();
    }

    function scheduleLoad() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(loadFrame, 80);
    }

    async function loadFrame() {
      const thisRequest = ++requestNumber;
      const frame = currentFrame();
      const params = new URLSearchParams({
        frame: String(frame),
        instance_threshold: elements.instanceThreshold.value,
        point_threshold: elements.pointThreshold.value,
        points: elements.showPoints.checked ? "1" : "0",
        edges: elements.showEdges.checked ? "1" : "0",
        labels: elements.showLabels.checked ? "1" : "0",
      });
      setStatus(`Loading frame ${frame.toLocaleString()}…`);
      try {
        const response = await fetch(`/api/frame?${params}`);
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || `HTTP ${response.status}`);
        }
        const blob = await response.blob();
        if (thisRequest !== requestNumber) return;
        if (imageUrl !== null) URL.revokeObjectURL(imageUrl);
        imageUrl = URL.createObjectURL(blob);
        elements.image.src = imageUrl;
        elements.totalCount.textContent =
          Number(response.headers.get("X-Total-Instances")).toLocaleString();
        elements.visibleCount.textContent =
          Number(response.headers.get("X-Visible-Instances")).toLocaleString();
        setStatus(`Frame ${frame.toLocaleString()} ready`);
      } catch (error) {
        if (thisRequest === requestNumber) {
          setStatus(`Could not load frame: ${error.message}`, true);
        }
      }
    }

    async function initialize() {
      try {
        const response = await fetch("/api/info");
        if (!response.ok) throw new Error(await response.text());
        info = await response.json();
        elements.chunkName.textContent = fileName(info.chunk);
        elements.chunkName.title = info.chunk;
        elements.position.max = String(info.frames.length - 1);
        elements.frameNumber.min = String(info.start_frame);
        elements.frameNumber.max = String(info.end_frame);
        elements.frameRange.textContent =
          `${info.start_frame.toLocaleString()}–${info.end_frame.toLocaleString()} (${info.frames.length})`;
        elements.videoName.textContent = fileName(info.video);
        elements.videoName.title = info.video;
        elements.videoShape.textContent =
          `${info.video_width.toLocaleString()} × ${info.video_height.toLocaleString()}`;
        elements.legend.replaceChildren(
          ...info.nodes.map((node) => {
            const item = document.createElement("div");
            item.className = "legend-item";
            const swatch = document.createElement("span");
            swatch.className = "swatch";
            swatch.style.background = node.color;
            const text = document.createElement("span");
            text.textContent = node.name;
            item.append(swatch, text);
            return item;
          })
        );
        updateFrameControls();
        await loadFrame();
      } catch (error) {
        setStatus(`Viewer initialization failed: ${error.message}`, true);
      }
    }

    elements.previous.addEventListener("click", () => setPosition(position - 1));
    elements.next.addEventListener("click", () => setPosition(position + 1));
    elements.play.addEventListener("click", togglePlayback);
    elements.random.addEventListener("click", () => {
      stopPlayback();
      setPosition(Math.floor(Math.random() * info.frames.length));
    });
    elements.position.addEventListener("input", () => {
      stopPlayback();
      position = Number(elements.position.value);
      updateFrameControls();
      scheduleLoad();
    });
    elements.frameNumber.addEventListener("change", () => {
      stopPlayback();
      const requested = Number(elements.frameNumber.value);
      let nearest = 0;
      let distance = Infinity;
      info.frames.forEach((frame, index) => {
        const candidate = Math.abs(frame - requested);
        if (candidate < distance) {
          nearest = index;
          distance = candidate;
        }
      });
      setPosition(nearest);
    });
    elements.instanceThreshold.addEventListener("input", () => {
      elements.instanceValue.textContent = Number(elements.instanceThreshold.value).toFixed(2);
      scheduleLoad();
    });
    elements.pointThreshold.addEventListener("input", () => {
      elements.pointValue.textContent = Number(elements.pointThreshold.value).toFixed(2);
      scheduleLoad();
    });
    [elements.showPoints, elements.showEdges, elements.showLabels].forEach((input) => {
      input.addEventListener("change", loadFrame);
    });
    elements.zoomOut.addEventListener("click", () => setZoom(zoom - 0.25));
    elements.zoomFit.addEventListener("click", () => setZoom(1));
    elements.zoomIn.addEventListener("click", () => setZoom(zoom + 0.25));
    window.addEventListener("keydown", (event) => {
      if (["INPUT", "TEXTAREA"].includes(event.target.tagName)) return;
      if (event.key === "ArrowLeft") setPosition(position - 1);
      else if (event.key === "ArrowRight") setPosition(position + 1);
      else if (event.key === " ") {
        event.preventDefault();
        togglePlayback();
      } else if (event.key.toLowerCase() === "r") {
        setPosition(Math.floor(Math.random() * info.frames.length));
      } else if (event.key === "+" || event.key === "=") setZoom(zoom + 0.25);
      else if (event.key === "-") setZoom(zoom - 0.25);
    });
    window.addEventListener("beforeunload", () => {
      if (imageUrl !== null) URL.revokeObjectURL(imageUrl);
    });

    initialize();
  </script>
</body>
</html>
"""


NODE_COLORS = (
    "#f2d14f",
    "#58c77b",
    "#6ea8fe",
    "#4fd5dc",
    "#d57bea",
    "#ff9f5a",
    "#ef6f73",
)


def _bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return blue, green, red


class ChunkViewer:
    """Load one prediction chunk and render annotated video frames."""

    def __init__(
        self,
        chunk_path: Path,
        video_override: Optional[Path] = None,
        display_width: int = 1920,
    ) -> None:
        self.chunk_path = chunk_path.expanduser().resolve()
        if not self.chunk_path.is_file():
            raise FileNotFoundError(f"Chunk not found: {self.chunk_path}")
        if display_width <= 0:
            raise ValueError("--display-width must be positive")

        self.labels = sio.load_slp(self.chunk_path.as_posix())
        if not self.labels.labeled_frames:
            raise ValueError(f"Chunk has no labeled frames: {self.chunk_path}")
        if len(self.labels.videos) != 1:
            raise ValueError(
                f"Expected one video in chunk, found {len(self.labels.videos)}"
            )
        if len(self.labels.skeletons) != 1:
            raise ValueError(
                f"Expected one skeleton in chunk, found {len(self.labels.skeletons)}"
            )

        embedded_video = Path(self.labels.videos[0].filename)
        self.video_path = (
            video_override.expanduser().resolve()
            if video_override is not None
            else embedded_video.expanduser().resolve()
        )
        if not self.video_path.is_file():
            raise FileNotFoundError(
                f"Source video not found: {self.video_path}. "
                "Pass its current path with --video."
            )

        self.frames = {}
        for labeled_frame in self.labels.labeled_frames:
            frame_idx = int(labeled_frame.frame_idx)
            if frame_idx in self.frames:
                raise ValueError(f"Duplicate frame {frame_idx} in {self.chunk_path}")
            self.frames[frame_idx] = labeled_frame
        self.frame_indices = sorted(self.frames)

        self.skeleton = self.labels.skeletons[0]
        self.node_names = [node.name for node in self.skeleton.nodes]
        node_indices = {name: i for i, name in enumerate(self.node_names)}
        self.edges = [
            (node_indices[edge.source.name], node_indices[edge.destination.name])
            for edge in self.skeleton.edges
        ]
        self.node_hex_colors = [
            NODE_COLORS[i % len(NODE_COLORS)] for i in range(len(self.node_names))
        ]
        self.node_bgr_colors = [_bgr(color) for color in self.node_hex_colors]

        self.display_width = display_width
        self._video_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._frame_cache: OrderedDict[
            int, tuple[np.ndarray, float, float]
        ] = OrderedDict()
        self._capture = cv2.VideoCapture(self.video_path.as_posix())
        self._next_decoded_frame = 0
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open source video: {self.video_path}")
        self.video_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if self.frame_indices[-1] >= self.video_frame_count:
            self.close()
            raise ValueError(
                f"Chunk frame {self.frame_indices[-1]} exceeds video length "
                f"({self.video_frame_count} frames)"
            )

    def close(self) -> None:
        with self._video_lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def info(self) -> dict:
        return {
            "chunk": self.chunk_path.as_posix(),
            "video": self.video_path.as_posix(),
            "frames": self.frame_indices,
            "start_frame": self.frame_indices[0],
            "end_frame": self.frame_indices[-1],
            "video_width": self.video_width,
            "video_height": self.video_height,
            "video_frame_count": self.video_frame_count,
            "video_fps": self.video_fps,
            "nodes": [
                {"name": name, "color": color}
                for name, color in zip(self.node_names, self.node_hex_colors)
            ],
            "edges": self.edges,
        }

    def _read_frame(self, frame_idx: int) -> tuple[np.ndarray, float, float]:
        with self._cache_lock:
            cached = self._frame_cache.get(frame_idx)
            if cached is not None:
                self._frame_cache.move_to_end(frame_idx)
                image, scale_x, scale_y = cached
                return image.copy(), scale_x, scale_y

        with self._video_lock:
            if self._capture is None:
                raise RuntimeError("Video is closed")
            if frame_idx != self._next_decoded_frame:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = self._capture.read()
            self._next_decoded_frame = frame_idx + 1 if ok else -1
        if not ok or image is None:
            raise RuntimeError(f"Could not decode video frame {frame_idx}")

        original_height, original_width = image.shape[:2]
        if original_width > self.display_width:
            output_width = self.display_width
            output_height = round(original_height * output_width / original_width)
            image = cv2.resize(
                image, (output_width, output_height), interpolation=cv2.INTER_AREA
            )
        output_height, output_width = image.shape[:2]
        scale_x = output_width / original_width
        scale_y = output_height / original_height

        with self._cache_lock:
            self._frame_cache[frame_idx] = (image.copy(), scale_x, scale_y)
            self._frame_cache.move_to_end(frame_idx)
            while len(self._frame_cache) > 4:
                self._frame_cache.popitem(last=False)
        return image, scale_x, scale_y

    def render_jpeg(
        self,
        frame_idx: int,
        instance_threshold: float,
        point_threshold: float,
        show_points: bool,
        show_edges: bool,
        show_labels: bool,
    ) -> tuple[bytes, int, int]:
        if frame_idx not in self.frames:
            raise KeyError(f"Frame {frame_idx} is not in this chunk")

        image, scale_x, scale_y = self._read_frame(frame_idx)
        labeled_frame = self.frames[frame_idx]
        instances = list(labeled_frame.instances)
        visible_instances = []
        for instance in instances:
            score = float(getattr(instance, "score", math.nan))
            if not math.isnan(score) and score >= instance_threshold:
                visible_instances.append(instance)

        output_width = image.shape[1]
        radius = max(2, round(output_width / 500))
        line_width = max(1, round(output_width / 900))
        font_scale = max(0.3, output_width / 4200)

        for instance in visible_instances:
            coordinates = np.asarray(instance.numpy(), dtype=float)[:, :2]
            if "score" in instance.points.dtype.names:
                point_scores = np.asarray(instance.points["score"], dtype=float)
            else:
                point_scores = np.ones(len(coordinates))
            valid = np.isfinite(coordinates).all(axis=1)
            valid &= np.isfinite(point_scores)
            valid &= point_scores >= point_threshold
            scaled = coordinates.copy()
            scaled[:, 0] *= scale_x
            scaled[:, 1] *= scale_y

            if show_edges:
                for source, destination in self.edges:
                    if not (valid[source] and valid[destination]):
                        continue
                    start = tuple(np.rint(scaled[source]).astype(int))
                    end = tuple(np.rint(scaled[destination]).astype(int))
                    cv2.line(
                        image,
                        start,
                        end,
                        self.node_bgr_colors[destination],
                        line_width,
                        cv2.LINE_AA,
                    )

            if show_points or show_labels:
                for index, point in enumerate(scaled):
                    if not valid[index]:
                        continue
                    center = tuple(np.rint(point).astype(int))
                    color = self.node_bgr_colors[index]
                    if show_points:
                        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
                    if show_labels:
                        label_position = (center[0] + radius + 1, center[1] - radius)
                        cv2.putText(
                            image,
                            self.node_names[index],
                            label_position,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            color,
                            max(1, line_width),
                            cv2.LINE_AA,
                        )

        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        )
        if not ok:
            raise RuntimeError(f"Could not encode rendered frame {frame_idx}")
        return encoded.tobytes(), len(instances), len(visible_instances)


def _query_float(
    query: dict[str, list[str]], name: str, default: float
) -> float:
    try:
        value = float(query.get(name, [str(default)])[0])
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _query_bool(query: dict[str, list[str]], name: str, default: bool) -> bool:
    raw = query.get(name, ["1" if default else "0"])[0].lower()
    if raw not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name} must be 0, 1, false, or true")
    return raw in {"1", "true"}


def make_handler(viewer: ChunkViewer) -> type[BaseHTTPRequestHandler]:
    class ViewerHandler(BaseHTTPRequestHandler):
        def log_message(self, format_string: str, *args) -> None:
            return

        def _send(
            self,
            status: int,
            content_type: str,
            body: bytes,
            extra_headers: Optional[dict[str, str]] = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for name, value in extra_headers.items():
                    self.send_header(name, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._send(status, "application/json; charset=utf-8", body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send(
                        200,
                        "text/html; charset=utf-8",
                        INDEX_HTML.encode("utf-8"),
                    )
                    return
                if parsed.path == "/api/info":
                    self._send_json(200, viewer.info())
                    return
                if parsed.path == "/api/frame":
                    query = parse_qs(parsed.query)
                    if "frame" not in query:
                        raise ValueError("frame is required")
                    frame_idx = int(query["frame"][0])
                    jpeg, total, visible = viewer.render_jpeg(
                        frame_idx=frame_idx,
                        instance_threshold=_query_float(
                            query, "instance_threshold", 0.0
                        ),
                        point_threshold=_query_float(query, "point_threshold", 0.2),
                        show_points=_query_bool(query, "points", True),
                        show_edges=_query_bool(query, "edges", True),
                        show_labels=_query_bool(query, "labels", False),
                    )
                    self._send(
                        200,
                        "image/jpeg",
                        jpeg,
                        {
                            "X-Total-Instances": str(total),
                            "X-Visible-Instances": str(visible),
                        },
                    )
                    return
                self._send_json(404, {"error": "Not found"})
            except (KeyError, ValueError) as error:
                self._send_json(400, {"error": str(error)})
            except Exception as error:
                self._send_json(500, {"error": str(error)})

    return ViewerHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunk", type=Path, help="A *.frames_START_END.slp file")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Override the source-video path stored in the SLP file.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--display-width",
        type=int,
        default=1920,
        help="Downscaled frame width served to the browser (default: 1920).",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the viewer URL in the default browser.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    viewer = ChunkViewer(
        chunk_path=args.chunk,
        video_override=args.video,
        display_width=args.display_width,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(viewer))
    server.daemon_threads = True
    actual_port = int(server.server_address[1])
    url = f"http://{args.host}:{actual_port}"

    print(f"Loaded {viewer.chunk_path.name}")
    print(
        f"Frames {viewer.frame_indices[0]}-{viewer.frame_indices[-1]} "
        f"from {viewer.video_path.name}"
    )
    print(f"Open {url}")
    print("Press Ctrl-C to stop.")
    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
