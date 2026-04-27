from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.font_manager import FontProperties
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

BG = "#000000"
NEON_GREEN = "#39ff14"
DIM_GREEN = "#1f7a0e"
FPS = 60
WIDTH = 1920
HEIGHT = 1080
DPI = 100
FIGSIZE = (WIDTH / DPI, HEIGHT / DPI)
SAMPLE_RATE = 44_100
AAC_BITRATE = "192k"
LINEWIDTH = 3.0
DEFAULT_DURATION = 5.0
TITLE_Y = 0.935


def ease(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 0.5 * (1.0 - math.cos(math.pi * u))


def reveal_curve(t: float, t0: float, dur: float) -> float:
    if dur <= 0:
        return 1.0 if t >= t0 else 0.0
    return ease((t - t0) / dur)


def pop(u: float) -> float:
    return 1.0 - math.exp(-4.0 * ease(u))


def _font_path() -> Path | None:
    repo_font = Path("VCR_OSD_MONO_1.001.ttf")
    if repo_font.exists():
        return repo_font
    module_font = Path(__file__).resolve().with_name("VCR_OSD_MONO_1.001.ttf")
    if module_font.exists():
        return module_font
    return None


def load_font(size: float) -> FontProperties:
    font_path = _font_path()
    if font_path is not None:
        return FontProperties(fname=str(font_path), size=size)
    return FontProperties(family="DejaVu Sans Mono", size=size)


def _upper(text: str) -> str:
    return str(text).upper().replace("_", " ")


def _new_figure(rect: tuple[float, float, float, float] = (0.10, 0.16, 0.80, 0.68)):
    fig = Figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG)
    FigureCanvasAgg(fig)
    ax = fig.add_axes(rect, facecolor=BG)
    return fig, ax


def _apply_tick_font(ax, size: float) -> None:
    tick_font = load_font(size)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontproperties(tick_font)


def _style_axes(ax, xlabel: str, ylabel: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NEON_GREEN)
        ax.spines[side].set_linewidth(1.3)

    ax.tick_params(axis="both", colors=NEON_GREEN, labelsize=12, width=1.1, length=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", color=DIM_GREEN, linestyle="-", linewidth=0.8, alpha=0.26)
    ax.set_xlabel(_upper(xlabel), color=NEON_GREEN)
    ax.set_ylabel(_upper(ylabel), color=NEON_GREEN)
    ax.xaxis.label.set_fontproperties(load_font(15))
    ax.yaxis.label.set_fontproperties(load_font(15))
    _apply_tick_font(ax, 11)


def _add_title(fig, title: str, *, alpha: float = 1.0):
    title_artist = fig.text(
        0.10,
        TITLE_Y,
        _upper(title),
        color=NEON_GREEN,
        fontproperties=load_font(20),
        alpha=alpha,
        ha="left",
        va="center",
    )
    return title_artist


def _add_series_labels(fig, labels: list[tuple[str, str]], *, alpha: float = 1.0) -> list:
    artists = []
    x = 0.75
    y = TITLE_Y
    for index, (label, color) in enumerate(labels):
        artists.append(
            fig.text(
                x,
                y - (index * 0.038),
                _upper(label),
                color=color,
                fontproperties=load_font(12),
                alpha=alpha,
                ha="left",
                va="center",
            )
        )
    return artists


def _value_formatter(decimals: int = 3):
    def _format(value, _pos):
        if abs(value) >= 100:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:,.1f}"
        return f"{value:.{decimals}f}"

    return FuncFormatter(_format)


def _padded_limits(values: np.ndarray, *, clamp_min: float | None = None, clamp_max: float | None = None):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        low, high = 0.0, 1.0
    else:
        low = float(finite.min())
        high = float(finite.max())
        if math.isclose(low, high):
            pad = abs(low) * 0.1 or 0.1
            low -= pad
            high += pad
        else:
            pad = (high - low) * 0.08
            low -= pad
            high += pad
    if clamp_min is not None:
        low = max(low, clamp_min)
    if clamp_max is not None:
        high = min(high, clamp_max)
    return low, high


def _ensure_parent(path: os.PathLike[str] | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig, output_path: os.PathLike[str] | str) -> None:
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=DPI, facecolor=BG)
    plt.close(fig)


def _save_animation(anim: FuncAnimation, output_path: os.PathLike[str] | str, *, fps: int, duration: float) -> None:
    output_path = _ensure_parent(output_path)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg is required to render forecast videos.")

    staging_dir = output_path.parent / f".render_{uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw_video = staging_dir / "video_no_audio.mp4"
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=9000,
            extra_args=["-pix_fmt", "yuv420p"],
        )
        anim.save(str(raw_video), writer=writer)
        final_video = staging_dir / "video_with_audio.mp4"
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(raw_video),
                "-f",
                "lavfi",
                "-t",
                f"{duration:.3f}",
                "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                AAC_BITRATE,
                "-shortest",
                "-movflags",
                "+faststart",
                str(final_video),
            ],
            check=True,
            capture_output=True,
        )
        shutil.copy2(final_video, output_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    plt.close(anim._fig)


def render_training_metric_visuals(
    train_values,
    val_values,
    output_png: os.PathLike[str] | str,
    output_mp4: os.PathLike[str] | str,
    *,
    title: str,
    y_label: str,
    train_label: str = "Train",
    val_label: str = "Val",
    duration: float = DEFAULT_DURATION,
    y_max: float | None = None,
) -> None:
    train = np.asarray(train_values, dtype=float)
    val = np.asarray(val_values, dtype=float)
    epochs = np.arange(1, len(train) + 1)
    ylim = _padded_limits(np.concatenate([train, val]), clamp_min=0.0, clamp_max=y_max)

    fig, ax = _new_figure()
    _style_axes(ax, "Epoch", y_label)
    ax.set_xlim(1, max(len(epochs), 2))
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(_value_formatter())
    ax.plot(epochs, train, color=NEON_GREEN, linewidth=LINEWIDTH, solid_capstyle="round")
    ax.plot(epochs, val, color=DIM_GREEN, linewidth=LINEWIDTH, solid_capstyle="round")
    _add_title(fig, title)
    _add_series_labels(fig, [(train_label, NEON_GREEN), (val_label, DIM_GREEN)])
    _save_figure(fig, output_png)

    fps = FPS
    n_frames = max(1, int(round(duration * fps)))
    fig, ax = _new_figure()
    _style_axes(ax, "Epoch", y_label)
    ax.set_xlim(1, max(len(epochs), 2))
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(_value_formatter())
    title_artist = _add_title(fig, title, alpha=0.0)
    label_artists = _add_series_labels(fig, [(train_label, NEON_GREEN), (val_label, DIM_GREEN)], alpha=0.0)
    train_line, = ax.plot([], [], color=NEON_GREEN, linewidth=LINEWIDTH, solid_capstyle="round")
    val_line, = ax.plot([], [], color=DIM_GREEN, linewidth=LINEWIDTH, solid_capstyle="round")

    def _init():
        train_line.set_data([], [])
        val_line.set_data([], [])
        return (train_line, val_line, title_artist, *label_artists)

    def _update(frame: int):
        t = frame / max(n_frames - 1, 1) * duration
        progress = pop((frame + 1) / n_frames)
        end = max(1, int(math.ceil(progress * len(epochs))))
        train_line.set_data(epochs[:end], train[:end])
        val_line.set_data(epochs[:end], val[:end])
        alpha = reveal_curve(t, 0.0, min(0.9, duration * 0.25))
        title_artist.set_alpha(alpha)
        for artist in label_artists:
            artist.set_alpha(alpha)
        return (train_line, val_line, title_artist, *label_artists)

    anim = FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=n_frames,
        blit=True,
        interval=1000 / fps,
    )
    _save_animation(anim, output_mp4, fps=fps, duration=duration)


def render_feature_importance_visuals(
    importance_df,
    output_png: os.PathLike[str] | str,
    output_mp4: os.PathLike[str] | str,
    *,
    title: str = "Feature Importance",
    max_features: int = 12,
    duration: float = DEFAULT_DURATION,
) -> None:
    plot_df = importance_df.sort_values("Importance", ascending=False).head(max_features).copy()
    features = [_upper(name) for name in plot_df["Feature"].tolist()]
    values = np.asarray(plot_df["Importance"].tolist(), dtype=float)
    y_positions = np.arange(len(features))
    x_low, x_high = _padded_limits(values)
    x_low = min(x_low, 0.0)
    x_high = max(x_high, 0.0)
    if math.isclose(x_low, x_high):
        x_high = x_low + 1.0
    label_size = float(max(8, 15 - 0.35 * max(len(features) - 8, 0)))

    fig, ax = _new_figure(rect=(0.18, 0.14, 0.72, 0.70))
    _style_axes(ax, "Permutation MSE Delta", "Feature")
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(-0.5, len(features) - 0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features, color=NEON_GREEN)
    ax.axvline(0.0, color=NEON_GREEN, linewidth=1.0, alpha=0.3)
    bars = ax.barh(
        y_positions,
        values,
        color=NEON_GREEN,
        edgecolor=NEON_GREEN,
        linewidth=0.8,
        alpha=0.9,
    )
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(_value_formatter())
    _apply_tick_font(ax, label_size)
    _add_title(fig, f"{title} - TOP {len(features)}")
    for bar, value in zip(bars, values):
        x = value + ((x_high - x_low) * 0.014 if value >= 0 else -(x_high - x_low) * 0.014)
        align = "left" if value >= 0 else "right"
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.4f}",
            color=NEON_GREEN,
            fontproperties=load_font(10),
            va="center",
            ha=align,
        )
    _save_figure(fig, output_png)

    fps = FPS
    n_frames = max(1, int(round(duration * fps)))
    fig, ax = _new_figure(rect=(0.18, 0.14, 0.72, 0.70))
    _style_axes(ax, "Permutation MSE Delta", "Feature")
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(-0.5, len(features) - 0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features, color=NEON_GREEN)
    ax.axvline(0.0, color=NEON_GREEN, linewidth=1.0, alpha=0.3)
    bars = ax.barh(
        y_positions,
        np.zeros_like(values),
        color=NEON_GREEN,
        edgecolor=NEON_GREEN,
        linewidth=0.8,
        alpha=0.9,
    )
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(_value_formatter())
    _apply_tick_font(ax, label_size)
    title_artist = _add_title(fig, f"{title} - TOP {len(features)}", alpha=0.0)
    value_artists = []
    span = x_high - x_low
    for bar, value in zip(bars, values):
        artist = ax.text(
            0.0,
            bar.get_y() + bar.get_height() / 2.0,
            "",
            color=NEON_GREEN,
            fontproperties=load_font(10),
            va="center",
            ha="left",
            alpha=0.0,
        )
        value_artists.append((artist, value))

    def _update(frame: int):
        t = frame / max(n_frames - 1, 1) * duration
        progress = pop((frame + 1) / n_frames)
        alpha = reveal_curve(t, 0.0, min(0.9, duration * 0.22))
        title_artist.set_alpha(alpha)
        for bar, value, (artist, artist_value) in zip(bars, values, value_artists):
            current = value * progress
            bar.set_width(current)
            x = current + (span * 0.014 if current >= 0 else -(span * 0.014))
            artist.set_text(f"{artist_value:.4f}")
            artist.set_x(x)
            artist.set_alpha(alpha)
            artist.set_ha("left" if current >= 0 else "right")
        return (title_artist, *bars, *(artist for artist, _ in value_artists))

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        blit=False,
        interval=1000 / fps,
    )
    _save_animation(anim, output_mp4, fps=fps, duration=duration)
