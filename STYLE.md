# Channel B-Roll Style Guide

Canonical reference for the Fourier Transforms video and future explainer B-roll.
Anchor clip: `output/b1_opening.mp4` (from `broll_b1_opening.py`).

All B-roll generators in this folder MUST import constants/helpers from `broll_style.py`
so the visual language stays identical across clips.

## 1. Identity

| Element | Value |
|---|---|
| Background | **Pure black** `#000000` |
| Waveform / accents | **Neon green** `#39ff14` |
| Secondary accent (optional) | Dim neon green `#1f7a0e` for gridlines and dim labels |
| Font | `VCR_OSD_MONO_1.001.ttf` (repo root) |
| Typography style | **All caps** for titles / labels, 16–26 pt. Never mixed case. |

## 2. Canvas & frame rate

- Resolution: **1920x1080** (16:9). Matplotlib `figsize=(19.2, 10.8), dpi=100`.
- Frame rate: **60 fps**.
- Never use `bbox_inches="tight"` (it rescales pixels). Always save full figure.

## 3. Layout patterns

- **Single-panel time-domain demo** (simple tone, complex sum):  
  Plot centered, `axes rect ~= [0.10, 0.16, 0.80, 0.68]`, axis spines neon green, title band above.
- **Stacked three-panel additive demo** (Wave A / Wave B / Sum):  
  Three equal panels top-to-bottom, shared x-axis style, label each panel top-left in VCR font.
- **Time-to-FT split** (the B-1 pattern):  
  Left ~`[0.034, 0.10, 0.44, 0.78]` = time-domain trace,  
  Right ~`[0.52, 0.10, 0.44, 0.78]` = FFT magnitude.  
  FT panel eases in with cosine opacity ramp (see `broll_style.reveal_curve`).

## 4. Lines, bars, axes

- Waveform stroke: neon green, **3 px**, `solid_capstyle="round"`.
- Spectrum bars: neon green fill + edge, `linewidth=0.15`, `align="center"`, width ≈ 0.92 × bin spacing.
- Show only the **bottom + left** spines. Hide top/right.
- Axis tick/label color: neon green. Axis labels: `FREQUENCY f (HZ)`, `SIGNAL |X(f)|`, `TIME (WINDOW)`, `AMPLITUDE`.
- Zero-baseline line (optional): neon green, 1 px, alpha ≈ 0.3.

## 5. Motion language

- **Easing:** prefer cosine ease-in-out `e(u) = 0.5 * (1 - cos(pi * u))`. No linear fades, no sharp cuts inside a clip.
- **Opacity “pop”:** `pop(u) = 1 - exp(-4 * u_eased)`; used for FT reveal.
- **Scrolling trace:** fixed-width time window (≈0.35–0.40 s) ending at `t_wall`. Newest sample on the right.
- **Sliding FFT:** Hann-windowed rFFT of the last ~0.5 s of the same signal each frame, so the spectrum matches what is on screen / in the audio.

## 6. Audio rules

- **Stereo 44.1 kHz, 16-bit PCM.**
- Mux as **AAC 192 kbps** via FFmpeg alongside H.264 `yuv420p` video.
- Apply a **cosine fade-in/out (~20 ms)** to every tone to kill clicks at boundaries.
- Peak-normalize each clip to **−0.5 dBFS** (≈ 0.94) before muxing.
- When in doubt, the audio should be the **same signal as the plot**, rendered through an identical math path.
- When a clip has no diegetic sound it should still render a silent stereo track of the same length — never leave video audio track empty (avoids AVC+no-audio muxing quirks in NLE).

## 7. Titles / labels

- One short all-caps title near the top (≤ 40 chars), VCR font, 18–22 pt, alpha driven by a reveal curve.
- Frequency labels: `"440 HZ"` (space, no decimals unless needed).
- Never use the em-dash `—` in titles (not in VCR OSD Mono). Use `"-"`.

## 8. File / naming conventions

- Scripts: `01_hook_transform.py`, `02_wave_primitives.py`, `03_fft_reveals.py`, …
- Each script owns one section of the script outline and exposes a CLI:
  `python <script>.py --clip bXX` (single) or `--clip all`.
- Output path: `output/bXX_short_name.mp4` (1920x1080, 60 fps, H.264 + AAC).
- Temporary PNG sequences live in a per-run `TemporaryDirectory`; never commit frames.

## 9. Shared code

All scripts import from `broll_style.py`:

- `NEON_GREEN`, `BG`, `WAVEFORM_COLOR`, `LINEWIDTH_WAVE`, `FPS`, `SAMPLE_RATE`
- `load_font()`, `new_figure()`, `style_time_axes()`, `style_spectrum_axes()`
- `fft_axis_and_mask(f_max, n=4096, fs=8192)`, `sliding_fft_mag(t_end, signal_fn, mask, n, fs)`
- `write_stereo_wav(path, L, R, sr)`, `silent_stereo(duration, sr)`, `tone(f, t, amp=1.0, phase=0.0)`, `fade_edges(x, sr, edge_s=0.02)`
- `render_mp4(frames_dir, wav_path, out_mp4, fps)` (FFmpeg wrapper)
- `reveal_curve(t, t0, dur)`, `ease(u)`, `pop(u)`

## 10. Checklist before committing a new B-roll clip

- [ ] 1920x1080, 60 fps, H.264 + AAC
- [ ] Neon green on pure black, VCR font for every text glyph
- [ ] Titles/labels all-caps, no em-dashes
- [ ] All audio faded at boundaries, peak ≈ 0.94
- [ ] Any FFT shown is a real windowed FFT of the displayed signal
- [ ] Output path matches `output/bXX_<slug>.mp4`
- [ ] `--clip` CLI flag works for this clip and for `all`
