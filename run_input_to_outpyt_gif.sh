#!/usr/bin/env bash
set -euo pipefail

# Full pipeline:
# 1) input.mp4 -> frames
# 2) frames -> depth_npy + depth_vis
# 3) depth_vis -> output.(gif|mp4)

INPUT_VIDEO="${1:-input.mp4}"
FPS="${2:-10}"
OUTPUT_PATH="${3:-output.gif}"
OUTPUT_MODE="${4:-gif}"

FRAME_DIR="frames"
OUTPUT_DIR="outputs/depth_from_video"
VIS_DIR="${OUTPUT_DIR}/depth_vis"
PALETTE_PATH="${OUTPUT_DIR}/palette.png"
TMP_GIF_PATH="${OUTPUT_DIR}/tmp_compressed.gif"

MODEL_PATH="model/unidepth-v2-vitl14"
DEVICE="${DEVICE:-cuda}"
VIS_MAX="${VIS_MAX:-20.0}"
VIS_MODE="${VIS_MODE:-normalize}"
VIS_MIN_PERCENTILE="${VIS_MIN_PERCENTILE:-2.0}"
VIS_MAX_PERCENTILE="${VIS_MAX_PERCENTILE:-98.0}"

# GIF compression knobs:
# - GIF_FPS: output gif fps, default keeps input FPS
# - GIF_SCALE_WIDTH / GIF_SCALE_HEIGHT: output size; keep aspect ratio with -1
# - GIF_MAX_COLORS: palette size (2-256), fewer colors => smaller file
# - GIF_DITHER: e.g. bayer, sierra2_4a, none
# - GIF_LOSSY: optional gifsicle lossy compression level (0 disables)
GIF_FPS="${GIF_FPS:-${FPS}}"
GIF_SCALE_WIDTH="${GIF_SCALE_WIDTH:-640}"
GIF_SCALE_HEIGHT="${GIF_SCALE_HEIGHT:--1}"
GIF_MAX_COLORS="${GIF_MAX_COLORS:-96}"
GIF_DITHER="${GIF_DITHER:-bayer}"
GIF_LOSSY="${GIF_LOSSY:-80}"

# Video export knobs (used when OUTPUT_MODE=video)
VIDEO_CRF="${VIDEO_CRF:-23}"
VIDEO_PRESET="${VIDEO_PRESET:-medium}"

if [[ ! -f "${INPUT_VIDEO}" ]]; then
  echo "Input video not found: ${INPUT_VIDEO}"
  exit 1
fi

if [[ "${OUTPUT_MODE}" != "gif" && "${OUTPUT_MODE}" != "video" ]]; then
  echo "Invalid OUTPUT_MODE: ${OUTPUT_MODE}"
  echo "Expected: gif | video"
  exit 1
fi

mkdir -p "${FRAME_DIR}" "${OUTPUT_DIR}"

echo "[1/4] Extracting frames from ${INPUT_VIDEO} ..."
ffmpeg -y -i "${INPUT_VIDEO}" -vf "fps=${FPS}" -q:v 2 -start_number 0 "${FRAME_DIR}/%06d.jpg"

echo "[2/4] Running UniDepthV2 inference ..."
python scripts/infer_depth_from_frames.py \
  --input_dir "${FRAME_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --device "${DEVICE}" \
  --vis_max "${VIS_MAX}" \
  --vis_mode "${VIS_MODE}" \
  --vis_min_percentile "${VIS_MIN_PERCENTILE}" \
  --vis_max_percentile "${VIS_MAX_PERCENTILE}" \
  --fx 505.038 \
  --fy 504.937 \
  --cx 1080.938 \
  --cy 1080.776

if [[ ! -d "${VIS_DIR}" ]]; then
  echo "Depth visualization directory not found: ${VIS_DIR}"
  exit 1
fi

if [[ "${OUTPUT_MODE}" == "gif" ]]; then
  echo "[3/4] Building GIF palette ..."
  echo "      - fps=${GIF_FPS}, scale=${GIF_SCALE_WIDTH}:${GIF_SCALE_HEIGHT}, max_colors=${GIF_MAX_COLORS}"
  ffmpeg -y \
    -framerate "${FPS}" \
    -i "${VIS_DIR}/%06d.png" \
    -vf "fps=${GIF_FPS},scale=${GIF_SCALE_WIDTH}:${GIF_SCALE_HEIGHT}:flags=lanczos,palettegen=max_colors=${GIF_MAX_COLORS}:stats_mode=diff" \
    "${PALETTE_PATH}"

  echo "[4/4] Encoding GIF: ${OUTPUT_PATH} ..."
  ffmpeg -y \
    -framerate "${FPS}" \
    -i "${VIS_DIR}/%06d.png" \
    -i "${PALETTE_PATH}" \
    -lavfi "fps=${GIF_FPS},scale=${GIF_SCALE_WIDTH}:${GIF_SCALE_HEIGHT}:flags=lanczos[x];[x][1:v]paletteuse=dither=${GIF_DITHER}:diff_mode=rectangle" \
    "${TMP_GIF_PATH}"

  # Optional second-pass compression via gifsicle if available.
  if command -v gifsicle >/dev/null 2>&1 && [[ "${GIF_LOSSY}" != "0" ]]; then
    echo "      - running gifsicle lossy=${GIF_LOSSY}"
    gifsicle -O3 --lossy="${GIF_LOSSY}" "${TMP_GIF_PATH}" -o "${OUTPUT_PATH}"
    rm -f "${TMP_GIF_PATH}"
  else
    mv -f "${TMP_GIF_PATH}" "${OUTPUT_PATH}"
  fi
else
  echo "[3/4] Encoding MP4 video: ${OUTPUT_PATH} ..."
  ffmpeg -y \
    -framerate "${FPS}" \
    -i "${VIS_DIR}/%06d.png" \
    -c:v libx264 \
    -preset "${VIDEO_PRESET}" \
    -crf "${VIDEO_CRF}" \
    -pix_fmt yuv420p \
    "${OUTPUT_PATH}"

  echo "[4/4] Skipped GIF steps (OUTPUT_MODE=video)."
fi

echo "Pipeline done."
echo "Output mode: ${OUTPUT_MODE}"
echo "Output saved to: ${OUTPUT_PATH}"
echo "Depth npy dir: ${OUTPUT_DIR}/depth_npy"
echo "Depth vis dir: ${VIS_DIR}"
