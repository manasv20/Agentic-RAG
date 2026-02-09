#!/usr/bin/env bash
# Split and combine screen recordings for LinkedIn video.
# Uses the 3 most recent .mov files on Desktop (oldest = Video 1, newest = Video 3).
# Segments (in order):
#   Video 1: 0:00 - 0:25 (25 sec)
#   Video 2: 0:00 - 0:57 (57 sec)
#   Video 2: 4:22 - 4:52 (30 sec)
#   Video 3: 0:00 - 0:38 (38 sec)
# Total: 2 min 30 sec

set -e
DESKTOP="${1:-$HOME/Desktop}"
WORK="$DESKTOP/VideoMergeTemp"
OUT="$DESKTOP/LinkedIn_Agentic_RAG_Combined.mp4"

# Find 3 most recent .mov files on Desktop, ordered oldest first (so V1=first recorded, V3=last)
count=0
while read -r _ filepath; do
  if [[ $count -eq 0 ]]; then V1="$filepath"; fi
  if [[ $count -eq 1 ]]; then V2="$filepath"; fi
  if [[ $count -eq 2 ]]; then V3="$filepath"; fi
  ((count++)) || true
done < <(find "$DESKTOP" -maxdepth 1 -name "*.mov" -type f -exec sh -c 'stat -f "%m $1" "$1"' _ {} \; 2>/dev/null | sort -n | tail -3)

if [[ $count -lt 3 ]]; then
  echo "Error: Need 3 .mov files on Desktop. Found: $count"
  echo "Desktop path used: $DESKTOP"
  exit 1
fi

echo "Using:"
echo "  V1 (oldest): $V1"
echo "  V2:          $V2"
echo "  V3 (newest): $V3"
echo ""

mkdir -p "$WORK"
cd "$WORK"

echo "Extracting segments..."
# Video 1: 0-25s
ffmpeg -y -i "$V1" -ss 0 -t 25 -c copy segment1.mp4

# Video 2: 0-57s
ffmpeg -y -i "$V2" -ss 0 -t 57 -c copy segment2.mp4

# Video 2: 4:22-4:52 (30 sec)
ffmpeg -y -i "$V2" -ss 262 -t 30 -c copy segment3.mp4

# Video 3: 0-38s
ffmpeg -y -i "$V3" -ss 0 -t 38 -c copy segment4.mp4

echo "Creating concat list..."
cat > concat.txt << 'EOF'
file 'segment1.mp4'
file 'segment2.mp4'
file 'segment3.mp4'
file 'segment4.mp4'
EOF

echo "Combining into final video..."
ffmpeg -y -f concat -safe 0 -i concat.txt -c copy "$OUT"

echo "Done. Output: $OUT"
echo "You can remove temp folder: $WORK"
