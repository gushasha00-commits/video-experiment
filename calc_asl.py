import cv2
import numpy as np
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

videos = {
    "极地海洋": r"C:\Users\谷莎莎\Desktop\video_experiment\videos\极地海洋.mp4",
    "欧洲地形": r"C:\Users\谷莎莎\Desktop\video_experiment\videos\欧洲地形.mp4",
    "太阳系": r"C:\Users\谷莎莎\Desktop\video_experiment\videos\太阳系.mp4",
}

# Parameters for scene cut detection
THRESHOLD = 30.0       # histogram correlation threshold (0-100, lower = more sensitive)
MIN_SHOT_LEN = 0.3     # minimum shot length in seconds (avoid false positives from flashes)
SAMPLING_RATE = 1      # compare every Nth frame (1 = every frame, 2 = every other, etc.)

print("=" * 70)
print(f"{'视频':<12} {'总时长(s)':<10} {'镜头数':<8} {'ASL(s)':<10} {'评判'}")
print("=" * 70)

for name, path in videos.items():
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"{name:<12} 无法打开视频")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    shot_count = 1  # start with the first shot
    prev_hist = None
    last_cut_time = 0.0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLING_RATE == 0:
            # Compute HSV histogram (H=50 bins, S=60 bins)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                current_time = frame_idx / fps

                # correlation < THRESHOLD means a big visual change → likely a cut
                if correlation * 100 < THRESHOLD and (current_time - last_cut_time) > MIN_SHOT_LEN:
                    shot_count += 1
                    last_cut_time = current_time

            prev_hist = hist

        frame_idx += 1

    cap.release()

    asl = duration / shot_count

    # Rough rhythm assessment for educational/science videos
    if asl < 2:
        judgment = "[FAST] 偏快，节奏较紧张"
    elif asl < 4:
        judgment = "[OK] 适中偏快"
    elif asl < 7:
        judgment = "[GOOD] 适中，适合科普类视频"
    elif asl < 10:
        judgment = "[OK] 适中偏慢"
    else:
        judgment = "[SLOW] 偏慢，可能略显拖沓"

    print(f"{name:<12} {duration:<10.1f} {shot_count:<8} {asl:<10.2f} {judgment}")

print("=" * 70)
print("参考: 科普/教育类视频 ASL 一般在 4-8 秒较为合适")
print("      电影平均 ASL 约 3-5 秒，TikTok 类快节奏约 1-2 秒")
