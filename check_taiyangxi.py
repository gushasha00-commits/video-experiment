import cv2
import numpy as np
import sys

video_path = r"C:\Users\谷莎莎\Desktop\video_experiment\videos\太阳系.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"太阳系: FPS={fps:.2f}, total_frames={total_frames}, duration={duration:.1f}s")
print()
print("尝试不同阈值:")
print("-" * 60)
print(f"{'阈值':<8} {'镜头数':<8} {'ASL(s)':<10} {'评判'}")
print("-" * 60)

for threshold in [30, 25, 20, 15, 10, 8, 6]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    shot_count = 1
    prev_hist = None
    last_cut_time = 0.0
    frame_idx = 0
    MIN_SHOT_LEN = 0.3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 1 == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                current_time = frame_idx / fps
                if correlation * 100 < threshold and (current_time - last_cut_time) > MIN_SHOT_LEN:
                    shot_count += 1
                    last_cut_time = current_time

            prev_hist = hist

        frame_idx += 1

    asl = duration / shot_count
    if asl < 4:
        judgment = "[OK] 适中偏快"
    elif asl < 7:
        judgment = "[GOOD] 适中"
    elif asl < 10:
        judgment = "[OK] 适中偏慢"
    else:
        judgment = "[SLOW] 偏慢"

    print(f"{threshold:<8} {shot_count:<8} {asl:<10.2f} {judgment}")

cap.release()
