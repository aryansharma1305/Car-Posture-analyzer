import cv2
import csv
import time
from pathlib import Path
from pose_core import PoseExtractor
from posture_angles import compute_angles
from smoothing import AngleSmoother
from posture_rules import classify_posture, PostureConfig

GREEN=(0,255,0); RED=(0,0,255); YEL=(0,255,255); WHT=(255,255,255)

def draw_text(img, text, x, y, color=WHT, scale=0.7, thick=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)  # change to 1/2 if you have multiple cameras
    extractor = PoseExtractor(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    smoother = AngleSmoother(alpha=0.25)
    cfg = PostureConfig()

    # timers
    label_durations = {}
    current_label = None
    last_tick = time.monotonic()

    # logging
    out_dir = Path("posture_logs"); out_dir.mkdir(exist_ok=True)
    session_start_ts = time.strftime("%Y%m%d_%H%M%S")
    angles_log_path = out_dir / f"angles_{session_start_ts}.csv"
    summary_path     = out_dir / f"summary_{session_start_ts}.csv"

    with open(angles_log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_sec","label","neck_from_vertical","trunk_from_vertical",
                    "left_hip_angle","right_hip_angle","left_knee_angle","right_knee_angle",
                    "left_shoulder_elev","right_shoulder_elev"])

    t0 = time.monotonic()
    fps_avg = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        t_frame = time.monotonic()

        results, pts, _ = extractor.process_bgr(frame)
        if results is not None:
            extractor.draw(frame, results)
            try:
                raw_angles = compute_angles(pts)
                angles = smoother(raw_angles)

                label, tags = classify_posture(angles, cfg)

                now = time.monotonic()
                dt = now - last_tick
                last_tick = now

                if current_label is None:
                    current_label = label
                if label != current_label:
                    label_durations[current_label] = label_durations.get(current_label, 0.0) + dt
                    current_label = label
                label_durations[current_label] = label_durations.get(current_label, 0.0) + dt

                # log roughly once per second
                if int(now - t0) != int(now - t0 - dt):
                    with open(angles_log_path, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow([round(now - t0,2), label] + [round(angles[k],2) for k in [
                            "neck_from_vertical","trunk_from_vertical",
                            "left_hip_angle","right_hip_angle",
                            "left_knee_angle","right_knee_angle",
                            "left_shoulder_elev","right_shoulder_elev"]])

                # UI
                color = GREEN if label == "Neutral" else (YEL if "mild" in tags else RED)
                draw_text(frame, f"Posture: {label}" + (" (mild)" if "mild" in tags else ""), 10, 30, color, 0.9, 2)

                y = 60
                for k in ["neck_from_vertical","trunk_from_vertical","left_hip_angle","right_hip_angle",
                          "left_knee_angle","right_knee_angle","left_shoulder_elev","right_shoulder_elev"]:
                    draw_text(frame, f"{k}: {angles[k]:.1f}°", 10, y); y += 22

                y += 8
                total = sum(label_durations.values()) + 1e-9
                for k in sorted(label_durations.keys()):
                    sec = label_durations[k]
                    draw_text(frame, f"{k}: {sec:6.1f}s  ({(sec/total*100):.1f}%)", 10, y); y += 22

            except Exception as e:
                draw_text(frame, f"Angle/cls error: {e}", 10, 30, RED)
        else:
            draw_text(frame, "No person detected", 10, 30, RED)

        fps = 1.0 / max(1e-6, (time.monotonic() - t_frame))
        fps_avg = fps if fps_avg is None else (0.9*fps_avg + 0.1*fps)
        draw_text(frame, f"FPS: {fps_avg:.1f}", frame.shape[1]-150, 30)

        cv2.imshow("Posture - Labels & Timers", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()

    total = sum(label_durations.values())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label","seconds","percent"])
        for k, sec in label_durations.items():
            pct = (sec/total*100) if total>0 else 0.0
            w.writerow([k, round(sec,1), round(pct,1)])

    print(f"\nSaved angle stream: {angles_log_path}")
    print(f"Saved session summary: {summary_path}")

if __name__ == "__main__":
    main()
