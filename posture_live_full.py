import cv2, time, csv, sys
from pathlib import Path
from pose_core import PoseExtractor
from posture_angles import compute_angles
from smoothing import AngleSmoother
from posture_fullbody_rules import detect_body_state, classify_upper_body, BodyConfig
from ergonomics_scores import compute_rula, compute_reba, feedback_lines, risk_buckets
from driver_comfort_analyzer import DriverComfortAnalyzer, score_driver_comfort
import json

WHT=(255,255,255); RED=(0,0,255); GRN=(0,255,0); YEL=(0,255,255); BLK=(0,0,0); CYA=(255,255,0)

def draw_text(img, txt, x, y, col=WHT, scale=0.8, thick=2):
    cv2.putText(img, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

def draw_header(img, rula, reba, fb_lines, duration_s, good_s, high_s, cam_idx, comfort_score):
    h, w = img.shape[:2]
    header_h = 120  # Increased height for comfort info
    cv2.rectangle(img, (0,0), (w, header_h), BLK, thickness=-1)
    
    # Title
    draw_text(img, "Mahindra Driver Comfort Analyzer", 10, 25, WHT, 0.8, 2)
    
    # Left column - Ergonomic Scores
    left_x = 10
    draw_text(img, "Ergonomic Scores", left_x, 45, CYA, 0.6, 2)
    draw_text(img, f"RULA: {rula[1]} ({rula[0]})", left_x, 65, (0,255,0) if rula[0]==1 else (0,255,255) if rula[0]==2 else RED, 0.6, 2)
    draw_text(img, f"REBA: {reba[1]} ({reba[0]})", left_x, 85, (0,255,0) if reba[0]==1 else (0,255,255) if reba[0]==2 else RED, 0.6, 2)
    
    # Middle column - Comfort Score
    mid_x = w//3 + 10
    draw_text(img, "Driver Comfort", mid_x, 45, CYA, 0.6, 2)
    comfort_color = (0,255,0) if comfort_score.overall_comfort >= 85 else (0,255,255) if comfort_score.overall_comfort >= 70 else (0,165,255) if comfort_score.overall_comfort >= 50 else RED
    draw_text(img, f"Score: {comfort_score.overall_comfort:.1f}", mid_x, 65, comfort_color, 0.6, 2)
    draw_text(img, f"Category: {comfort_score.comfort_category}", mid_x, 85, comfort_color, 0.6, 2)
    draw_text(img, f"Risk: {comfort_score.ergonomic_risk:.1f}", mid_x, 105, RED if comfort_score.ergonomic_risk > 50 else YEL, 0.6, 2)
    
    # Right column - Session Info
    right_x = 2*w//3 + 10
    draw_text(img, "Session Info", right_x, 45, CYA, 0.6, 2)
    mm, ss = divmod(int(duration_s), 60)
    hh, mm = divmod(mm, 60)
    dur_txt = f"Duration: {hh:02d}:{mm:02d}:{ss:02d}"
    draw_text(img, dur_txt, right_x, 65, WHT, 0.6, 2)
    draw_text(img, f"Good Time: {int(good_s)}s", right_x, 85, GRN, 0.6, 2)
    draw_text(img, f"High Risk: {int(high_s)}s", right_x, 105, RED, 0.6, 2)
    
    # Camera info
    draw_text(img, f"Cam: {cam_idx}  (press 'c' to switch)", w-310, 25, CYA, 0.55, 2)

def draw_comfort_recommendations(img, comfort_score, y_start=130):
    """Draw comfort recommendations below the header."""
    if not comfort_score.recommendations:
        return y_start
    
    draw_text(img, "Comfort Recommendations:", 10, y_start, CYA, 0.7, 2)
    y = y_start + 25
    
    for i, rec in enumerate(comfort_score.recommendations[:3]):  # Show top 3
        color = GRN if "good" in rec.lower() else YEL if "adjust" in rec.lower() else RED
        draw_text(img, f"• {rec}", 15, y, color, 0.55, 2)
        y += 20
    
    return y

def get_vis(results):
    lm = results.pose_landmarks.landmark
    idx = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
    }
    return {k: lm[i].visibility for k, i in idx.items()}

class CameraManager:
    """Advanced camera management with automatic detection and fallback"""
    
    def __init__(self, max_cameras=20):
        self.max_cameras = max_cameras
        self.available_cameras = []
        self.current_camera = None
        self.cap = None
        self.detect_cameras()
    
    def detect_cameras(self):
        """Automatically detect all available cameras"""
        print("🔍 Detecting available cameras...")
        self.available_cameras = []
        
        for i in range(self.max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Test if we can actually read frames
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    camera_info = {
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'resolution': f"{width}x{height}"
                    }
                    self.available_cameras.append(camera_info)
                    print(f"✅ Camera {i}: {width}x{height} @ {fps:.1f}fps")
                cap.release()
        
        if not self.available_cameras:
            print("❌ No cameras detected!")
            return False
        
        print(f"📊 Found {len(self.available_cameras)} working camera(s)")
        return True
    
    def select_camera(self, preferred_index=None):
        """Select camera with fallback logic"""
        if not self.available_cameras:
            print("❌ No cameras available")
            return False
        
        # If preferred index is specified, try to use it
        if preferred_index is not None:
            for cam in self.available_cameras:
                if cam['index'] == preferred_index:
                    return self.open_camera(preferred_index)
            print(f"⚠️ Camera {preferred_index} not available, using best available")
        
        # Select best camera (highest resolution, then highest FPS)
        best_camera = max(self.available_cameras, 
                         key=lambda x: (x['width'] * x['height'], x['fps']))
        
        print(f"🎯 Selected camera {best_camera['index']}: {best_camera['resolution']} @ {best_camera['fps']:.1f}fps")
        return self.open_camera(best_camera['index'])
    
    def open_camera(self, camera_index):
        """Open specific camera"""
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {camera_index}")
            return False
        
        self.current_camera = camera_index
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        return True
    
    def switch_to_next_camera(self):
        """Switch to next available camera"""
        if not self.available_cameras:
            return False
        
        if self.current_camera is None:
            return self.select_camera()
        
        # Find current camera index in available list
        current_idx = -1
        for i, cam in enumerate(self.available_cameras):
            if cam['index'] == self.current_camera:
                current_idx = i
                break
        
        # Switch to next camera (circular)
        next_idx = (current_idx + 1) % len(self.available_cameras)
        next_camera = self.available_cameras[next_idx]['index']
        
        print(f"🔄 Switching from camera {self.current_camera} to camera {next_camera}")
        return self.open_camera(next_camera)
    
    def get_frame(self):
        """Get frame from current camera"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            # Try to reopen camera
            print(f"⚠️ Camera {self.current_camera} frame read failed, attempting to reopen...")
            if self.open_camera(self.current_camera):
                ret, frame = self.cap.read()
        
        return ret, frame
    
    def get_camera_info(self):
        """Get current camera information"""
        if self.current_camera is None:
            return "No camera"
        
        for cam in self.available_cameras:
            if cam['index'] == self.current_camera:
                return f"{cam['index']} ({cam['resolution']})"
        
        return str(self.current_camera)
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

def parse_cam_arg():
    """Parse camera argument from command line"""
    try:
        if "--cam" in sys.argv:
            i = sys.argv.index("--cam")
            return int(sys.argv[i+1])
        # Also accept single positional like: python posture_live_full.py 1
        if len(sys.argv) >= 2 and sys.argv[1].isdigit():
            return int(sys.argv[1])
    except (ValueError, IndexError):
        pass
    return None

def show_camera_menu(camera_manager):
    """Show camera selection menu"""
    print("\n" + "="*60)
    print("🎥 CAMERA SELECTION MENU")
    print("="*60)
    
    for i, cam in enumerate(camera_manager.available_cameras):
        marker = " → " if cam['index'] == camera_manager.current_camera else "   "
        print(f"{marker}Camera {cam['index']}: {cam['resolution']} @ {cam['fps']:.1f}fps")
    
    print("\n💡 Controls:")
    print("   Press 'c' during runtime to switch cameras")
    print("   Press 'ESC' or 'q' to quit")
    print("   Press 'm' to show this menu again")
    print("="*60)

def main():
    print("🚗 Mahindra Driver Comfort Analyzer - Advanced Camera System")
    print("="*70)
    
    # Initialize camera manager
    camera_manager = CameraManager()
    
    if not camera_manager.available_cameras:
        print("❌ No cameras detected. Please check:")
        print("   - Camera drivers are installed")
        print("   - Camera permissions are granted")
        print("   - No other apps are using the camera")
        return
    
    # Parse command line camera preference
    preferred_camera = parse_cam_arg()
    
    # Select camera
    if not camera_manager.select_camera(preferred_camera):
        print("❌ Failed to initialize camera")
        return
    
    # Show camera menu
    show_camera_menu(camera_manager)
    
    # Initialize other components
    try:
        print("🔧 Initializing PoseExtractor...")
        pe = PoseExtractor(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        print("✅ PoseExtractor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize PoseExtractor: {e}")
        print("💡 Check if MediaPipe is properly installed")
        return
    
    try:
        print("🔧 Initializing AngleSmoother...")
        sm = AngleSmoother(alpha=0.25)
        print("✅ AngleSmoother initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AngleSmoother: {e}")
        return
    
    try:
        print("🔧 Initializing BodyConfig...")
        cfg = BodyConfig()
        print("✅ BodyConfig initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize BodyConfig: {e}")
        return
    
    try:
        print("🔧 Initializing DriverComfortAnalyzer...")
        comfort_analyzer = DriverComfortAnalyzer()
        print("✅ DriverComfortAnalyzer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DriverComfortAnalyzer: {e}")
        return

    # Setup logging
    out_dir = Path("posture_logs"); out_dir.mkdir(exist_ok=True)
    session = time.strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"driver_comfort_{session}.csv"
    
    # Enhanced logging with comfort scores
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "timestamp","body_state","upper_body","trunk_deg","neck_deg",
            "knee_L","knee_R","hip_L","hip_R","comfort_score","comfort_category",
            "posture_quality","ergonomic_risk","fatigue_indicator"
        ])

    # Session tracking
    t0 = time.monotonic()
    last_tick = t0
    good_time = 0.0
    high_time = 0.0
    fps_avg = None
    
    # Store posture history for fatigue analysis
    posture_history = []
    
    # Show startup info
    print(f"\n🎬 Starting session with camera {camera_manager.get_camera_info()}")
    print(f"📁 Logging to: {log_path}")
    print("🚀 Press 'c' to switch cameras, 'm' for menu, 'q' to quit\n")

    # Test PoseExtractor with a simple frame
    print("🧪 Testing PoseExtractor with current camera...")
    test_ret, test_frame = camera_manager.get_frame()
    if test_ret and test_frame is not None:
        try:
            test_result = pe.process_bgr(test_frame)
            if test_result is not None:
                print("✅ PoseExtractor test successful - ready to start monitoring")
            else:
                print("⚠️ PoseExtractor test failed - no pose detected in test frame")
        except Exception as e:
            print(f"❌ PoseExtractor test failed with error: {e}")
            print("💡 This may indicate a MediaPipe or camera compatibility issue")
    else:
        print("❌ Cannot get test frame from camera")
        return

    while True:
        # Get frame from camera
        ret, frame = camera_manager.get_frame()
        if not ret:
            print("❌ Failed to get frame, trying to recover...")
            time.sleep(0.1)
            continue
        
        t1 = time.monotonic()

        # Process frame
        try:
            print(f"🔍 Processing frame: {frame.shape if frame is not None else 'None'}")
            pose_result = pe.process_bgr(frame)
            print(f"📊 Pose result: {type(pose_result)} - {pose_result}")
            
            if pose_result is None:
                # No pose detected or error occurred
                print("⚠️ No pose detected - frame may be empty or pose detection failed")
                draw_text(frame, "No person detected", 10, 150, RED)
                draw_text(frame, "Position yourself in front of the camera", 10, 180, YEL)
                draw_text(frame, "Make sure you are visible in the frame", 10, 210, YEL)
                
                # Still show basic info even without pose
                now = time.monotonic()
                fps = 1.0 / max(1e-6, (time.monotonic() - t1))
                fps_avg = fps if fps_avg is None else (0.9*fps_avg + 0.1*fps)
                draw_text(frame, f"FPS: {fps_avg:.1f}", frame.shape[1] - 150, 25)
                
                # Show camera info
                camera_info = camera_manager.get_camera_info()
                draw_text(frame, f"Camera: {camera_info}", 10, frame.shape[0] - 20, CYA, 0.6, 2)
                
                cv2.imshow("Mahindra Driver Comfort Analyzer", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC or q to quit
                    break
                elif key == ord('c'):  # Switch camera
                    if camera_manager.switch_to_next_camera():
                        print(f"✅ Switched to camera {camera_manager.current_camera}")
                    else:
                        print("❌ Failed to switch camera")
                elif key == ord('m'):  # Show menu
                    show_camera_menu(camera_manager)
                
                continue
            
            # Unpack pose results
            results, pts, (img_h, img_w) = pose_result
            print(f"✅ Pose detected: {img_w}x{img_h}")
            
            # Process pose data
            pe.draw(frame, results)
            vis = get_vis(results)
            angles = sm(compute_angles(pts))
            
            # Store posture history for fatigue analysis
            posture_history.append(angles.copy())
            if len(posture_history) > 50:  # Keep last 50 data points
                posture_history.pop(0)
            
            body_state, extras = detect_body_state(pts, angles, vis, (img_h, img_w), cfg)
            upper = classify_upper_body(angles)
            rula = compute_rula(angles)
            reba = compute_reba(angles)
            fb = feedback_lines(angles)
            
            # Calculate driver comfort score
            session_duration = time.monotonic() - t0
            comfort_score = comfort_analyzer.calculate_comfort_score(
                angles, posture_history, session_duration
            )
            
            now = time.monotonic()
            dt = now - last_tick
            last_tick = now
            bucket = risk_buckets(rula[0], reba[0])
            if bucket == "good":
                good_time += dt
            elif bucket == "high":
                high_time += dt
            
            # Draw enhanced header with comfort info
            draw_header(frame, rula, reba, fb, now - t0, good_time, high_time, 
                       camera_manager.current_camera, comfort_score)
            
            # Draw comfort recommendations
            y_pos = draw_comfort_recommendations(frame, comfort_score)
            
            # Draw posture information
            y = y_pos + 20
            col = GRN if body_state in ("Standing","Sitting") else (YEL if body_state=="Unknown" else RED)
            draw_text(frame, f"Body: {body_state}", 10, y, col, 0.8, 2); y+=24
            draw_text(frame, f"Upper: {upper}" + (f" / {extras['lean_side']}" if extras['lean_side'] else ""), 10, y, WHT); y+=22
            draw_text(frame, f"trunk:{angles['trunk_from_vertical']:.1f}° neck:{angles['neck_from_vertical']:.1f}°", 10, y); y+=22
            draw_text(frame, f"knee L/R:{angles['left_knee_angle']:.1f}/{angles['right_knee_angle']:.1f}°", 10, y); y+=22
            draw_text(frame, f"hip L/R:{angles['left_hip_angle']:.1f}/{angles['right_hip_angle']:.1f}°", 10, y); y+=22
            
            if not extras["lower_visible"]:
                draw_text(frame, "Tip: Move camera back to include hips+knees.", 10, y+10, YEL, 0.6, 2)
            
            # Log data with comfort scores
            if int(now - t0) != int(now - t0 - (now - t1)):
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        round(now - t0,2), body_state, upper,
                        round(angles["trunk_from_vertical"],1), round(angles["neck_from_vertical"],1),
                        round(angles["left_knee_angle"],1), round(angles["right_knee_angle"],1),
                        round(angles["left_hip_angle"],1), round(angles["right_hip_angle"],1),
                        round(comfort_score.overall_comfort,1), comfort_score.comfort_category,
                        round(comfort_score.posture_quality,1), round(comfort_score.ergonomic_risk,1),
                        round(comfort_score.fatigue_indicator,1)
                    ])
        
        except Exception as e:
            # Handle any errors during pose processing
            print(f"⚠️ Error processing pose: {e}")
            import traceback
            traceback.print_exc()
            draw_text(frame, "Error processing pose", 10, 150, RED)
            draw_text(frame, "Check camera connection", 10, 180, YEL)
            draw_text(frame, "Press 'c' to try different camera", 10, 210, YEL)
            
            # Still show basic info
            now = time.monotonic()
            fps = 1.0 / max(1e-6, (time.monotonic() - t1))
            fps_avg = fps if fps_avg is None else (0.9*fps_avg + 0.1*fps)
            draw_text(frame, f"FPS: {fps_avg:.1f}", frame.shape[1] - 150, 25)
            
            # Show camera info
            camera_info = camera_manager.get_camera_info()
            draw_text(frame, f"Camera: {camera_info}", 10, frame.shape[0] - 20, CYA, 0.6, 2)
            
            cv2.imshow("Mahindra Driver Comfort Analyzer", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q to quit
                break
            elif key == ord('c'):  # Switch camera
                if camera_manager.switch_to_next_camera():
                    print(f"✅ Switched to camera {camera_manager.current_camera}")
                else:
                    print("❌ Failed to switch camera")
            elif key == ord('m'):  # Show menu
                show_camera_menu(camera_manager)
            
            continue
        
        # Display info (only reached when pose processing succeeds)
        fps = 1.0 / max(1e-6, (time.monotonic() - t1))
        fps_avg = fps if fps_avg is None else (0.9*fps_avg + 0.1*fps)
        draw_text(frame, f"FPS: {fps_avg:.1f}", img_w - 150, 25)
        
        # Show camera info
        camera_info = camera_manager.get_camera_info()
        draw_text(frame, f"Camera: {camera_info}", 10, img_h - 20, CYA, 0.6, 2)
        
        cv2.imshow("Mahindra Driver Comfort Analyzer", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q to quit
            break
        elif key == ord('c'):  # Switch camera
            if camera_manager.switch_to_next_camera():
                print(f"✅ Switched to camera {camera_manager.current_camera}")
            else:
                print("❌ Failed to switch camera")
        elif key == ord('m'):  # Show menu
            show_camera_menu(camera_manager)

    # Cleanup
    camera_manager.release()
    cv2.destroyAllWindows()
    
    print(f"\n💾 Driver comfort data saved: {log_path}")
    
    # Generate final comfort report
    if posture_history:
        session_info = {
            'duration': time.monotonic() - t0,
            'driver_id': 'test_driver',
            'seat_type': 'test_seat',
            'session_id': session
        }
        
        # Convert posture history to format expected by analyzer
        session_data = [{'angles': angles} for angles in posture_history]
        final_report = comfort_analyzer.generate_comfort_report(session_data, session_info)
        
        # Save comfort report
        report_path = out_dir / f"comfort_report_{session}.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        print(f"📊 Comfort report saved: {report_path}")
        
        # Print summary
        print(f"\n=== DRIVER COMFORT SUMMARY ===")
        print(f"Session Duration: {final_report['summary']['total_duration_minutes']:.1f} minutes")
        print(f"Average Comfort: {final_report['summary']['average_comfort']:.1f}/100")
        print(f"Final Category: {final_report['summary']['final_comfort_category']}")
        print(f"Comfort Trend: {final_report['comfort_trend']['description']}")
        print(f"Top Recommendations:")
        for rec in final_report['recommendations'][:3]:
            print(f"  • {rec}")

if __name__ == "__main__":
    main()
