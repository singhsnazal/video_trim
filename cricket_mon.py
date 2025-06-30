import streamlit as st
import os
import cv2
import easyocr
import tempfile
import re
from pathlib import Path
import shutil

# ==== Streamlit Page Configuration ====
st.set_page_config(page_title="Cricket Ball-by-Ball Video Analyzer", layout="wide")
st.title("🏏 Cricket Ball-by-Ball Video Analyzer & Clip Extractor")
st.markdown("Upload full-match footage and automatically extract ball-level clips classified by run type or over/ball.")

# ==== Session State Initialization ====
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'clip_dir' not in st.session_state:
    st.session_state.clip_dir = tempfile.mkdtemp()

# ==== Load EasyOCR Reader Once ====
@st.cache_resource
def get_ocr_reader():
    """Initializes and caches the EasyOCR reader."""
    return easyocr.Reader(['en'], gpu=False)

reader = get_ocr_reader()

# ==== Core Function Definitions ====

def save_clip(video_path, start_f, end_f, over_label, ball_number, run_type, output_dir):
    """
    Save a clip from the given video between two frame indices.

    Args:
        video_path (Path): Path to the input video.
        start_f (int): Start frame index.
        end_f (int): End frame index.
        over_label (str): Over string (e.g., '10.3').
        ball_number (int): Ball number within the over.
        run_type (str): Type of run (e.g., '4', 'dot').
        output_dir (Path): Output folder to save the clip.

    Returns:
        str: Full path of the saved clip.
    """
    over_folder = f"over_{str(over_label).split('.')[0]}"
    over_path = Path(output_dir) / over_folder
    over_path.mkdir(parents=True, exist_ok=True)

    filename = f"ball_{ball_number}_{run_type.replace(' ', '_')}.mp4"
    clip_path = over_path / filename

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))

    current_frame = start_f
    while current_frame <= end_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    out.release()
    cap.release()
    return str(clip_path)


def extract_over_and_runs(text):
    """
    Extract over and run info using regex.

    Args:
        text (str): OCR-extracted text.

    Returns:
        tuple[str, int]: Over and run detected.
    """
    over_match = re.search(r"\b(\d{1,2}\.\d)\b", text)
    run_match = re.search(r"\b(\d{1,3})\b", text)

    over = over_match.group(1) if over_match else None
    run = int(run_match.group(1)) if run_match else None

    # Avoid overlap of over number with run score
    if over and run and over_match and run == int(over.split('.')[0]):
        run = None
    return over, run


def process_video(video_path, clip_output_dir):
    """
    Extract and label ball-by-ball clips from full match video.

    Args:
        video_path (Path): Path to video file.
        clip_output_dir (str): Directory to save clips.

    Returns:
        list[dict]: Metadata of all extracted clips.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error(f"❌ Could not open video: {video_path.name}")
        return []

    fps = max(1, cap.get(cv2.CAP_PROP_FPS))
    interval = int(fps)
    frame_no, ball_no = 0, 1
    prev_over, prev_run = None, None
    start_frame = 0
    clip_log = []

    st.text(f"Processing: {video_path.name}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0, text="Analyzing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % interval == 0:
            h, w = frame.shape[:2]
            crop = frame[int(h * 0.85):h, 0:w]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ocr_text = " ".join(reader.readtext(gray, detail=0, paragraph=True))
            over, run = extract_over_and_runs(ocr_text)

            if over and over != prev_over:
                if prev_over is not None:
                    run_type = "dot"
                    if prev_run is not None and run is not None:
                        diff = run - prev_run
                        run_type = str(diff) if diff in [1, 2, 3, 4, 5, 6] else ("dot" if diff == 0 else "other")

                    clip_path = save_clip(video_path, start_frame, frame_no - 1, prev_over, ball_no, run_type, clip_output_dir)

                    clip_log.append({
                        "video_name": video_path.name,
                        "clip_path": clip_path,
                        "over_ball_str": prev_over,
                        "over": int(float(prev_over)),
                        "ball": ball_no,
                        "run_type": run_type
                    })

                    ball_no = 1 if int(over.split('.')[1]) == 1 and int(float(over)) > int(float(prev_over)) else ball_no + 1

                prev_over, prev_run = over, run
                start_frame = frame_no

        frame_no += 1
        progress.progress(frame_no / total_frames, text=f"Processing {video_path.name} ({int(100 * frame_no / total_frames)}%)")

    cap.release()
    progress.empty()
    return clip_log

# ==== File Upload Section ====

uploaded_files = st.file_uploader("Upload Cricket Match Videos", type=["mp4", "mov", "avi"], accept_multiple_files=True)

if uploaded_files and not st.session_state.get('processed_data'):
    all_clips = []
    with st.spinner("Analyzing videos. Please wait..."):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                tmp.write(file.getvalue())
                path = Path(tmp.name)
            clip_data = process_video(path, st.session_state.clip_dir)
            all_clips.extend(clip_data)
            os.unlink(path)

    if all_clips:
        st.session_state.processed_data = sorted(all_clips, key=lambda x: (x['video_name'], x['over'], x['ball']))
        st.success(f"✅ Processed {len(st.session_state.processed_data)} clips.")
    else:
        st.warning("⚠️ No clips extracted. Ensure the video has a visible scoreboard.")

# ==== Clip Display Section ====

if st.session_state.processed_data:
    st.markdown("---")
    st.header("🎬 Explore Extracted Clips")

    tab1, tab2 = st.tabs(["📊 View by Run Type", "🎯 View by Over & Ball"])

    with tab1:
        run_types = sorted(set(item['run_type'] for item in st.session_state.processed_data))
        selected_run = st.selectbox("Select Run Type", run_types)
        clips = [clip for clip in st.session_state.processed_data if clip['run_type'] == selected_run]
        st.info(f"{len(clips)} clips found for **{selected_run}** runs.")
        cols = st.columns(3)
        for i, clip in enumerate(clips):
            with cols[i % 3]:
                st.markdown(f"**{clip['video_name']}** - Over {clip['over_ball_str']}")
                st.video(clip['clip_path'])

    with tab2:
        videos = sorted(set(clip['video_name'] for clip in st.session_state.processed_data))
        selected_video = st.selectbox("1. Select Video", videos)
        if selected_video:
            overs = sorted(set(clip['over'] for clip in st.session_state.processed_data if clip['video_name'] == selected_video))
            selected_over = st.selectbox("2. Select Over", overs)
            if selected_over:
                balls = sorted(clip['ball'] for clip in st.session_state.processed_data if clip['over'] == selected_over and clip['video_name'] == selected_video)
                selected_ball = st.selectbox("3. Select Ball", balls)
                if selected_ball:
                    selected_clip = next(
                        (clip for clip in st.session_state.processed_data if clip['video_name'] == selected_video and clip['over'] == selected_over and clip['ball'] == selected_ball),
                        None
                    )
                    if selected_clip:
                        st.markdown(f"#### Clip: Over {selected_clip['over_ball_str']} | Run Type: **{selected_clip['run_type']}**")
                        st.video(selected_clip['clip_path'])
                    else:
                        st.warning("Clip not found.")

# ==== Cleanup Option ====
if st.button("🔄 Clear Cache and Restart"):
    if 'clip_dir' in st.session_state:
        shutil.rmtree(st.session_state.clip_dir, ignore_errors=True)
    st.session_state.clear()
    st.cache_resource.clear()
    st.rerun()
