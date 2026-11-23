"""Streamlit UI for local audio → SRT transcription."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from transcribe_local import (
    SUPPORTED_EXTS,
    segments_to_srt,
    segments_to_txt,
    transcribe_file,
)

st.set_page_config(
    page_title="Audio → SRT",
    page_icon="🎙️",
    layout="centered",
)

st.title("Audio → SRT")
st.caption("Lokal · faster-whisper · GPU jika tersedia")

with st.sidebar:
    st.header("Opsi")
    model = st.selectbox(
        "Model",
        options=["small", "medium", "large-v3", "base", "tiny"],
        index=0,
        help="small = cepat, large-v3 = akurat (lebih lambat / download lebih besar)",
    )
    language = st.selectbox(
        "Bahasa",
        options=[
            ("Auto-detect", None),
            ("Indonesia (id)", "id"),
            ("English (en)", "en"),
            ("Japanese (ja)", "ja"),
            ("Korean (ko)", "ko"),
            ("Chinese (zh)", "zh"),
        ],
        format_func=lambda x: x[0],
    )[1]
    device = st.selectbox(
        "Device",
        options=["auto", "cuda", "cpu"],
        index=0,
    )
    vad = st.checkbox("VAD filter (skip silence)", value=True)
    beam_size = st.slider("Beam size", min_value=1, max_value=10, value=5)

AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
    ".mp4": "audio/mp4",
    ".wma": "audio/x-ms-wma",
}

uploaded = st.file_uploader(
    "Upload audio",
    type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTS)],
    help="wav / m4a / mp3 / flac / ogg / webm / mp4",
)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower() or ".wav"
    stem = Path(uploaded.name).stem
    mime = AUDIO_MIME.get(suffix, "audio/wav")
    try:
        st.audio(uploaded.getvalue(), format=mime)
    except Exception:
        st.caption(f"Preview tidak tersedia untuk `{suffix}` — transcribe tetap bisa jalan.")

    if st.button("Transcribe → SRT", type="primary", use_container_width=True):
        with st.spinner("Transcribing… (pertama kali model akan di-download)"):
            try:
                with tempfile.TemporaryDirectory(prefix="ats-web-") as tmp:
                    src = Path(tmp) / f"upload{suffix}"
                    src.write_bytes(uploaded.getvalue())
                    segments, meta = transcribe_file(
                        src,
                        model_name=model,
                        language=language,
                        device=device,
                        vad=vad,
                        beam_size=beam_size,
                    )
            except Exception as exc:
                st.error(f"Gagal: {exc}")
                st.stop()

        srt_text = segments_to_srt(segments)
        txt_text = segments_to_txt(segments)
        cue_count = sum(
            1 for s in segments if (s.text or "").strip()
        )

        st.success(
            f"{cue_count} cues · bahasa `{meta['language']}` "
            f"(p={meta['language_probability']:.2f}) · "
            f"{meta['device']}/{meta['compute_type']}"
        )

        dl_srt, dl_txt = st.columns(2)
        with dl_srt:
            st.download_button(
                "Download .srt",
                data=srt_text.encode("utf-8"),
                file_name=f"{stem}.srt",
                mime="application/x-subrip",
                type="primary",
                use_container_width=True,
            )
        with dl_txt:
            st.download_button(
                "Download .txt",
                data=txt_text.encode("utf-8"),
                file_name=f"{stem}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        tab_srt, tab_txt, tab_table = st.tabs(["SRT", "Teks", "Cue table"])
        with tab_srt:
            st.code(srt_text or "(kosong)", language="text")
        with tab_txt:
            st.text_area("Transcript", txt_text, height=240)
        with tab_table:
            rows = [
                {
                    "start": round(float(s.start), 3),
                    "end": round(float(s.end), 3),
                    "text": (s.text or "").strip(),
                }
                for s in segments
                if (s.text or "").strip()
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.info("Upload file audio untuk mulai.")
