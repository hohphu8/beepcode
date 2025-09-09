# Streamlit BeepCode - Encode/Decode + Mic
# @hohphu8 - 09/2025

import io
import zlib
from typing import List, Tuple, Dict, Optional

import numpy as np
import soundfile as sf
import streamlit as st

# Mic dependencies (optional)
HAS_AUDIOREC = False
HAS_WEBRTC = False
try:
    from st_audiorec import st_audiorec  # type: ignore
    HAS_AUDIOREC = True
except Exception:
    HAS_AUDIOREC = False
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase  # type: ignore
    import av  # needed by streamlit-webrtc
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

# =====================
# Cấu hình
# =====================
SR = 16000  # sample rate cố định cho file xuất/decoder
TONES = {"00": 400.0, "01": 600.0, "10": 800.0, "11": 1000.0}
ORDER = ["00", "01", "10", "11"]

st.set_page_config(page_title="BeepCode - Beep Encoder/Decoder", page_icon="✨", layout="centered")

# =====================
# Utils bit/byte
# =====================
def u16(x: int) -> int:
    return x & 0xFFFF

def u32(x: int) -> int:
    return x & 0xFFFFFFFF

def bytes_to_bits(b: bytes) -> str:
    return "".join(f"{x:08b}" for x in b)

def bits_to_bytes(bitstr: str) -> bytes:
    if len(bitstr) % 8:
        bitstr += "0" * (8 - (len(bitstr) % 8))
    out = bytearray()
    for i in range(0, len(bitstr), 8):
        out.append(int(bitstr[i:i+8], 2))
    return bytes(out)

def group2(bits: str) -> List[str]:
    if len(bits) % 2:
        bits += "0"
    return [bits[i:i+2] for i in range(0, len(bits), 2)]

def ungroup2(arr: List[str]) -> str:
    return "".join(arr)

# CRC32
def crc32_py(bytes_: bytes) -> int:
    return u32(zlib.crc32(bytes_))

# =====================
# DSP helpers
# =====================
def normalize(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    if y.size:
        y = y - y.mean()
        mx = float(np.max(np.abs(y)))
        if mx > 1e-9:
            y = (y / mx).astype(np.float32)
    return y

def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    n = len(y)
    if n == target_len:
        return y
    if n > target_len:
        return y[:target_len]
    pad = target_len - n
    left = pad // 2
    right = pad - left
    return np.pad(y, (left, right), mode="constant")

def synth_tone(freq: float, dur_ms: int) -> np.ndarray:
    n = int(SR * dur_ms / 1000.0)
    t = np.arange(n, dtype=np.float32) / SR
    y = np.sin(2 * np.pi * freq * t).astype(np.float32)
    # Hann nhẹ để nối mượt khi gap nhỏ/0
    wlen = max(16, int(0.08 * n))
    if wlen*2 < n:
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(wlen) / (wlen - 1)))
        y[:wlen] *= w
        y[-wlen:] *= w[::-1]
    return y

def silence_ms(dur_ms: int) -> np.ndarray:
    return np.zeros(int(SR * dur_ms / 1000.0), dtype=np.float32)

# VAD fallback
def simple_vad_segments(y: np.ndarray, frame_ms=10, min_seg_ms=120, min_sil_ms=80, energy_floor=0.02) -> List[Tuple[int, int]]:
    frame_n = max(1, int(SR * frame_ms / 1000.0))
    n_frames = max(1, len(y) // frame_n)
    energies = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        fr = y[i*frame_n:(i+1)*frame_n]
        energies[i] = float(np.sqrt(np.mean(fr*fr) + 1e-12))
    med = float(np.median(energies)) if n_frames else 0.001
    thr = max(energy_floor, 0.3 * med)
    voiced = (energies > thr).astype(np.int32)
    segs = []
    i = 0
    while i < n_frames:
        if voiced[i]:
            j = i
            while j < n_frames and voiced[j]:
                j += 1
            s = i * frame_n
            e = j * frame_n
            if (e - s) >= int(SR * min_seg_ms / 1000.0):
                segs.append((s, e))
            i = j
        else:
            i += 1
    merged = []
    min_sil_n = int(SR * min_sil_ms / 1000.0)
    for s, e in segs:
        if not merged:
            merged.append([s, e])
        else:
            ps, pe = merged[-1]
            if s - pe < min_sil_n:
                merged[-1][1] = e
            else:
                merged.append([s, e])
    return [(s, e) for s, e in merged]

# Cắt theo khoảng lặng 0 tuyệt đối
def segment_by_silence(raw: np.ndarray, min_gap_ms=40, min_seg_ms=100, eps=1e-7) -> List[Tuple[int, int]]:
    min_gap = int(SR * min_gap_ms / 1000.0)
    min_seg = int(SR * min_seg_ms / 1000.0)
    is_sil = (np.abs(raw) <= eps)
    N = len(raw)
    gaps: List[Tuple[int, int]] = []
    i = 0
    while i < N:
        if is_sil[i]:
            j = i + 1
            while j < N and is_sil[j]:
                j += 1
            if (j - i) >= min_gap:
                gaps.append((i, j))
            i = j
        else:
            i += 1
    starts = [0] + [b for _, b in gaps]
    ends = [a for a, _ in gaps] + [N]
    segs: List[Tuple[int, int]] = []
    for s, e in zip(starts, ends):
        if (e - s) >= min_seg:
            seg = raw[s:e]
            if float(np.max(np.abs(seg))) > eps * 10:
                segs.append((s, e))
    return segs

# Cosine similarity
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-9
    return float(np.dot(a, b) / denom)

# =====================
# Encode/Decode core
# =====================
def encode_to_wav_bytes(text: str, symbol_ms: int = 320, gap_ms: int = 120, rep: int = 1) -> bytes:
    payload = text.encode("utf-8")
    payload_bits = bytes_to_bits(payload)
    crc = crc32_py(payload)

    sync_syms = ["11"] * 8
    cal_syms = ["00", "01", "10", "11"]
    header_bits = f"{u16(len(payload)):016b}{u16(symbol_ms):016b}{u16(gap_ms):016b}"
    header_syms = group2(header_bits)

    payload_syms = group2(payload_bits)
    if rep > 1:
        payload_syms = [s for s in payload_syms for _ in range(rep)]

    crc_bits = f"{u32(crc):032b}"
    crc_syms = group2(crc_bits)

    all_syms = sync_syms + cal_syms + header_syms + payload_syms + crc_syms

    pieces: List[np.ndarray] = [silence_ms(150)]
    for tb in all_syms:
        pieces.append(synth_tone(TONES[tb], symbol_ms))
        if gap_ms > 0:
            pieces.append(silence_ms(gap_ms))
    pieces.append(silence_ms(200))

    y = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    y = (0.85 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)

    bio = io.BytesIO()
    sf.write(bio, y, SR, format="WAV", subtype="PCM_16")
    return bio.getvalue()


def decode_from_wav_bytes(wav_bytes: bytes) -> Tuple[str, Dict]:
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        x_old = np.linspace(0, 1, len(data), endpoint=False)
        x_new = np.linspace(0, 1, int(len(data) * SR / sr), endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
    raw = data.copy()
    norm = normalize(data)

    segs = segment_by_silence(raw, 40, 100)
    if len(segs) < 12 or len(segs) > 10000:
        segs = simple_vad_segments(norm, 10, 120, 80, 0.02)
    if len(segs) < 12:
        raise ValueError(f"Segments too few: {len(segs)} (need >=12)")

    cal_start = 8
    cal_lens = [segs[cal_start + i][1] - segs[cal_start + i][0]
                for i in range(4)]
    sym_len = int(np.median(cal_lens))

    protos: Dict[str, np.ndarray] = {}
    for idx, code in enumerate(ORDER):
        s, e = segs[cal_start + idx]
        chunk = norm[s:e]
        protos[code] = pad_or_trim(chunk, sym_len)

    symbols: List[str] = []
    sims: List[float] = []
    for s, e in segs:
        chunk = pad_or_trim(norm[s:e], sym_len)
        best, bestsim = "00", -1.0
        for k in ORDER:
            sc = cosine_sim(chunk, protos[k])
            if sc > bestsim:
                best, bestsim = k, sc
        symbols.append(best)
        sims.append(bestsim)

    def find_sync(sym: List[str], code: str = "11", ln: int = 8) -> Tuple[int, int]:
        target = [code] * ln
        for i in range(0, len(sym) - ln + 1):
            if sym[i:i+ln] == target:
                return i, i + ln
        return -1, -1

    i0, i1 = find_sync(symbols, "11", 8)
    if i0 < 0:
        raise ValueError("SYNC not found. Try larger symbol/gap.")

    def try_header(hsym: int):
        hbits = ungroup2(symbols[i1 + 4: i1 + 4 + hsym])
        if hsym == 24:
            if len(hbits) < 48:
                return None
            payload_len = int(hbits[:16], 2)
            symbol_ms = int(hbits[16:32], 2)
            gap_ms = int(hbits[32:48], 2)
            start = i1 + 4 + 24
        else:
            if len(hbits) < 32:
                return None
            payload_len = int(hbits[:16], 2)
            symbol_ms = int(hbits[16:24], 2)
            gap_ms = int(hbits[24:32], 2)
            start = i1 + 4 + 16
        return dict(payload_len=payload_len, symbol_ms=symbol_ms, gap_ms=gap_ms, header_symbols=hsym, start=start)

    hdr = try_header(24) or try_header(16)
    if hdr is None:
        raise ValueError("Header parse failed")

    base_syms = hdr["payload_len"] * 4
    rest = symbols[hdr["start"]:]
    if len(rest) < 16:
        raise ValueError("CRC symbols missing")

    rep = 1
    if base_syms > 0:
        q = (len(rest) - 16) / base_syms
        cand = int(round(q))
        if cand in (1, 2, 3) and abs(q - cand) < 0.34:
            rep = cand

    payload_syms_rep = rest[: base_syms * rep]
    crc_syms = rest[base_syms * rep: base_syms * rep + 16]

    def majority_reduce(sym_list: List[str], repn: int) -> List[str]:
        if repn <= 1:
            return sym_list[:]
        out = []
        for i in range(0, len(sym_list), repn):
            chunk = sym_list[i:i+repn]
            vals, counts = np.unique(chunk, return_counts=True)
            out.append(vals[int(np.argmax(counts))])
        return out

    payload_syms = majority_reduce(payload_syms_rep, rep)
    if len(payload_syms) < base_syms:
        raise ValueError("Payload symbols too short after majority reduce")

    payload_bits = ungroup2(payload_syms)[: hdr["payload_len"] * 8]
    payload = bits_to_bytes(payload_bits)

    crc_recv = int(ungroup2(crc_syms), 2)
    crc_calc = crc32_py(payload)
    if crc_recv != crc_calc:
        raise ValueError(
            f"CRC mismatch: got={crc_recv:08x} calc={crc_calc:08x}")

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        text = payload.decode("utf-8", errors="replace")

    meta = dict(
        segments=len(segs), symbols=len(symbols), avg_similarity=float(np.mean(sims) if sims else 0.0),
        symLen=sym_len, repDetected=rep, headerSymbols=hdr["header_symbols"],
        symbolMs_hdr=hdr["symbol_ms"], gapMs_hdr=hdr["gap_ms"], payloadLen=hdr["payload_len"],
    )
    return text, meta

# =====================
# UI - Streamlit
# =====================
st.title("BeepCode - Encoder/Decoder")
st.caption("Encode text thành WAV beep-beep, và decode từ file hoặc microphone.")

webrtc_err = None
with st.sidebar:
    st.header("Cấu hình Encode")
    symbol_ms = st.number_input("Symbol (ms)", 105, 1000, 320, 5)
    gap_ms = st.number_input("Gap (ms)", 40, 500, 120, 5)
    rep = st.selectbox("Repetition", [1, 3, 5], index=0, help="Lặp symbol payload để chống lỗi (majority)")

    st.caption(f"webrtc: {'OK' if HAS_WEBRTC else 'FAIL'}")
    if webrtc_err:
        st.code(webrtc_err)

# ===== Encode =====
st.subheader("Encode → WAV")
text_in = st.text_area("Văn bản", "", height=120, placeholder="Hello World! BeepCode...")
col2 = st.columns(2)[0]
with col2:
    if st.button("Tạo & Tải về", use_container_width=True):
        if not text_in:
            st.warning("Vui lòng nhập văn bản để mã hóa.")
        else:
            try:
                wav_bytes = encode_to_wav_bytes(text_in, symbol_ms, gap_ms, rep)
                st.session_state["last_wav"] = wav_bytes
                st.download_button("Download output.wav", wav_bytes, file_name="output.wav", mime="audio/wav")
                st.audio(wav_bytes, format="audio/wav")
            except Exception as e:
                st.error(f"Encode lỗi: {e}")

st.markdown("---")

# ===== Decode =====
st.subheader("Decode")
mode = st.radio("Nguồn vào", ["Upload WAV", "Microphone"], horizontal=True)

if mode == "Upload WAV":
    up = st.file_uploader("Chọn file WAV (có thể dùng file vừa tải ở trên)", 
        type=["wav", "wave", "audio"], accept_multiple_files=False)
    if up is not None:
        try:
            wav_bytes = up.read()
            text_dec, meta = decode_from_wav_bytes(wav_bytes)
            st.success("Decode OK")
            st.text_area("Decoded Text", text_dec, height=120)
            st.json(meta, expanded=False)
        except Exception as e:
            st.error(f"Decode lỗi: {e}")
else:
    st.info("Chọn 1 trong 2 cách thu mic bên dưới. Nếu cách 1 không có, dùng cách 2.")

    # Cách 1: st-audiorec
    if HAS_AUDIOREC:
        st.write("**Mic cách 1 - st-audiorec**")
        wav_audio = st_audiorec()  # returns wav bytes or None
        if wav_audio is not None:
            try:
                text_dec, meta = decode_from_wav_bytes(wav_audio)
                st.success("Decode OK (st-audiorec)")
                st.text_area("Decoded Text", text_dec, height=120, key="dec1")
                st.json(meta, expanded=False)
            except Exception as e:
                st.error(f"Decode lỗi: {e}")
    else:
        st.warning("st-audiorec chưa cài. Cài bằng: `pip install st-audiorec`.")

    st.markdown("---")

    # Cách 2: streamlit-webrtc (ổn định hơn, realtime)
    if HAS_WEBRTC:
        st.write("**Mic cách 2 — streamlit-webrtc**")
        st.caption("Bấm START, phát beep từ loa khác; sau đó STOP và decode.")

        class Collector(AudioProcessorBase):
            def __init__(self) -> None:
                self.frames = []

            def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
                # frame: shape (channels, samples)
                pcm = frame.to_ndarray()
                if pcm.ndim == 2:
                    # mono hóa
                    mono = pcm.mean(axis=0).astype("float32")
                else:
                    mono = pcm.astype("float32")
                self.frames.append(mono)
                return frame

        ctx = webrtc_streamer(
            key="beepcode-webrtc",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True, "video": False},
            audio_processor_factory=Collector,
        )

        if ctx and ctx.state.playing:
            st.info("Đang nghe mic... (bấm STOP & Decode)")

        if st.button("STOP & Decode", use_container_width=True):
            try:
                proc = ctx.audio_processor  # type: ignore
                if proc is None or not getattr(proc, "frames", None):
                    st.error("Chưa thu được audio.")
                else:
                    import numpy as np, io, soundfile as sf
                    pcm = np.concatenate(proc.frames).astype("float32")
                    # Lấy sample rate vào (nhiều khi là 32000/48000); không có thì mặc định SR
                    in_sr = getattr(getattr(ctx, "client_settings", None), "audio", None) or 48000
                    # Resample về 16000 nếu cần
                    SR = 16000
                    if in_sr != SR:
                        x_old = np.linspace(0, 1, len(pcm), endpoint=False)
                        x_new = np.linspace(0, 1, int(len(pcm) * SR / in_sr), endpoint=False)
                        pcm = np.interp(x_new, x_old, pcm).astype("float32")
                    bio = io.BytesIO()
                    sf.write(bio, pcm, SR, format="WAV", subtype="PCM_16")
                    text_dec, meta = decode_from_wav_bytes(bio.getvalue())
                    st.success("Decode OK (webrtc)")
                    st.text_area("Decoded Text", text_dec, height=120, key="dec2")
                    st.json(meta, expanded=False)
            except Exception as e:
                st.error(f"Decode lỗi: {e}")

    else:
        st.warning("streamlit-webrtc chưa cài. Cài: `pip install streamlit-webrtc av`. Nếu lỗi libav, cài thêm FFmpeg cho hệ điều hành.")

st.markdown("""
---
**Gợi ý**: symbol 320ms / gap 120ms siêu an toàn. Mạnh tay hơn có thể 120/50 hoặc 105/40. 

Made with ❤️ by [@hohphu8](https://github.com/hohphu8)
"""
)
