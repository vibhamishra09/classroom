# --- Add near the top with other imports ---
import requests, json, os, gradio as gr

def ollama_feedback(transcript_text, topics, feedback):
    base = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    model_id = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    temp = float(os.getenv("OLLAMA_TEMP", "1"))
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "500"))
    read_timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
    require = os.getenv("REQUIRE_OLLAMA", "0") == "1"

    # shrink long transcripts to keep request snappy
    limit = int(os.getenv("PROMPT_TRANSCRIPT_CHARS", "3000"))
    t_snip = (transcript_text or "")
    if len(t_snip) > limit:
        half = limit // 2
        t_snip = t_snip[:half] + "\n...\n" + t_snip[-half:]

    topics_csv = ", ".join(topics[:8]) if topics else "N/A"
    metrics = {
        "wpm": feedback.get("wpm"),
        "fillers_pct": (feedback.get("fillers") or {}).get("filler_ratio_pct"),
        "structure_score": (feedback.get("structure") or {}).get("score_out_of_10"),
        "avg_sentence_len": feedback.get("avg_sentence_length_words"),
        "counts": (feedback.get("teaching") or {}).get("counts"),
    }
    def mmss(t): t = max(float(t or 0), 0.0); m = int(t//60); s = int(t%60); return f"{m:02d}:{s:02d}"
    segs = []
    for s in (feedback.get("__segments_for_prompt__") or [])[:80]:
        txt = (s.get("text","").strip().replace("\n"," "))[:90]
        segs.append(f"({mmss(s.get('start'))}–{mmss(s.get('end'))}) {txt}")

    system_msg = (
        "You are “TeachCoach”—a strict pedagogy coach. Only improvement advice; no audio/production; no praise. "
        "Do topic detection; then: Top 5 fixes (why+how), minute-by-minute fixes (mm:ss→fix), 2–3 examples, "
        "4 check questions, 10-step next-class plan. Be concrete with timestamps."
    )
    user_msg = (
        f"Context:\n- Provided Topics (hint): {topics_csv}\n- Metrics: {metrics}\n\n"
        f"Transcript (truncated):\n{t_snip}\n\nSegments:\n" + "\n".join(segs)
    )

    # pre-warm/keep-alive to avoid cold start wait
    try:
        requests.post(f"{base}/api/generate",
            json={"model": model_id, "prompt": " ", "stream": False, "keep_alive": "30m"},
            timeout=(10, 30))
    except Exception as e:
        print("[ollama] warmup skipped:", e)

    # 1) /api/chat
    try:
        payload = {
            "model": model_id,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": temp, "num_ctx": num_ctx, "num_predict": num_predict},
            "messages": [{"role": "system", "content": system_msg},
                         {"role": "user", "content": user_msg}],
        }
        r = requests.post(f"{base}/api/chat", json=payload, timeout=(10, read_timeout))
        if r.status_code == 404:
            raise FileNotFoundError("chat not available")
        r.raise_for_status()
        data = r.json()
        content = ((data.get("message") or {}).get("content") or "").strip()
        if content:
            return content, "ollama-chat"
    except Exception as e:
        print("[ollama/chat] failed:", e)

    # 2) Fallback /api/generate (older servers)
    try:
        merged = f"[SYSTEM]\n{system_msg}\n[/SYSTEM]\n[USER]\n{user_msg}\n[/USER]\n"
        payload = {
            "model": model_id,
            "stream": False,
            "keep_alive": "30m",
            "prompt": merged,
            "options": {"temperature": temp, "num_ctx": num_ctx, "num_predict": num_predict},
        }
        r = requests.post(f"{base}/api/generate", json=payload, timeout=(10, read_timeout))
        r.raise_for_status()
        data = r.json()
        content = (data.get("response") or "").strip()
        if content:
            return content, "ollama-generate"
    except Exception as e:
        print("[ollama/generate] failed:", e)
        if require:
            raise gr.Error(f"Ollama failed (require on): {e}")

    return None, "heuristic"
