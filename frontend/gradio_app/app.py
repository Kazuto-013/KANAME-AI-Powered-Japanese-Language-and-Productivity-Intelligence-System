from __future__ import annotations

import os
import sys
from typing import Any, Optional, Tuple

import numpy as np

try:
    import gradio as gr
except Exception:
    print("ERROR: gradio is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    from ai_engine.voice_intelligence import stt_engine
    from ai_engine.japanese_nlp import sudachi_engine
    from ai_engine.translation import translate_text, TranslationResult, available_engines
    from ai_engine import machine_learning as ml_module
    _ENGINES_AVAILABLE = True
    _ENGINE_ERR: Optional[str] = None
except Exception as e:
    _ENGINES_AVAILABLE = False
    _ENGINE_ERR = f"Engine import failed: {e}"
    stt_engine = None  # type: ignore
    sudachi_engine = None  # type: ignore
    translate_text = None  # type: ignore
    ml_module = None  # type: ignore


TITLE = "KANAME [要] — AI Intelligence Hub (Voice · Translation · ML Playground"
DESCRIPTION = """
Specialized Gradio interface for **KANAME** AI workflows.
Provides **Live Speech-to-Text** (faster-whisper CTranslate2),
**JP ↔ EN Bidirectional Translation** (LLM hybrid + 4 domain glossaries + dictionary fallback),
**Japanese NLP Analysis** (SudachiPy with keywords extraction + sentiment) and **ML Playground** (Classification / Clustering / Intent Detection using scikit-learn + Sudachi TF-IDF).
"""


def _engine_status_markdown() -> str:
    if not _ENGINES_AVAILABLE:
        return (
            "### ❌ **Engine unavailable:** "
            f"{_ENGINE_ERR or 'Unknown error.'}\n\n"
            "Please run this app from the project root as: `python frontend/gradio_app/app.py`"
        )
    ok = stt_engine.is_available()
    avail = available_engines() if callable(available_engines) else {}
    llm_keys = sum(1 for v in avail.values() if isinstance(v, dict) and v.get("available")) if avail else 0
    return (
        "### ✅ **KANAME AI Engine Status**  \n"
        f"- **STT Engine:** faster-whisper (CTranslate2 Whisper) → {'✅' if ok else '❌'}\n"
        f"- **NLP Engine (SudachiPy):** ✅ TF-IDF keywords + rule sentiment loaded  \n"
        f"- **Translation Engine:** {llm_keys} LLM keys · dictionary fallback always active  \n"
        f"- **ML Engine:** scikit-learn → LinearSVC + KMeans + hybrid intent (rules+ML)  \n"
    )


def run_stt(
    audio_file_path: Optional[str],
    microphone_audio: Optional[Tuple[int, np.ndarray]],
    model_size: str,
    language: str,
    beam_size: int,
    temperature: float,
    vad_filter: bool,
    initial_prompt: Optional[str],
):
    if not _ENGINES_AVAILABLE:
        return "[ERROR] Engines not loaded. " + (_ENGINE_ERR or "")
    if not stt_engine.is_available():
        return (
            "[ERROR] faster-whisper engine not available.\n"
            f"Reason: {stt_engine.get_error_message()}\n"
        )

    audio_numpy = None
    actual_path = audio_file_path
    if microphone_audio is not None:
        try:
            sr, arr = microphone_audio
            if sr != 16000:
                pass
            audio_numpy = np.asarray(arr)
            actual_path = None
        except Exception as e:
            return f"[ERROR] Failed to decode microphone audio: {e}"

    if actual_path is None and audio_numpy is None:
        return (
            "No audio input provided. "
            "Please either upload an audio file OR record from the microphone."
        )

    try:
        result = stt_engine.transcribe(
            audio_path=actual_path,
            audio_numpy=audio_numpy,
            model_size=model_size,
            language=language,
            beam_size=int(beam_size),
            temperature=float(temperature),
            vad_filter=bool(vad_filter),
            initial_prompt=initial_prompt or None,
            sample_rate=16000,
        )
    except FileNotFoundError as e:
        return f"[ERROR] File not found: {e}"
    except ValueError as e:
        return f"[ERROR] Bad parameters: {e}"
    except Exception as e:
        return f"[ERROR] STT failed: {e}"

    header = (
        f"✅ **Transcription complete.**\n"
        f"- **Language detected:** {result.language} "
        f"(p={result.language_probability:.2%})\n"
        f"- **Duration:** {result.duration:.2f}s  "
        f"**Model:** {result.model_size}  "
        f"**Segments:** {len(result.segments)}\n"
    )
    if result.warnings:
        header += "- **Warnings:** " + " ; ".join(result.warnings) + "\n"
    header += "\n---\n\n"
    return header + (result.text or "(no speech detected)")


def run_post_nlp(transcribed_text: str, sudachi_mode: str):
    if not transcribed_text or not transcribed_text.strip():
        return "No transcribed text to process. Run STT first."
    if transcribed_text.startswith("[ERROR]") or transcribed_text.startswith("No audio"):
        return "Fix STT step before running NLP."
    if not _ENGINES_AVAILABLE or sudachi_engine is None:
        return "SudachiPy NLP engine not loaded."

    try:
        only_text = transcribed_text
        if "\n---\n" in only_text:
            only_text = only_text.split("\n---\n", 1)[1]
        only_text = only_text.strip()
        if not only_text:
            return "(no clean text to analyze)"
        result = sudachi_engine.analyze(only_text, mode=sudachi_mode)
        header = (
            f"✅ **NLP Analysis (Mode {result.mode})**\n"
            f"- **Tokens:** {result.token_count}  "
            f"**Wakati:** `{result.wakati}`\n"
            f"- **Kana:** `{result.kana_only_text}`\n"
            f"- **Nouns:** {', '.join(result.noun_list) or '—'}\n"
            f"- **Verbs (dict form):** {', '.join(result.verb_list) or '—'}\n"
            f"- **Adjectives (dict form):** {', '.join(result.adjective_list) or '—'}\n"
        )
        pos_rows = "\n".join(
            f"  • {k} → {v}"
            for k, v in sorted(result.pos_counts.items(), key=lambda kv: -kv[1])
        )
        return header + "\n**POS counts:**\n" + (pos_rows or "(none)")
    except Exception as e:
        return f"[ERROR] NLP failed: {e}"


def run_keywords(text: str, top_n: int, mode: str) -> str:
    if not text or not text.strip():
        return "Please paste Japanese text first."
    if not _ENGINES_AVAILABLE or sudachi_engine is None:
        return "[ERROR] SudachiPy NLP engine not loaded."
    try:
        only_text = text
        if "\n---\n" in only_text:
            only_text = only_text.split("\n---\n", 1)[1]
        only_text = only_text.strip()
        result = sudachi_engine.extract_keywords(only_text, mode=mode, top_n=int(top_n))
        rows = "\n".join(
            f"| {i+1:>2} | `{it.keyword:<16}` | score `{it.score:.3f}` | freq {it.frequency} | {it.pos_hint or ''} |"
            for i, it in enumerate(result.items)
        )
        head = f"✅ **Keywords** (mode={result.mode}, top_n={result.keyword_count})  \n| # | keyword | score | freq | POS hint |\n|---|---|---|---|---|\n"
        return head + (rows or "| — | (no keywords) | — | — | — |")
    except Exception as e:
        return f"[ERROR] Keywords extraction failed: {e}"


def run_sentiment(text: str, mode: str) -> str:
    if not text or not text.strip():
        return "Please paste Japanese text first."
    if not _ENGINES_AVAILABLE or sudachi_engine is None:
        return "[ERROR] SudachiPy NLP engine not loaded."
    try:
        only_text = text
        if "\n---\n" in only_text:
            only_text = only_text.split("\n---\n", 1)[1]
        only_text = only_text.strip()
        r = sudachi_engine.analyze_sentiment(only_text, mode=mode)
        bar = f"[{('▓' * int(max(0, -r.score)*30)).ljust(30)}]" if r.score < 0 else (
            f"[{('▓' * int(r.score*30)).rjust(30)}]" if r.score > 0 else "[                              ]"
        )
        header = (
            f"✅ **Sentiment Result**  \n"
            f"- **Label:** `{r.label}`  ·  **Score:** `{r.score:+.3f}`  ·  **Confidence:** `{r.confidence:.3f}`\n"
            f"- **Gauge (left=neg, right=pos):** `{bar}`\n"
            f"- **Positive ({r.positive_count}):** {', '.join(r.positive_matches[:15]) or '—'}\n"
            f"- **Negative ({r.negative_count}):** {', '.join(r.negative_matches[:15]) or '—'}\n"
        )
        return header
    except Exception as e:
        return f"[ERROR] Sentiment failed: {e}"


def run_translate(
    text: str, source_lang: str, target_lang: str, domain: Optional[str]
) -> str:
    if not text or not text.strip():
        return "Enter some Japanese or English text first."
    if not _ENGINES_AVAILABLE or translate_text is None:
        return "[ERROR] Translation engine not loaded."
    try:
        only_text = text.strip()
        dom = None if domain in ("", "none", None) else domain
        r: TranslationResult = translate_text(only_text, source_lang=source_lang, target_lang=target_lang, domain=dom)
        quality = (
            f"- **Quality (0–1):** overall `{r.quality.overall:.3f}` · "
            f"length `{r.quality.length_ratio_score:.3f}` · numeric `{r.quality.numeric_preservation_score:.3f}` · "
            f"glossary `{r.quality.glossary_coverage_score:.3f}` · format `{r.quality.format_preservation_score:.3f}`\n"
            if r.quality is not None
            else ""
        )
        return (
            f"✅ **Translation complete**  \n"
            f"- Engine: `{r.engine}`  ·  source `{r.source_lang}` → target `{r.target_lang}`  ·  domain `{r.domain or '(default)'}`\n"
            f"- Glossary pairs applied: {r.glossary_hits}\n"
            f"{quality}"
            f"\n---\n\n{r.translated_text}"
        )
    except Exception as e:
        return f"[ERROR] Translation failed: {e}"


def run_ml_classify(text: str, top_k: int) -> str:
    if not text or not text.strip():
        return "Enter Japanese text to classify."
    if not _ENGINES_AVAILABLE or ml_module is None:
        return "[ERROR] ML engine not loaded."
    try:
        preds = ml_module.classify_text(text, top_k=int(top_k))
        prob_rows = "\n".join(
            f"| {i+1:>2} | `{l:<20}` | `{p:.3f}` |"
            for i, (l, p) in enumerate(sorted(preds.probabilities.items(), key=lambda kv: -kv[1])[:top_k])
        )
        return (
            f"✅ **Classification** (engine={preds.engine})  \n"
            f"- **Predicted:** `{preds.predicted_label}`  ·  **Confidence:** `{preds.confidence:.3f}`\n"
            f"| # | Label | Probability |\n|---|---|---|\n{prob_rows}"
        )
    except Exception as e:
        return f"[ERROR] Classify failed: {e}"


def run_ml_cluster(texts_raw: str, n_clusters: int) -> str:
    if not texts_raw or not texts_raw.strip():
        return "Paste newline-separated Japanese sentences to cluster."
    if not _ENGINES_AVAILABLE or ml_module is None:
        return "[ERROR] ML engine not loaded."
    try:
        texts = [line.strip() for line in texts_raw.splitlines() if line.strip()]
        r = ml_module.cluster_texts(texts, n_clusters=int(n_clusters))
        out = f"✅ **Clustering** — {r.total_texts} texts → {r.n_clusters} clusters (engine={r.engine})  \n\n"
        for c in r.clusters:
            out += (
                f"#### Cluster {c.cluster_id + 1} (size={c.size})  \n"
                f"- **Top keywords:** {', '.join(c.top_keywords[:8]) or '—'}\n"
                f"- **Example sentences:**\n"
                + "\n".join(f"  • {s}" for s in c.example_sentences[:4])
                + "\n\n"
            )
        return out
    except Exception as e:
        return f"[ERROR] Clustering failed: {e}"


def run_ml_intent(text: str) -> str:
    if not text or not text.strip():
        return "Enter Japanese text to detect intent."
    if not _ENGINES_AVAILABLE or ml_module is None:
        return "[ERROR] ML engine not loaded."
    try:
        p = ml_module.detect_intent(text)
        rows = "\n".join(
            f"| `{l:<14}` | `{v:.3f}` |"
            for l, v in sorted(p.probabilities.items(), key=lambda kv: -kv[1])
        )
        return (
            f"✅ **Intent Detection**  \n"
            f"- **Intent:** `{p.intent}`  ·  **Confidence:** `{p.confidence:.3f}`  \n"
            f"- **Method:** `{p.method}`" + (f"  ·  **Rule hit:** `{p.rule_hit}`" if p.rule_hit else "") + "\n"
            f"| Intent | Probability |\n|---|---|\n{rows}"
        )
    except Exception as e:
        return f"[ERROR] Intent failed: {e}"


def build_interface():
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)
        gr.Markdown(_engine_status_markdown())

        with gr.Tab("Speech to Text (STT)"):
            gr.Markdown("### Step 1 — Provide Audio + Parameters")
            with gr.Row():
                audio_upload = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="① Upload Audio File (wav / mp3 / m4a / flac / webm…)",
                )
                audio_mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="② — OR — Record from Microphone",
                )

            with gr.Accordion("⚙ Advanced STT parameters", open=False):
                with gr.Row():
                    model_size = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        value="base",
                        label="Whisper Model Size",
                        info="Base recommended for Japanese on CPU.",
                    )
                    language = gr.Dropdown(
                        choices=["auto", "ja", "en"],
                        value="auto",
                        label="Language",
                        info="'auto' detects automatically; use 'ja' to force Japanese.",
                    )
                with gr.Row():
                    beam_size = gr.Slider(1, 10, value=5, step=1, label="Beam Size")
                    temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Temperature (0=greedy)")
                with gr.Row():
                    vad_filter = gr.Checkbox(value=True, label="Enable VAD Filter (recommended)")
                    initial_prompt = gr.Textbox(
                        label="Initial Prompt (Japanese bias, optional)",
                        placeholder="例：日本語、形態素解析、機械学習",
                        lines=1,
                    )

            stt_btn = gr.Button("🔊 Run Speech Recognition", variant="primary", size="lg")
            stt_output = gr.Textbox(
                label="Transcribed Text (STT Output)",
                lines=10,
            )

        with gr.Tab("NLP Analysis (Post-Transcription)"):
            gr.Markdown("### Step 2 — NLP analysis on transcribed Japanese")
            ai_input = gr.Textbox(
                label="Transcribed Input (auto-populated from STT above)",
                lines=4,
                placeholder="Japanese text...",
            )
            with gr.Row():
                sudachi_mode = gr.Dropdown(
                    choices=["C", "B", "A"],
                    value="C",
                    label="Sudachi Split Mode",
                    info="C = longest (default), B = middle, A = shortest",
                )
                kw_top_n = gr.Slider(3, 40, value=15, step=1, label="Top-N keywords")
            nlp_btn = gr.Button("🧠 Run KANAME NLP → SudachiPy", variant="primary")
            with gr.Row():
                kw_btn = gr.Button("🎯 Extract Keywords", variant="secondary")
                sen_btn = gr.Button("😊 Analyze Sentiment", variant="secondary")
            nlp_output = gr.Markdown(label="NLP Result")
            kw_output = gr.Markdown(label="Keywords")
            sen_output = gr.Markdown(label="Sentiment")

        with gr.Tab("🌐 Translation (JP ↔ EN)"):
            gr.Markdown("### Bidirectional translation with domain glossary + LLM hybrid fallback")
            with gr.Row():
                t_src = gr.Dropdown(["auto", "ja", "en"], value="auto", label="Source language",
                                    info="'auto' detects automatically")
                t_tgt = gr.Dropdown(["en", "ja"], value="en", label="Target language")
                t_dom = gr.Dropdown(
                    ["none", "business_general", "technology_ai", "finance_accounting", "legal_contract"],
                    value="none", label="Domain glossary",
                    info="Optional: match source text domain for glossary pre-processing",
                )
            tr_input = gr.Textbox(label="Source text (JA or EN)", lines=7,
                                  value="御社の担当者様におかれましては、平素より格別のご高配を賜り、厚く御礼申し上げます。")
            tr_btn = gr.Button("🌐 Translate (LLM hybrid + dictionary fallback)", variant="primary", size="lg")
            tr_output = gr.Markdown(label="Translated output + quality score")

        with gr.Tab("🤖 ML Playground (Classify / Cluster / Intent)"):
            gr.Markdown("### Japanese Text ML — Sudachi TF-IDF + scikit-learn (LinearSVC / KMeans)")
            with gr.Tab("🏷 Classification"):
                cls_txt = gr.Textbox(label="Japanese text to classify", lines=5,
                                     value="深層学習モデルをGPUクラスタ上で分散学習させ、推論レイテンシを12msに改善しました。")
                cls_top = gr.Slider(2, 5, value=5, step=1, label="Top-K labels")
                cls_btn = gr.Button("🏷 Classify text", variant="primary")
                cls_out = gr.Markdown(label="Classification result")
            with gr.Tab("🧩 Clustering"):
                gr.Markdown("*Paste **newline-separated Japanese sentences** and cluster them with KMeans.*")
                clu_txt = gr.Textbox(
                    label="Sentences (1 per line)",
                    lines=11,
                    value="Pythonでscikit-learnを使って分類モデルを構築しました。\n"
                          "深層学習モデルをGPUで学習させ、推論APIとして公開しました。\n"
                          "Kubernetesクラスタにマイクロサービスをデプロイし、HPAを設定しました。\n"
                          "来週の契約締結に向けて、特許使用許諾書を確認中です。\n"
                          "本店監査役は、第12条に基づき報告書を提出しなければならない。\n"
                          "今回の事件は民法上の不法行為に該当する可能性があります。\n"
                          "被験者18名に対しアンケート調査を行い、因子分析を実施しました。\n"
                          "この研究結果は、統計学的に有意水準5%で効果が確認されました。\n"
                          "昨日の映画はとても面白かった。次の週末も観に行きたい。\n"
                          "近くのイタリアンレストランのパスタがめちゃくちゃ美味しかった。",
                )
                clu_n = gr.Slider(2, 6, value=3, step=1, label="Number of clusters (K)")
                clu_btn = gr.Button("🧩 Cluster sentences", variant="primary")
                clu_out = gr.Markdown(label="Clustering result")
            with gr.Tab("💬 Intent Detection"):
                int_txt = gr.Textbox(label="Japanese text (greeting / question / request / command / opinion / other)",
                                     lines=4, value="お世話になっております。KANAME株式会社の山田でございます。")
                int_btn = gr.Button("💬 Detect intent (rule-early + ML fallback)", variant="primary")
                int_out = gr.Markdown(label="Intent result")

        stt_btn.click(
            fn=run_stt,
            inputs=[audio_upload, audio_mic, model_size, language, beam_size, temperature, vad_filter, initial_prompt],
            outputs=stt_output,
        )

        def _pipe_to_nlp(x: str):
            if not x:
                return ""
            if "\n---\n" in x:
                return x.split("\n---\n", 1)[1].strip()
            return x.strip()

        stt_output.change(fn=_pipe_to_nlp, inputs=stt_output, outputs=ai_input)
        nlp_btn.click(fn=run_post_nlp, inputs=[ai_input, sudachi_mode], outputs=nlp_output)
        kw_btn.click(fn=run_keywords, inputs=[ai_input, kw_top_n, sudachi_mode], outputs=kw_output)
        sen_btn.click(fn=run_sentiment, inputs=[ai_input, sudachi_mode], outputs=sen_output)

        tr_btn.click(fn=run_translate, inputs=[tr_input, t_src, t_tgt, t_dom], outputs=tr_output)
        cls_btn.click(fn=run_ml_classify, inputs=[cls_txt, cls_top], outputs=cls_out)
        clu_btn.click(fn=run_ml_cluster, inputs=[clu_txt, clu_n], outputs=clu_out)
        int_btn.click(fn=run_ml_intent, inputs=[int_txt], outputs=int_out)

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
