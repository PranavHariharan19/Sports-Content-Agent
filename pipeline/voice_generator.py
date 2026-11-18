"""Voice generation using edge-tts with offline fallback and word-level timestamps (Section 7)."""
import asyncio
from pathlib import Path
import re
import time
from typing import List, Optional, Tuple

import edge_tts

from app.logging_config import get_logger
from schemas.voice import AudioMetadata, WordTimestamp

logger = get_logger(__name__)

DEFAULT_SPORTS_VOICE = "en-GB-RyanNeural"  # Professional, authoritative sports commentator voice
DEFAULT_SAMPLE_RATE = 24000


class VoiceGenerator:
    """Synthesizes broadcast audio from script text and extracts word-level timestamps."""

    def __init__(
        self,
        voice: str = DEFAULT_SPORTS_VOICE,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        job_id: str,
    ) -> Tuple[AudioMetadata, List[WordTimestamp]]:
        """Synthesize text using edge-tts with word boundaries, falling back to offline generator if needed."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean_text = " ".join(text.split()).strip()

        try:
            return await self._synthesize_edge_tts(clean_text, output_path, job_id)
        except Exception as e:
            logger.warning("edge-tts synthesis failed (%s). Triggering offline fallback...", str(e))
            return await self._synthesize_offline_fallback(clean_text, output_path, job_id)

    async def _synthesize_edge_tts(
        self,
        text: str,
        output_path: Path,
        job_id: str,
    ) -> Tuple[AudioMetadata, List[WordTimestamp]]:
        """Generate audio and word timestamps via Microsoft Edge TTS service."""
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
            boundary="WordBoundary",
        )

        words: List[WordTimestamp] = []
        audio_chunks: List[bytes] = []

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # offset and duration are in 100-nanosecond units (10^-7 sec)
                # 1 ms = 10,000 * 100ns units
                start_ms = int(chunk["offset"] // 10_000)
                duration_ms = int(chunk["duration"] // 10_000)
                end_ms = start_ms + max(50, duration_ms)
                word_text = chunk["text"].strip()
                if word_text:
                    words.append(
                        WordTimestamp(
                            word=word_text,
                            start_ms=start_ms,
                            end_ms=end_ms,
                        )
                    )

        if not audio_chunks:
            raise RuntimeError("No audio data returned by edge-tts service")

        # Write audio file (MP3)
        with open(output_path, "wb") as f:
            for b in audio_chunks:
                f.write(b)

        # Calculate duration
        total_duration = words[-1].end_sec if words else 0.0
        # If words ended early or duration is 0, estimate from file size or word count
        if total_duration <= 0.0:
            total_duration = max(1.0, len(text.split()) / 2.5)

        logger.info(
            "Synthesized voice with edge-tts for job %s: %s (%.2fs, %d words, voice=%s)",
            job_id,
            output_path,
            total_duration,
            len(words),
            self.voice,
        )

        metadata = AudioMetadata(
            audio_path=str(output_path),
            duration_seconds=round(total_duration, 2),
            voice_used=self.voice,
            sample_rate=DEFAULT_SAMPLE_RATE,
            word_count=len(words),
            format="mp3",
        )

        return metadata, words

    async def _synthesize_offline_fallback(
        self,
        text: str,
        output_path: Path,
        job_id: str,
    ) -> Tuple[AudioMetadata, List[WordTimestamp]]:
        """Fallback synthesis when network/edge-tts is unreachable."""
        words_raw = text.split()
        words: List[WordTimestamp] = []

        # Rhythmic pacing: average speaking rate ~2.5 words/sec (400ms per word)
        current_ms = 100
        for w in words_raw:
            # Word duration scales with length
            base_dur = max(200, int(len(w) * 65))
            if re.search(r"[\.,!?;]$", w):
                base_dur += 200  # pause on punctuation
            end_ms = current_ms + base_dur
            words.append(
                WordTimestamp(
                    word=w,
                    start_ms=current_ms,
                    end_ms=end_ms,
                )
            )
            current_ms = end_ms + 40  # 40ms inter-word gap

        total_duration = words[-1].end_sec if words else 1.0

        # Try pyttsx3 offline synthesis if available
        success_pyttsx3 = False
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # If output is mp3 or wav, save to path
            wav_fallback = output_path.with_suffix(".wav")
            engine.save_to_file(text, str(wav_fallback))
            engine.runAndWait()
            if wav_fallback.exists() and wav_fallback.stat().st_size > 0:
                output_path = wav_fallback
                success_pyttsx3 = True
        except Exception as e:
            logger.debug("pyttsx3 engine failed (%s); generating synthetic audio file", str(e))

        # If pyttsx3 failed, generate a clean placeholder wave
        if not success_pyttsx3:
            # Write a minimal valid RIFF WAV container with silence/tone
            import wave
            import struct
            with wave.open(str(output_path.with_suffix(".wav")), "w") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                num_frames = int(total_duration * 16000)
                silence_frames = struct.pack("<h", 0) * num_frames
                wav_file.writeframes(silence_frames)
            output_path = output_path.with_suffix(".wav")

        metadata = AudioMetadata(
            audio_path=str(output_path),
            duration_seconds=round(total_duration, 2),
            voice_used="offline-fallback",
            sample_rate=16000,
            word_count=len(words),
            format=output_path.suffix.lstrip("."),
        )

        logger.info(
            "Synthesized voice with offline fallback for job %s: %s (%.2fs, %d words)",
            job_id,
            output_path,
            total_duration,
            len(words),
        )

        return metadata, words
