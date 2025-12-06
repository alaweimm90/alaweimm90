"""
Advanced Transcription Engine for Research Content

Handles transcription of lectures, presentations, and research discussions
with specialized vocabulary and technical term recognition.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from enum import Enum
import wave
import contextlib

# Speech recognition libraries
import whisper
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Natural language processing
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

logger = logging.getLogger(__name__)


class TranscriptionModel(Enum):
    """Available transcription models"""
    WHISPER_LARGE = "whisper_large"
    WHISPER_MEDIUM = "whisper_medium"
    WHISPER_BASE = "whisper_base"
    WAV2VEC2 = "wav2vec2"
    GOOGLE_SPEECH = "google_speech"
    AZURE_SPEECH = "azure_speech"
    AWS_TRANSCRIBE = "aws_transcribe"


class ContentType(Enum):
    """Type of research content"""
    LECTURE = "lecture"
    PRESENTATION = "presentation"
    DISCUSSION = "discussion"
    INTERVIEW = "interview"
    LAB_RECORDING = "lab_recording"
    CONFERENCE_TALK = "conference_talk"
    TUTORIAL = "tutorial"
    UNKNOWN = "unknown"


@dataclass
class TranscriptionSegment:
    """Segment of transcribed content"""
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker: Optional[str] = None
    language: Optional[str] = None
    technical_terms: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    emphasis: Optional[float] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result"""
    full_text: str
    segments: List[TranscriptionSegment]
    duration: float
    content_type: ContentType
    language: str
    speakers: List[str]
    technical_vocabulary: List[str]
    key_points: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0


class TranscriptionEngine:
    """
    Comprehensive transcription engine for research audio/video content
    with support for technical vocabulary and multi-speaker scenarios
    """

    def __init__(
        self,
        model: TranscriptionModel = TranscriptionModel.WHISPER_LARGE,
        language: str = "en",
        enable_gpu: bool = True,
        technical_vocab_path: Optional[str] = None
    ):
        """
        Initialize transcription engine

        Args:
            model: Transcription model to use
            language: Primary language for transcription
            enable_gpu: Use GPU acceleration if available
            technical_vocab_path: Path to technical vocabulary file
        """
        self.model_type = model
        self.language = language
        self.device = "cuda" if enable_gpu and torch.cuda.is_available() else "cpu"

        # Initialize models
        self._initialize_models()

        # Load technical vocabulary
        self.technical_vocab = self._load_technical_vocabulary(technical_vocab_path)

        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")

        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        logger.info(f"TranscriptionEngine initialized with {model.value} on {self.device}")

    def _initialize_models(self):
        """Initialize transcription models"""
        if self.model_type in [TranscriptionModel.WHISPER_LARGE,
                               TranscriptionModel.WHISPER_MEDIUM,
                               TranscriptionModel.WHISPER_BASE]:
            # Initialize Whisper
            model_size = self.model_type.value.split('_')[1]
            self.whisper_model = whisper.load_model(model_size, device=self.device)

        elif self.model_type == TranscriptionModel.WAV2VEC2:
            # Initialize Wav2Vec2
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-large-960h"
            )
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-large-960h"
            ).to(self.device)

        elif self.model_type == TranscriptionModel.GOOGLE_SPEECH:
            # Initialize Google Speech Recognition
            self.recognizer = sr.Recognizer()

        # Add other model initializations as needed

    def transcribe(
        self,
        input_path: Union[str, Path],
        content_type: Optional[ContentType] = None,
        extract_key_points: bool = True,
        identify_speakers: bool = True,
        timestamp_precision: float = 0.1
    ) -> TranscriptionResult:
        """
        Transcribe audio/video file

        Args:
            input_path: Path to audio/video file
            content_type: Optional content type hint
            extract_key_points: Extract key points from transcription
            identify_speakers: Perform speaker diarization
            timestamp_precision: Precision for timestamps in seconds

        Returns:
            TranscriptionResult with full transcription and metadata
        """
        input_path = Path(input_path)

        # Extract audio if video
        audio_path = self._extract_audio_from_video(input_path)

        # Get audio duration
        duration = self._get_audio_duration(audio_path)

        # Detect content type if not provided
        if content_type is None:
            content_type = self._detect_content_type(audio_path)

        # Perform transcription
        segments = self._transcribe_audio(audio_path, timestamp_precision)

        # Post-process transcription
        segments = self._post_process_segments(segments)

        # Identify technical terms
        technical_terms = self._identify_technical_terms(segments)

        # Extract speaker information if requested
        speakers = []
        if identify_speakers:
            speakers = self._identify_unique_speakers(segments)

        # Generate full text
        full_text = " ".join([seg.text for seg in segments])

        # Extract key points if requested
        key_points = []
        if extract_key_points:
            key_points = self._extract_key_points(full_text, segments)

        # Calculate quality score
        quality_score = self._assess_quality(segments)

        return TranscriptionResult(
            full_text=full_text,
            segments=segments,
            duration=duration,
            content_type=content_type,
            language=self.language,
            speakers=speakers,
            technical_vocabulary=technical_terms,
            key_points=key_points,
            metadata={
                'input_file': str(input_path),
                'model': self.model_type.value,
                'timestamp_precision': timestamp_precision
            },
            quality_score=quality_score
        )

    def _extract_audio_from_video(self, input_path: Path) -> Path:
        """Extract audio track from video file if needed"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

        if input_path.suffix.lower() in video_extensions:
            # Extract audio using pydub
            audio = AudioSegment.from_file(str(input_path))
            audio_path = input_path.with_suffix('.wav')
            audio.export(str(audio_path), format='wav')
            return audio_path

        return input_path

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds"""
        with contextlib.closing(wave.open(str(audio_path), 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration

    def _detect_content_type(self, audio_path: Path) -> ContentType:
        """Detect type of content from audio characteristics"""
        # Simplified detection based on audio features
        # In production, would use more sophisticated analysis

        audio = AudioSegment.from_wav(str(audio_path))

        # Analyze silence patterns
        chunks = split_on_silence(
            audio,
            min_silence_len=1000,
            silence_thresh=audio.dBFS - 14
        )

        # Heuristics based on chunk patterns
        avg_chunk_length = np.mean([len(chunk) for chunk in chunks]) if chunks else 0

        if avg_chunk_length > 30000:  # Long continuous speech
            return ContentType.LECTURE
        elif len(chunks) > 50:  # Many short segments
            return ContentType.DISCUSSION
        else:
            return ContentType.PRESENTATION

    def _transcribe_audio(
        self,
        audio_path: Path,
        timestamp_precision: float
    ) -> List[TranscriptionSegment]:
        """Perform actual transcription"""
        segments = []

        if self.model_type in [TranscriptionModel.WHISPER_LARGE,
                               TranscriptionModel.WHISPER_MEDIUM,
                               TranscriptionModel.WHISPER_BASE]:
            segments = self._transcribe_with_whisper(audio_path, timestamp_precision)

        elif self.model_type == TranscriptionModel.WAV2VEC2:
            segments = self._transcribe_with_wav2vec2(audio_path, timestamp_precision)

        elif self.model_type == TranscriptionModel.GOOGLE_SPEECH:
            segments = self._transcribe_with_google(audio_path, timestamp_precision)

        return segments

    def _transcribe_with_whisper(
        self,
        audio_path: Path,
        timestamp_precision: float
    ) -> List[TranscriptionSegment]:
        """Transcribe using OpenAI Whisper"""
        result = self.whisper_model.transcribe(
            str(audio_path),
            language=self.language,
            verbose=False,
            word_timestamps=True
        )

        segments = []
        for segment in result.get('segments', []):
            segments.append(TranscriptionSegment(
                text=segment['text'].strip(),
                start_time=segment['start'],
                end_time=segment['end'],
                confidence=segment.get('confidence', 0.95),
                language=result.get('language', self.language)
            ))

        return segments

    def _transcribe_with_wav2vec2(
        self,
        audio_path: Path,
        timestamp_precision: float
    ) -> List[TranscriptionSegment]:
        """Transcribe using Wav2Vec2"""
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Process in chunks
        chunk_size = 16000 * 30  # 30 seconds
        segments = []

        for i in range(0, waveform.shape[1], chunk_size):
            chunk = waveform[:, i:i+chunk_size]

            # Get input values
            input_values = self.wav2vec2_processor(
                chunk.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values.to(self.device)

            # Get predictions
            with torch.no_grad():
                logits = self.wav2vec2_model(input_values).logits

            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.wav2vec2_processor.batch_decode(predicted_ids)[0]

            segments.append(TranscriptionSegment(
                text=transcription,
                start_time=i / 16000,
                end_time=min((i + chunk_size) / 16000, waveform.shape[1] / 16000),
                confidence=0.9,  # Wav2Vec2 doesn't provide confidence
                language=self.language
            ))

        return segments

    def _transcribe_with_google(
        self,
        audio_path: Path,
        timestamp_precision: float
    ) -> List[TranscriptionSegment]:
        """Transcribe using Google Speech Recognition"""
        segments = []

        # Load audio
        with sr.AudioFile(str(audio_path)) as source:
            audio = self.recognizer.record(source)

        # Split into chunks for better accuracy
        audio_segment = AudioSegment.from_wav(str(audio_path))
        chunks = split_on_silence(
            audio_segment,
            min_silence_len=500,
            silence_thresh=audio_segment.dBFS - 14
        )

        current_time = 0.0
        for chunk in chunks:
            # Export chunk
            chunk_path = Path("/tmp/chunk.wav")
            chunk.export(str(chunk_path), format="wav")

            # Transcribe chunk
            with sr.AudioFile(str(chunk_path)) as source:
                chunk_audio = self.recognizer.record(source)

            try:
                text = self.recognizer.recognize_google(
                    chunk_audio,
                    language=self.language
                )

                segments.append(TranscriptionSegment(
                    text=text,
                    start_time=current_time,
                    end_time=current_time + len(chunk) / 1000.0,
                    confidence=0.85,  # Google doesn't provide confidence
                    language=self.language
                ))

                current_time += len(chunk) / 1000.0

            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                logger.error(f"Google Speech Recognition error: {e}")

        return segments

    def _post_process_segments(
        self,
        segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """Post-process transcription segments"""
        processed = []

        for segment in segments:
            # Clean text
            text = segment.text.strip()

            # Fix common transcription errors
            text = self._fix_common_errors(text)

            # Detect technical terms
            technical_terms = self._detect_technical_terms(text)

            # Detect references (papers, authors, etc.)
            references = self._detect_references(text)

            processed.append(TranscriptionSegment(
                text=text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence,
                speaker=segment.speaker,
                language=segment.language,
                technical_terms=technical_terms,
                references=references
            ))

        return processed

    def _fix_common_errors(self, text: str) -> str:
        """Fix common transcription errors"""
        # Common replacements for technical terms
        replacements = {
            'neuro network': 'neural network',
            'machine earning': 'machine learning',
            'data bass': 'database',
            'all gore rhythm': 'algorithm',
            'pie thon': 'Python',
            'jay son': 'JSON',
            'sequel': 'SQL'
        }

        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)

        return text

    def _detect_technical_terms(self, text: str) -> List[str]:
        """Detect technical terms in text"""
        technical_terms = []

        # Use NLP to find noun phrases
        doc = self.nlp(text)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]

        # Check against technical vocabulary
        for phrase in noun_phrases:
            if phrase.lower() in self.technical_vocab:
                technical_terms.append(phrase)

        # Also check individual tokens
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() in self.technical_vocab:
                if token.text not in technical_terms:
                    technical_terms.append(token.text)

        return technical_terms

    def _detect_references(self, text: str) -> List[str]:
        """Detect paper/author references in text"""
        references = []

        # Pattern for author names (e.g., "Smith et al.")
        import re
        author_pattern = re.compile(r'([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)*(?:\s+et\s+al\.?)?)')
        matches = author_pattern.findall(text)
        references.extend(matches)

        # Pattern for years (likely paper references)
        year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        years = year_pattern.findall(text)

        # Combine author and year if they appear close together
        for author in matches:
            for year in years:
                if f"{author}" in text and year in text:
                    references.append(f"{author} ({year})")

        return list(set(references))

    def _load_technical_vocabulary(self, vocab_path: Optional[str]) -> set:
        """Load technical vocabulary for domain-specific recognition"""
        vocab = set()

        # Default technical terms
        default_terms = {
            'neural network', 'machine learning', 'deep learning',
            'algorithm', 'database', 'artificial intelligence',
            'convolutional', 'recurrent', 'transformer',
            'gradient descent', 'backpropagation', 'optimization',
            'hypothesis', 'methodology', 'analysis', 'synthesis',
            'correlation', 'regression', 'classification',
            'protein', 'molecule', 'chemical', 'reaction',
            'quantum', 'mechanics', 'thermodynamics'
        }

        vocab.update(default_terms)

        # Load custom vocabulary if provided
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, 'r') as f:
                custom_terms = [line.strip().lower() for line in f]
                vocab.update(custom_terms)

        return vocab

    def _identify_technical_terms(
        self,
        segments: List[TranscriptionSegment]
    ) -> List[str]:
        """Identify all unique technical terms across segments"""
        all_terms = set()

        for segment in segments:
            all_terms.update(segment.technical_terms)

        return sorted(list(all_terms))

    def _identify_unique_speakers(
        self,
        segments: List[TranscriptionSegment]
    ) -> List[str]:
        """Identify unique speakers from segments"""
        speakers = set()

        for segment in segments:
            if segment.speaker:
                speakers.add(segment.speaker)

        if not speakers:
            # If no speaker info, estimate from speech patterns
            # This is simplified - would use actual speaker diarization
            speakers = {'Speaker 1'}

        return sorted(list(speakers))

    def _extract_key_points(
        self,
        full_text: str,
        segments: List[TranscriptionSegment]
    ) -> List[str]:
        """Extract key points from transcription"""
        key_points = []

        # Split into sentences
        sentences = sent_tokenize(full_text)

        # Find sentences with high information content
        for sentence in sentences:
            # Check for key phrases
            key_phrases = [
                'important', 'key point', 'main idea', 'conclusion',
                'hypothesis', 'finding', 'result', 'demonstrate',
                'we found', 'we discovered', 'this shows'
            ]

            if any(phrase in sentence.lower() for phrase in key_phrases):
                key_points.append(sentence)

            # Check for technical term density
            doc = self.nlp(sentence)
            technical_count = sum(1 for token in doc
                                if token.text.lower() in self.technical_vocab)

            if technical_count > 3:  # High technical content
                key_points.append(sentence)

        # Limit to most relevant points
        return key_points[:10]

    def _assess_quality(self, segments: List[TranscriptionSegment]) -> float:
        """Assess transcription quality"""
        if not segments:
            return 0.0

        # Average confidence across segments
        avg_confidence = np.mean([seg.confidence for seg in segments])

        # Check for consistency
        consistency_score = 1.0
        total_duration = segments[-1].end_time if segments else 0

        # Check for gaps in transcription
        for i in range(1, len(segments)):
            gap = segments[i].start_time - segments[i-1].end_time
            if gap > 2.0:  # More than 2 seconds gap
                consistency_score *= 0.95

        # Combine scores
        quality_score = (avg_confidence * 0.7 + consistency_score * 0.3)

        return min(quality_score, 1.0)

    def export_transcript(
        self,
        result: TranscriptionResult,
        output_path: Union[str, Path],
        format: str = "srt"
    ):
        """
        Export transcription in various formats

        Args:
            result: TranscriptionResult to export
            output_path: Output file path
            format: Export format (srt, vtt, json, txt)
        """
        output_path = Path(output_path)

        if format == "srt":
            self._export_srt(result, output_path)
        elif format == "vtt":
            self._export_vtt(result, output_path)
        elif format == "json":
            self._export_json(result, output_path)
        elif format == "txt":
            self._export_txt(result, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_srt(self, result: TranscriptionResult, output_path: Path):
        """Export as SRT subtitle file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result.segments, 1):
                # Format timestamps
                start = self._format_timestamp(segment.start_time)
                end = self._format_timestamp(segment.end_time)

                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{segment.text}\n\n")

    def _export_vtt(self, result: TranscriptionResult, output_path: Path):
        """Export as WebVTT file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")

            for segment in result.segments:
                start = self._format_timestamp(segment.start_time, vtt=True)
                end = self._format_timestamp(segment.end_time, vtt=True)

                f.write(f"{start} --> {end}\n")
                f.write(f"{segment.text}\n\n")

    def _export_json(self, result: TranscriptionResult, output_path: Path):
        """Export as JSON"""
        data = {
            'full_text': result.full_text,
            'duration': result.duration,
            'content_type': result.content_type.value,
            'language': result.language,
            'speakers': result.speakers,
            'technical_vocabulary': result.technical_vocabulary,
            'key_points': result.key_points,
            'quality_score': result.quality_score,
            'metadata': result.metadata,
            'segments': [
                {
                    'text': seg.text,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'confidence': seg.confidence,
                    'speaker': seg.speaker,
                    'technical_terms': seg.technical_terms,
                    'references': seg.references
                }
                for seg in result.segments
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_txt(self, result: TranscriptionResult, output_path: Path):
        """Export as plain text"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcription of {result.metadata.get('input_file', 'Unknown')}\n")
            f.write(f"Duration: {result.duration:.2f} seconds\n")
            f.write(f"Content Type: {result.content_type.value}\n")
            f.write(f"Language: {result.language}\n")
            f.write(f"Quality Score: {result.quality_score:.2f}\n")
            f.write("\n" + "="*50 + "\n\n")

            if result.key_points:
                f.write("KEY POINTS:\n")
                for i, point in enumerate(result.key_points, 1):
                    f.write(f"{i}. {point}\n")
                f.write("\n" + "="*50 + "\n\n")

            f.write("FULL TRANSCRIPT:\n\n")

            for segment in result.segments:
                timestamp = f"[{self._format_timestamp(segment.start_time, simple=True)}]"
                speaker = f"{segment.speaker}: " if segment.speaker else ""
                f.write(f"{timestamp} {speaker}{segment.text}\n\n")

    def _format_timestamp(
        self,
        seconds: float,
        vtt: bool = False,
        simple: bool = False
    ) -> str:
        """Format timestamp for subtitle formats"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        if simple:
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
            else:
                return f"{minutes:02d}:{secs:05.2f}"

        if vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        else:
            # SRT format
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')