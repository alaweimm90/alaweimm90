"""
TalAI Audio/Video Research Content Processing Module

Advanced system for extracting and analyzing research content from
audio and video sources including lectures, presentations, and lab recordings.
"""

from .transcription_engine import TranscriptionEngine
from .speaker_diarization import SpeakerDiarization
from .slide_extractor import SlideExtractor
from .video_analyzer import VideoAnalyzer
from .audio_processor import AudioProcessor
from .timestamp_generator import TimestampGenerator
from .content_segmenter import ContentSegmenter
from .platform_integrator import PlatformIntegrator

__all__ = [
    'TranscriptionEngine',
    'SpeakerDiarization',
    'SlideExtractor',
    'VideoAnalyzer',
    'AudioProcessor',
    'TimestampGenerator',
    'ContentSegmenter',
    'PlatformIntegrator'
]

__version__ = '1.0.0'