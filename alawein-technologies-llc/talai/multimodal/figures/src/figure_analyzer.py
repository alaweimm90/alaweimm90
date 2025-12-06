"""
Core Figure Analysis Engine for Scientific Publications

Integrates with vision-language models to understand and extract
structured data from scientific figures.
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FigureType(Enum):
    """Enumeration of scientific figure types"""
    SCATTER_PLOT = "scatter_plot"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    NETWORK_DIAGRAM = "network_diagram"
    FLOWCHART = "flowchart"
    MICROSCOPY = "microscopy"
    GEL_ELECTROPHORESIS = "gel_electrophoresis"
    MOLECULAR_STRUCTURE = "molecular_structure"
    SCHEMATIC = "schematic"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


@dataclass
class FigureMetadata:
    """Metadata extracted from scientific figures"""
    figure_id: str
    source_paper: Optional[str] = None
    page_number: Optional[int] = None
    figure_number: Optional[str] = None
    caption: Optional[str] = None
    width: int = 0
    height: int = 0
    dpi: int = 72
    color_mode: str = "RGB"
    file_format: str = "PNG"
    extraction_confidence: float = 0.0
    processing_time: float = 0.0
    references: List[str] = field(default_factory=list)
    subfigures: List[Dict] = field(default_factory=list)


@dataclass
class ExtractedData:
    """Structured data extracted from figures"""
    data_points: List[Dict[str, Any]]
    axes_labels: Dict[str, str]
    legend_items: List[str]
    error_bars: Optional[Dict[str, List[float]]]
    statistics: Dict[str, Any]
    raw_values: Optional[np.ndarray]
    confidence_scores: Dict[str, float]


class FigureAnalyzer:
    """
    Main figure analysis engine with vision-language model integration
    Supports multiple backend models and caching for efficient processing
    """

    def __init__(
        self,
        model_backend: str = "claude",
        cache_dir: str = "/tmp/figure_cache",
        gpu_enabled: bool = True,
        batch_size: int = 8,
        max_workers: int = 4
    ):
        """
        Initialize the Figure Analyzer

        Args:
            model_backend: Vision model backend ('claude', 'gpt4v', 'local')
            cache_dir: Directory for caching processed figures
            gpu_enabled: Whether to use GPU acceleration
            batch_size: Batch size for processing multiple figures
            max_workers: Number of parallel workers
        """
        self.model_backend = model_backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_enabled = gpu_enabled and torch.cuda.is_available()
        self.batch_size = batch_size
        self.max_workers = max_workers

        # Initialize preprocessing pipeline
        self.preprocessor = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Initialize model connections
        self._initialize_models()

        # Cache for processed figures
        self.figure_cache = {}

        logger.info(f"FigureAnalyzer initialized with backend: {model_backend}")

    def _initialize_models(self):
        """Initialize vision-language model connections"""
        if self.model_backend == "claude":
            # Claude 3 Vision API setup
            self.vision_api = self._setup_claude_vision()
        elif self.model_backend == "gpt4v":
            # GPT-4V API setup
            self.vision_api = self._setup_gpt4_vision()
        else:
            # Local vision model (e.g., CLIP, LayoutLM)
            self.vision_model = self._load_local_model()

    def _setup_claude_vision(self):
        """Setup Claude 3 Vision API connection"""
        # Placeholder for Claude API setup
        return {
            "api_key": os.getenv("CLAUDE_API_KEY"),
            "endpoint": "https://api.anthropic.com/v1/vision"
        }

    def _setup_gpt4_vision(self):
        """Setup GPT-4 Vision API connection"""
        # Placeholder for GPT-4V API setup
        return {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "endpoint": "https://api.openai.com/v1/chat/completions"
        }

    def _load_local_model(self):
        """Load local vision model for figure understanding"""
        if self.gpu_enabled:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Example: Load a pretrained vision model
        # This could be CLIP, LayoutLMv3, or custom trained model
        model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        model.to(device)
        model.eval()
        return model

    def analyze_figure(
        self,
        image_path: Union[str, Path],
        paper_context: Optional[Dict] = None,
        extract_data: bool = True,
        generate_accessibility: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a scientific figure

        Args:
            image_path: Path to the figure image
            paper_context: Context from the paper (title, abstract, etc.)
            extract_data: Whether to extract numerical data
            generate_accessibility: Generate accessibility descriptions

        Returns:
            Dictionary containing analysis results
        """
        image_path = Path(image_path)

        # Check cache
        cache_key = self._get_cache_key(image_path)
        if cache_key in self.figure_cache:
            logger.info(f"Using cached results for {image_path}")
            return self.figure_cache[cache_key]

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        metadata = self._extract_metadata(image, image_path)

        # Detect figure type
        figure_type = self._detect_figure_type(image)

        # Extract components
        components = self._extract_components(image, figure_type)

        # Extract data if requested
        extracted_data = None
        if extract_data:
            extracted_data = self._extract_data(image, figure_type, components)

        # Generate accessibility description
        accessibility_desc = None
        if generate_accessibility:
            accessibility_desc = self._generate_accessibility_description(
                image, figure_type, extracted_data, paper_context
            )

        # Cross-reference with paper text
        cross_references = self._find_cross_references(metadata, paper_context)

        # Compile results
        results = {
            'metadata': metadata.__dict__,
            'figure_type': figure_type.value,
            'components': components,
            'extracted_data': extracted_data.__dict__ if extracted_data else None,
            'accessibility_description': accessibility_desc,
            'cross_references': cross_references,
            'quality_metrics': self._assess_figure_quality(image)
        }

        # Cache results
        self.figure_cache[cache_key] = results

        return results

    def _get_cache_key(self, image_path: Path) -> str:
        """Generate cache key for figure"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _extract_metadata(self, image: Image.Image, path: Path) -> FigureMetadata:
        """Extract metadata from figure"""
        return FigureMetadata(
            figure_id=path.stem,
            width=image.width,
            height=image.height,
            dpi=image.info.get('dpi', (72, 72))[0],
            color_mode=image.mode,
            file_format=image.format or path.suffix[1:].upper()
        )

    def _detect_figure_type(self, image: Image.Image) -> FigureType:
        """
        Detect the type of scientific figure using vision models
        """
        # Convert to tensor for model input
        img_tensor = self.preprocessor(image).unsqueeze(0)

        if self.model_backend == "claude":
            # Use Claude Vision API
            figure_type = self._detect_with_claude(image)
        elif self.model_backend == "gpt4v":
            # Use GPT-4V API
            figure_type = self._detect_with_gpt4v(image)
        else:
            # Use local model
            figure_type = self._detect_with_local_model(img_tensor)

        return figure_type

    def _detect_with_claude(self, image: Image.Image) -> FigureType:
        """Detect figure type using Claude Vision"""
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        prompt = """Analyze this scientific figure and classify it into one of these types:
        - scatter_plot: Points plotted on x-y axes
        - line_chart: Continuous lines showing trends
        - bar_chart: Rectangular bars comparing values
        - histogram: Distribution of data
        - heatmap: Color-coded matrix
        - network_diagram: Nodes and edges
        - flowchart: Process flow diagram
        - microscopy: Microscope image
        - gel_electrophoresis: Gel bands
        - molecular_structure: Chemical/protein structure
        - schematic: Technical diagram
        - composite: Multiple subfigures

        Return only the type name."""

        # Simulated API call - would use actual API in production
        # response = claude_vision_api.analyze(img_str, prompt)

        # For demonstration, use pattern matching
        return self._fallback_detection(image)

    def _detect_with_gpt4v(self, image: Image.Image) -> FigureType:
        """Detect figure type using GPT-4 Vision"""
        # Similar to Claude implementation
        return self._fallback_detection(image)

    def _detect_with_local_model(self, img_tensor: torch.Tensor) -> FigureType:
        """Detect figure type using local vision model"""
        with torch.no_grad():
            outputs = self.vision_model(img_tensor)

        # Process model outputs to determine figure type
        # This would involve mapping model predictions to figure types
        return FigureType.UNKNOWN

    def _fallback_detection(self, image: Image.Image) -> FigureType:
        """Fallback detection using traditional CV methods"""
        # Convert to numpy array
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines (for charts)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

        # Detect circles (for scatter plots)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=0, maxRadius=0)

        # Simple heuristics
        if lines is not None and len(lines) > 10:
            return FigureType.LINE_CHART
        elif circles is not None and len(circles[0]) > 20:
            return FigureType.SCATTER_PLOT
        else:
            return FigureType.UNKNOWN

    def _extract_components(self, image: Image.Image, figure_type: FigureType) -> Dict:
        """Extract figure components (axes, legend, title, etc.)"""
        components = {
            'has_axes': False,
            'has_legend': False,
            'has_title': False,
            'has_colorbar': False,
            'regions': []
        }

        # Use object detection to identify components
        img_array = np.array(image)

        # Simple region detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:10]:  # Limit to top 10 regions
            x, y, w, h = cv2.boundingRect(contour)
            components['regions'].append({
                'bbox': [x, y, w, h],
                'area': w * h,
                'aspect_ratio': w / h if h > 0 else 0
            })

        return components

    def _extract_data(
        self,
        image: Image.Image,
        figure_type: FigureType,
        components: Dict
    ) -> ExtractedData:
        """Extract numerical data from figure"""
        data_points = []
        axes_labels = {}
        legend_items = []
        error_bars = None
        statistics = {}

        if figure_type == FigureType.SCATTER_PLOT:
            data_points = self._extract_scatter_data(image)
        elif figure_type == FigureType.LINE_CHART:
            data_points = self._extract_line_data(image)
        elif figure_type == FigureType.BAR_CHART:
            data_points = self._extract_bar_data(image)

        return ExtractedData(
            data_points=data_points,
            axes_labels=axes_labels,
            legend_items=legend_items,
            error_bars=error_bars,
            statistics=statistics,
            raw_values=None,
            confidence_scores={'overall': 0.85}
        )

    def _extract_scatter_data(self, image: Image.Image) -> List[Dict]:
        """Extract data points from scatter plot"""
        # Placeholder implementation
        return [{'x': i, 'y': np.random.randn()} for i in range(10)]

    def _extract_line_data(self, image: Image.Image) -> List[Dict]:
        """Extract line data points"""
        # Placeholder implementation
        return [{'x': i, 'y': np.sin(i/10)} for i in range(100)]

    def _extract_bar_data(self, image: Image.Image) -> List[Dict]:
        """Extract bar chart values"""
        # Placeholder implementation
        return [{'category': f'Cat{i}', 'value': np.random.randint(10, 100)}
                for i in range(5)]

    def _generate_accessibility_description(
        self,
        image: Image.Image,
        figure_type: FigureType,
        data: Optional[ExtractedData],
        context: Optional[Dict]
    ) -> str:
        """Generate accessibility description for figure"""
        desc = f"This is a {figure_type.value.replace('_', ' ')}. "

        if data and data.data_points:
            desc += f"It contains {len(data.data_points)} data points. "

        if context and 'caption' in context:
            desc += f"Caption: {context['caption']}. "

        desc += "The figure shows the relationship between variables in the study."

        return desc

    def _find_cross_references(
        self,
        metadata: FigureMetadata,
        context: Optional[Dict]
    ) -> List[str]:
        """Find cross-references to figure in paper text"""
        references = []

        if context and 'full_text' in context:
            # Search for figure references
            import re
            pattern = rf"[Ff]ig\.?\s*{metadata.figure_number}|[Ff]igure\s*{metadata.figure_number}"
            matches = re.findall(pattern, context['full_text'])
            references = list(set(matches))

        return references

    def _assess_figure_quality(self, image: Image.Image) -> Dict[str, float]:
        """Assess figure quality metrics"""
        img_array = np.array(image)

        # Calculate quality metrics
        metrics = {
            'resolution_score': min(image.width * image.height / 1000000, 1.0),
            'contrast': np.std(img_array) / 255.0,
            'sharpness': self._calculate_sharpness(img_array),
            'noise_level': self._estimate_noise(img_array)
        }

        metrics['overall_quality'] = np.mean(list(metrics.values()))

        return metrics

    def _calculate_sharpness(self, img: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian) / 10000  # Normalize

    def _estimate_noise(self, img: np.ndarray) -> float:
        """Estimate noise level in image"""
        # Use difference between original and denoised
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        noise = np.mean(np.abs(img.astype(float) - denoised.astype(float)))
        return 1.0 - min(noise / 50, 1.0)  # Convert to quality score

    def batch_analyze(
        self,
        image_paths: List[Union[str, Path]],
        paper_context: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple figures in parallel

        Args:
            image_paths: List of paths to figure images
            paper_context: Shared context from paper

        Returns:
            List of analysis results
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.analyze_figure, path, paper_context): path
                for path in image_paths
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {path}: {e}")
                    results.append({'error': str(e), 'path': str(path)})

        return results

    def export_results(self, results: Dict, output_format: str = "json") -> str:
        """Export analysis results in various formats"""
        if output_format == "json":
            return json.dumps(results, indent=2, default=str)
        elif output_format == "csv":
            # Convert to CSV format
            import csv
            from io import StringIO

            output = StringIO()
            writer = csv.writer(output)

            # Flatten nested structure for CSV
            if 'extracted_data' in results and results['extracted_data']:
                writer.writerow(['x', 'y', 'type'])
                for point in results['extracted_data'].get('data_points', []):
                    writer.writerow([point.get('x'), point.get('y'), results['figure_type']])

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {output_format}")