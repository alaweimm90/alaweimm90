"""
OCR Engine for Scientific Figures

Specialized OCR for extracting text from charts, including
axis labels, legends, and annotations.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import easyocr
from typing import List, Dict, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)


class FigureOCR:
    """
    Advanced OCR engine optimized for scientific figures
    Handles mathematical notation, subscripts, superscripts, and Greek letters
    """

    def __init__(self, use_gpu: bool = False, languages: List[str] = ['en']):
        """
        Initialize OCR engine

        Args:
            use_gpu: Use GPU acceleration for OCR
            languages: List of languages to support
        """
        self.use_gpu = use_gpu
        self.languages = languages

        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(languages, gpu=use_gpu)

        # Regex patterns for scientific notation
        self.scientific_patterns = {
            'exponent': re.compile(r'(\d+\.?\d*)[eE]([-+]?\d+)'),
            'subscript': re.compile(r'([A-Za-z]+)_(\d+)'),
            'greek': re.compile(r'(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega)', re.I),
            'units': re.compile(r'(\d+\.?\d*)\s*(nm|µm|mm|cm|m|kg|g|mg|s|ms|µs|ns|K|°C|mol|M|mM|µM)')
        }

        logger.info(f"FigureOCR initialized with languages: {languages}")

    def extract_all_text(self, image: np.ndarray) -> List[Dict]:
        """
        Extract all text from figure

        Args:
            image: Input image

        Returns:
            List of detected text with positions and confidence
        """
        results = []

        # Use EasyOCR for primary detection
        easy_results = self.reader.readtext(image)

        for bbox, text, confidence in easy_results:
            # Convert bbox to standard format
            x_min = min(point[0] for point in bbox)
            y_min = min(point[1] for point in bbox)
            x_max = max(point[0] for point in bbox)
            y_max = max(point[1] for point in bbox)

            # Post-process scientific text
            processed_text = self._process_scientific_text(text)

            results.append({
                'text': processed_text,
                'original_text': text,
                'bbox': [x_min, y_min, x_max, y_max],
                'confidence': confidence,
                'type': self._classify_text_type(processed_text, [x_min, y_min, x_max, y_max], image.shape)
            })

        # Additional Tesseract pass for comparison
        tesseract_results = self._tesseract_extraction(image)
        results = self._merge_ocr_results(results, tesseract_results)

        return results

    def extract_axis_labels(self, image: np.ndarray) -> Dict[str, str]:
        """
        Extract axis labels from figure

        Args:
            image: Input image

        Returns:
            Dictionary with x and y axis labels
        """
        height, width = image.shape[:2]

        # Define regions for axes
        x_axis_region = image[int(height * 0.8):, :]  # Bottom 20%
        y_axis_region = image[:, :int(width * 0.2)]    # Left 20%

        # Extract text from regions
        x_texts = self._extract_from_region(x_axis_region, 'horizontal')
        y_texts = self._extract_from_region(y_axis_region, 'vertical')

        # Find most likely axis labels
        x_label = self._find_axis_label(x_texts)
        y_label = self._find_axis_label(y_texts)

        return {
            'x_axis': x_label,
            'y_axis': y_label
        }

    def extract_legend(self, image: np.ndarray) -> List[Dict]:
        """
        Extract legend items from figure

        Args:
            image: Input image

        Returns:
            List of legend entries with colors
        """
        # Detect legend region
        legend_region = self._detect_legend_region(image)

        if legend_region is None:
            return []

        # Extract legend area
        x, y, w, h = legend_region
        legend_img = image[y:y+h, x:x+w]

        # Extract text from legend
        texts = self.extract_all_text(legend_img)

        # Match with color patches
        legend_items = []
        for text_item in texts:
            # Find nearby color patch
            color = self._find_associated_color(legend_img, text_item['bbox'])

            legend_items.append({
                'label': text_item['text'],
                'color': color,
                'confidence': text_item['confidence']
            })

        return legend_items

    def extract_tick_labels(self, image: np.ndarray) -> Dict[str, List[str]]:
        """
        Extract tick labels from axes

        Args:
            image: Input image

        Returns:
            Dictionary with x and y tick labels
        """
        height, width = image.shape[:2]

        # Extract from axis regions
        x_tick_region = image[int(height * 0.7):int(height * 0.9), :]
        y_tick_region = image[:, :int(width * 0.15)]

        # Extract tick labels
        x_ticks = self._extract_tick_values(x_tick_region, 'horizontal')
        y_ticks = self._extract_tick_values(y_tick_region, 'vertical')

        return {
            'x_ticks': x_ticks,
            'y_ticks': y_ticks
        }

    def _process_scientific_text(self, text: str) -> str:
        """Process and clean scientific notation in text"""
        processed = text

        # Handle scientific notation
        processed = self.scientific_patterns['exponent'].sub(r'\1×10^\2', processed)

        # Handle common OCR mistakes in scientific text
        replacements = {
            'x10': '×10',
            'e-': '×10^-',
            'E-': '×10^-',
            'um': 'µm',
            'uM': 'µM',
            'degC': '°C',
            'deg': '°'
        }

        for old, new in replacements.items():
            processed = processed.replace(old, new)

        return processed

    def _classify_text_type(self, text: str, bbox: List[int], img_shape: Tuple) -> str:
        """Classify text type based on content and position"""
        x_min, y_min, x_max, y_max = bbox
        height, width = img_shape[:2]

        # Position-based classification
        if y_max > height * 0.9:
            return 'x_axis_label' if len(text) > 5 else 'x_tick'
        elif x_max < width * 0.1:
            return 'y_axis_label' if len(text) > 5 else 'y_tick'
        elif y_min < height * 0.1:
            return 'title'

        # Content-based classification
        if self.scientific_patterns['units'].search(text):
            return 'measurement'
        elif re.match(r'^-?\d+\.?\d*$', text):
            return 'value'

        return 'annotation'

    def _tesseract_extraction(self, image: np.ndarray) -> List[Dict]:
        """Fallback OCR using Tesseract"""
        results = []

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Get detailed OCR data
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:  # Confidence threshold
                text = data['text'][i].strip()
                if text:
                    results.append({
                        'text': self._process_scientific_text(text),
                        'original_text': text,
                        'bbox': [data['left'][i], data['top'][i],
                                data['left'][i] + data['width'][i],
                                data['top'][i] + data['height'][i]],
                        'confidence': data['conf'][i] / 100.0,
                        'type': 'unknown'
                    })

        return results

    def _merge_ocr_results(self, primary: List[Dict], secondary: List[Dict]) -> List[Dict]:
        """Merge results from multiple OCR engines"""
        merged = primary.copy()

        # Add non-duplicate results from secondary
        for sec_item in secondary:
            is_duplicate = False

            for prim_item in primary:
                # Check for overlap
                iou = self._calculate_iou(sec_item['bbox'], prim_item['bbox'])
                if iou > 0.5:
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(sec_item)

        return merged

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)

        if x_max < x_min or y_max < y_min:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)

        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _extract_from_region(self, region: np.ndarray, orientation: str) -> List[str]:
        """Extract text from specific region"""
        texts = []

        if orientation == 'vertical':
            # Rotate for better OCR
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

        results = self.reader.readtext(region)

        for _, text, conf in results:
            if conf > 0.5:
                texts.append(text)

        return texts

    def _find_axis_label(self, texts: List[str]) -> str:
        """Identify most likely axis label from extracted texts"""
        # Filter out numbers and very short texts
        candidates = [t for t in texts if len(t) > 3 and not re.match(r'^[\d\.\-]+$', t)]

        if candidates:
            # Return longest text (likely the label)
            return max(candidates, key=len)

        return ""

    def _detect_legend_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect legend bounding box in figure"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Look for rectangular regions with text
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find potential legend boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Legend is usually in corner and relatively small
            if (w < image.shape[1] * 0.3 and h < image.shape[0] * 0.3 and
                w > 50 and h > 30):

                # Check if region contains text
                region = image[y:y+h, x:x+w]
                if self._contains_text(region):
                    return (x, y, w, h)

        return None

    def _contains_text(self, region: np.ndarray) -> bool:
        """Check if region contains text"""
        results = self.reader.readtext(region)
        return len(results) > 0

    def _find_associated_color(self, legend_img: np.ndarray,
                              text_bbox: List[int]) -> Optional[Tuple[int, int, int]]:
        """Find color patch associated with legend text"""
        x_min, y_min, x_max, y_max = text_bbox

        # Look to the left of text for color patch
        if x_min > 20:
            patch_region = legend_img[y_min:y_max, max(0, x_min-20):x_min]

            if patch_region.size > 0:
                # Get dominant color
                if len(patch_region.shape) == 3:
                    mean_color = np.mean(patch_region.reshape(-1, 3), axis=0)
                    return tuple(mean_color.astype(int))

        return None

    def _extract_tick_values(self, region: np.ndarray, orientation: str) -> List[str]:
        """Extract tick label values"""
        if orientation == 'vertical':
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

        texts = []
        results = self.reader.readtext(region)

        for _, text, conf in results:
            if conf > 0.5:
                # Clean and validate tick values
                cleaned = text.strip()
                if re.match(r'^-?\d+\.?\d*$', cleaned):
                    texts.append(cleaned)

        # Sort numerically if possible
        try:
            texts.sort(key=float)
        except ValueError:
            pass

        return texts

    def extract_mathematical_expressions(self, image: np.ndarray) -> List[Dict]:
        """
        Extract mathematical expressions and equations

        Args:
            image: Input image

        Returns:
            List of detected mathematical expressions
        """
        expressions = []

        # Preprocess for math OCR
        processed = self._preprocess_for_math(image)

        # Extract all text
        all_text = self.extract_all_text(processed)

        for item in all_text:
            # Check if text contains mathematical symbols
            if self._is_mathematical(item['text']):
                # Convert to LaTeX if possible
                latex = self._convert_to_latex(item['text'])

                expressions.append({
                    'raw_text': item['text'],
                    'latex': latex,
                    'bbox': item['bbox'],
                    'confidence': item['confidence']
                })

        return expressions

    def _preprocess_for_math(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for mathematical OCR"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # Denoise
        processed = cv2.medianBlur(processed, 3)

        return processed

    def _is_mathematical(self, text: str) -> bool:
        """Check if text contains mathematical notation"""
        math_symbols = ['=', '+', '-', '×', '÷', '∫', '∑', '∏', 'π',
                       'α', 'β', 'γ', 'θ', 'λ', 'μ', 'σ', 'ω',
                       '^', '_', '√', '∞']

        return any(symbol in text for symbol in math_symbols)

    def _convert_to_latex(self, text: str) -> str:
        """Convert OCR text to LaTeX notation"""
        latex = text

        # Replace common symbols
        replacements = {
            '×': r'\times',
            '÷': r'\div',
            '∫': r'\int',
            '∑': r'\sum',
            '∏': r'\prod',
            'π': r'\pi',
            'α': r'\alpha',
            'β': r'\beta',
            'γ': r'\gamma',
            'θ': r'\theta',
            'λ': r'\lambda',
            'μ': r'\mu',
            'σ': r'\sigma',
            'ω': r'\omega',
            '∞': r'\infty',
            '√': r'\sqrt'
        }

        for old, new in replacements.items():
            latex = latex.replace(old, new)

        # Handle superscripts and subscripts
        latex = re.sub(r'\^(\w+)', r'^{\1}', latex)
        latex = re.sub(r'_(\w+)', r'_{\1}', latex)

        return latex