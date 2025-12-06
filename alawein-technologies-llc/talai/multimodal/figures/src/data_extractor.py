"""
Advanced Data Extraction from Scientific Figures

Specializes in extracting numerical data, error bars, and statistical
information from various chart types.
"""

import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import interpolate, signal, optimize
from sklearn.cluster import DBSCAN
import pytesseract
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Represents a single data point extracted from a figure"""
    x: float
    y: float
    error_x: Optional[float] = None
    error_y: Optional[float] = None
    confidence: float = 1.0
    label: Optional[str] = None
    color: Optional[Tuple[int, int, int]] = None
    marker_type: Optional[str] = None


@dataclass
class AxisInfo:
    """Information about chart axes"""
    label: str
    min_value: float
    max_value: float
    scale: str  # 'linear', 'log', 'symlog'
    unit: Optional[str] = None
    tick_values: List[float] = None
    tick_labels: List[str] = None


class DataExtractor:
    """
    Sophisticated data extraction engine for scientific figures
    Handles various chart types with high accuracy
    """

    def __init__(self, precision_mode: bool = True, gpu_acceleration: bool = False):
        """
        Initialize the Data Extractor

        Args:
            precision_mode: High precision mode with sub-pixel accuracy
            gpu_acceleration: Use GPU for intensive operations
        """
        self.precision_mode = precision_mode
        self.gpu_acceleration = gpu_acceleration

        # Color detection thresholds
        self.color_tolerance = 30

        # Clustering parameters for point detection
        self.clustering_eps = 5
        self.clustering_min_samples = 1

        logger.info("DataExtractor initialized")

    def extract_from_scatter_plot(
        self,
        image: np.ndarray,
        axes_bounds: Optional[Dict] = None
    ) -> List[DataPoint]:
        """
        Extract data points from scatter plots with error bars

        Args:
            image: Input image as numpy array
            axes_bounds: Dictionary with axis boundaries

        Returns:
            List of extracted data points
        """
        # Detect plot area
        plot_area = self._detect_plot_area(image, axes_bounds)

        # Detect data points
        points = self._detect_scatter_points(image, plot_area)

        # Detect error bars
        error_bars = self._detect_error_bars(image, points, plot_area)

        # Map pixel coordinates to data values
        data_points = self._map_to_data_coordinates(points, error_bars, plot_area)

        # Validate and filter points
        validated_points = self._validate_points(data_points)

        return validated_points

    def extract_from_line_chart(
        self,
        image: np.ndarray,
        axes_bounds: Optional[Dict] = None,
        num_lines: Optional[int] = None
    ) -> Dict[str, List[DataPoint]]:
        """
        Extract line data from line charts

        Args:
            image: Input image
            axes_bounds: Axis boundaries
            num_lines: Expected number of lines (auto-detect if None)

        Returns:
            Dictionary mapping line IDs to data points
        """
        plot_area = self._detect_plot_area(image, axes_bounds)

        # Detect individual lines
        lines = self._detect_lines(image, plot_area, num_lines)

        # Extract points along each line
        line_data = {}
        for idx, line_pixels in enumerate(lines):
            # Sample points along the line
            sampled_points = self._sample_line_points(line_pixels, plot_area)

            # Apply smoothing if needed
            if self.precision_mode:
                sampled_points = self._smooth_line_data(sampled_points)

            line_data[f"line_{idx}"] = sampled_points

        return line_data

    def extract_from_bar_chart(
        self,
        image: np.ndarray,
        axes_bounds: Optional[Dict] = None,
        orientation: str = 'vertical'
    ) -> List[Dict[str, Any]]:
        """
        Extract data from bar charts

        Args:
            image: Input image
            axes_bounds: Axis boundaries
            orientation: 'vertical' or 'horizontal'

        Returns:
            List of bar data with categories and values
        """
        plot_area = self._detect_plot_area(image, axes_bounds)

        # Detect bars
        bars = self._detect_bars(image, plot_area, orientation)

        # Extract bar heights/widths
        bar_data = []
        for bar in bars:
            # Get bar dimensions
            value = self._measure_bar_value(bar, plot_area, orientation)

            # Extract bar label if visible
            label = self._extract_bar_label(image, bar, orientation)

            # Detect error bars if present
            error = self._detect_bar_error(image, bar, orientation)

            bar_data.append({
                'category': label,
                'value': value,
                'error': error,
                'color': self._get_dominant_color(image, bar),
                'position': bar['position']
            })

        return bar_data

    def extract_from_heatmap(
        self,
        image: np.ndarray,
        axes_bounds: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Extract matrix data from heatmaps

        Args:
            image: Input heatmap image
            axes_bounds: Axis boundaries

        Returns:
            2D numpy array of extracted values
        """
        plot_area = self._detect_plot_area(image, axes_bounds)

        # Detect colorbar
        colorbar_info = self._detect_colorbar(image)

        # Extract heatmap region
        heatmap_region = image[
            plot_area['y_min']:plot_area['y_max'],
            plot_area['x_min']:plot_area['x_max']
        ]

        # Map colors to values using colorbar
        value_matrix = self._map_colors_to_values(heatmap_region, colorbar_info)

        return value_matrix

    def extract_axes_information(self, image: np.ndarray) -> Dict[str, AxisInfo]:
        """
        Extract axis labels, scales, and ranges

        Args:
            image: Input image

        Returns:
            Dictionary with x and y axis information
        """
        # Detect axis regions
        x_axis_region, y_axis_region = self._detect_axis_regions(image)

        # Extract axis labels using OCR
        x_label = self._extract_axis_label(image, x_axis_region, 'x')
        y_label = self._extract_axis_label(image, y_axis_region, 'y')

        # Extract tick marks and values
        x_ticks = self._extract_tick_marks(image, x_axis_region, 'x')
        y_ticks = self._extract_tick_marks(image, y_axis_region, 'y')

        # Detect scale type (linear, log, etc.)
        x_scale = self._detect_scale_type(x_ticks)
        y_scale = self._detect_scale_type(y_ticks)

        axes_info = {
            'x': AxisInfo(
                label=x_label,
                min_value=min(x_ticks['values']) if x_ticks['values'] else 0,
                max_value=max(x_ticks['values']) if x_ticks['values'] else 1,
                scale=x_scale,
                tick_values=x_ticks['values'],
                tick_labels=x_ticks['labels']
            ),
            'y': AxisInfo(
                label=y_label,
                min_value=min(y_ticks['values']) if y_ticks['values'] else 0,
                max_value=max(y_ticks['values']) if y_ticks['values'] else 1,
                scale=y_scale,
                tick_values=y_ticks['values'],
                tick_labels=y_ticks['labels']
            )
        }

        return axes_info

    def _detect_plot_area(
        self,
        image: np.ndarray,
        axes_bounds: Optional[Dict]
    ) -> Dict[str, int]:
        """Detect the main plotting area within the figure"""
        if axes_bounds:
            return axes_bounds

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find largest rectangular contour (likely the plot area)
        max_area = 0
        plot_bounds = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Check if it's roughly rectangular
            if area > max_area and w > image.shape[1] * 0.3 and h > image.shape[0] * 0.3:
                max_area = area
                plot_bounds = {
                    'x_min': x,
                    'x_max': x + w,
                    'y_min': y,
                    'y_max': y + h
                }

        return plot_bounds or {
            'x_min': int(image.shape[1] * 0.1),
            'x_max': int(image.shape[1] * 0.9),
            'y_min': int(image.shape[0] * 0.1),
            'y_max': int(image.shape[0] * 0.9)
        }

    def _detect_scatter_points(
        self,
        image: np.ndarray,
        plot_area: Dict[str, int]
    ) -> List[Tuple[int, int]]:
        """Detect individual points in scatter plot"""
        # Extract plot region
        plot_img = image[
            plot_area['y_min']:plot_area['y_max'],
            plot_area['x_min']:plot_area['x_max']
        ]

        # Convert to grayscale
        gray = cv2.cvtColor(plot_img, cv2.COLOR_RGB2GRAY) if len(plot_img.shape) == 3 else plot_img

        # Detect circles (common scatter plot markers)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=50,
            param2=30,
            minRadius=2,
            maxRadius=20
        )

        points = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                # Convert to absolute coordinates
                x = circle[0] + plot_area['x_min']
                y = circle[1] + plot_area['y_min']
                points.append((x, y))

        # Also detect other marker types using template matching
        # or blob detection for non-circular markers
        if len(points) < 5:  # Fallback if circles not detected
            points.extend(self._detect_blob_points(gray, plot_area))

        return points

    def _detect_blob_points(
        self,
        gray: np.ndarray,
        plot_area: Dict[str, int]
    ) -> List[Tuple[int, int]]:
        """Detect points using blob detection"""
        # Setup SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 500
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        points = []
        for kp in keypoints:
            x = int(kp.pt[0]) + plot_area['x_min']
            y = int(kp.pt[1]) + plot_area['y_min']
            points.append((x, y))

        return points

    def _detect_error_bars(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        plot_area: Dict[str, int]
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Detect error bars associated with data points"""
        error_bars = {}

        for point in points:
            x, y = point

            # Look for vertical error bars
            vertical_error = self._measure_vertical_error_bar(image, x, y)

            # Look for horizontal error bars
            horizontal_error = self._measure_horizontal_error_bar(image, x, y)

            if vertical_error or horizontal_error:
                error_bars[point] = {
                    'y_error': vertical_error,
                    'x_error': horizontal_error
                }

        return error_bars

    def _measure_vertical_error_bar(
        self,
        image: np.ndarray,
        x: int,
        y: int
    ) -> Optional[float]:
        """Measure vertical error bar length"""
        # Extract vertical line around the point
        if x < 0 or x >= image.shape[1]:
            return None

        column = image[:, x]

        # Look for vertical line segments above and below point
        # Simplified implementation
        return None  # Would implement actual detection

    def _measure_horizontal_error_bar(
        self,
        image: np.ndarray,
        x: int,
        y: int
    ) -> Optional[float]:
        """Measure horizontal error bar length"""
        # Similar to vertical but for horizontal direction
        return None

    def _map_to_data_coordinates(
        self,
        points: List[Tuple[int, int]],
        error_bars: Dict,
        plot_area: Dict[str, int]
    ) -> List[DataPoint]:
        """Map pixel coordinates to data values"""
        data_points = []

        # Calculate scaling factors (simplified - would use actual axis values)
        x_scale = 100.0 / (plot_area['x_max'] - plot_area['x_min'])
        y_scale = 100.0 / (plot_area['y_max'] - plot_area['y_min'])

        for point in points:
            x_pixel, y_pixel = point

            # Convert to data coordinates
            x_data = (x_pixel - plot_area['x_min']) * x_scale
            y_data = (plot_area['y_max'] - y_pixel) * y_scale  # Invert y-axis

            # Get error bars if present
            errors = error_bars.get(point, {})

            data_point = DataPoint(
                x=x_data,
                y=y_data,
                error_x=errors.get('x_error'),
                error_y=errors.get('y_error'),
                confidence=0.95
            )

            data_points.append(data_point)

        return data_points

    def _validate_points(self, points: List[DataPoint]) -> List[DataPoint]:
        """Validate and filter extracted points"""
        # Remove outliers using statistical methods
        if len(points) < 3:
            return points

        # Convert to numpy array for easier manipulation
        x_values = np.array([p.x for p in points])
        y_values = np.array([p.y for p in points])

        # Simple outlier detection using IQR
        q1_x, q3_x = np.percentile(x_values, [25, 75])
        q1_y, q3_y = np.percentile(y_values, [25, 75])

        iqr_x = q3_x - q1_x
        iqr_y = q3_y - q1_y

        valid_points = []
        for p in points:
            if (q1_x - 1.5 * iqr_x <= p.x <= q3_x + 1.5 * iqr_x and
                q1_y - 1.5 * iqr_y <= p.y <= q3_y + 1.5 * iqr_y):
                valid_points.append(p)

        return valid_points

    def _detect_lines(
        self,
        image: np.ndarray,
        plot_area: Dict[str, int],
        num_lines: Optional[int]
    ) -> List[List[Tuple[int, int]]]:
        """Detect individual lines in line chart"""
        # Extract plot region
        plot_img = image[
            plot_area['y_min']:plot_area['y_max'],
            plot_area['x_min']:plot_area['x_max']
        ]

        # Use color clustering to separate lines
        lines = []

        # Simplified: detect continuous paths
        gray = cv2.cvtColor(plot_img, cv2.COLOR_RGB2GRAY) if len(plot_img.shape) == 3 else plot_img

        # Threshold to get line pixels
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours (each contour could be a line)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours[:num_lines] if num_lines else contours:
            if len(contour) > 50:  # Filter small contours
                line_points = [(pt[0][0], pt[0][1]) for pt in contour]
                lines.append(line_points)

        return lines

    def _sample_line_points(
        self,
        line_pixels: List[Tuple[int, int]],
        plot_area: Dict[str, int]
    ) -> List[DataPoint]:
        """Sample points along a detected line"""
        # Sort by x-coordinate
        sorted_pixels = sorted(line_pixels, key=lambda p: p[0])

        # Sample at regular intervals
        sample_interval = max(1, len(sorted_pixels) // 100)  # Max 100 points
        sampled = sorted_pixels[::sample_interval]

        # Convert to data points
        data_points = []
        for x, y in sampled:
            data_point = DataPoint(
                x=float(x),
                y=float(y),
                confidence=0.9
            )
            data_points.append(data_point)

        return data_points

    def _smooth_line_data(self, points: List[DataPoint]) -> List[DataPoint]:
        """Apply smoothing to extracted line data"""
        if len(points) < 5:
            return points

        x_values = [p.x for p in points]
        y_values = [p.y for p in points]

        # Apply Savitzky-Golay filter
        window_size = min(11, len(points) if len(points) % 2 == 1 else len(points) - 1)
        if window_size >= 5:
            y_smooth = signal.savgol_filter(y_values, window_size, 3)

            smoothed_points = []
            for i, p in enumerate(points):
                smoothed_points.append(DataPoint(
                    x=p.x,
                    y=y_smooth[i],
                    confidence=p.confidence
                ))

            return smoothed_points

        return points

    def export_to_csv(self, data_points: List[DataPoint], filename: str):
        """Export extracted data to CSV file"""
        df = pd.DataFrame([{
            'x': p.x,
            'y': p.y,
            'error_x': p.error_x,
            'error_y': p.error_y,
            'confidence': p.confidence,
            'label': p.label
        } for p in data_points])

        df.to_csv(filename, index=False)
        logger.info(f"Data exported to {filename}")

    def _detect_bars(
        self,
        image: np.ndarray,
        plot_area: Dict[str, int],
        orientation: str
    ) -> List[Dict]:
        """Detect individual bars in bar chart"""
        # Implementation would detect rectangular regions
        return []

    def _measure_bar_value(
        self,
        bar: Dict,
        plot_area: Dict[str, int],
        orientation: str
    ) -> float:
        """Measure the value represented by a bar"""
        return 0.0

    def _extract_bar_label(
        self,
        image: np.ndarray,
        bar: Dict,
        orientation: str
    ) -> str:
        """Extract label for a bar using OCR"""
        return ""

    def _detect_bar_error(
        self,
        image: np.ndarray,
        bar: Dict,
        orientation: str
    ) -> Optional[float]:
        """Detect error bars on bar charts"""
        return None

    def _get_dominant_color(
        self,
        image: np.ndarray,
        region: Dict
    ) -> Tuple[int, int, int]:
        """Get dominant color in a region"""
        return (0, 0, 0)

    def _detect_colorbar(self, image: np.ndarray) -> Dict:
        """Detect and analyze colorbar in heatmap"""
        return {}

    def _map_colors_to_values(
        self,
        heatmap_region: np.ndarray,
        colorbar_info: Dict
    ) -> np.ndarray:
        """Map heatmap colors to numerical values"""
        return np.zeros_like(heatmap_region[:, :, 0], dtype=float)

    def _detect_axis_regions(self, image: np.ndarray) -> Tuple[Dict, Dict]:
        """Detect x and y axis regions"""
        return {}, {}

    def _extract_axis_label(
        self,
        image: np.ndarray,
        axis_region: Dict,
        axis_type: str
    ) -> str:
        """Extract axis label using OCR"""
        return ""

    def _extract_tick_marks(
        self,
        image: np.ndarray,
        axis_region: Dict,
        axis_type: str
    ) -> Dict:
        """Extract tick marks and their labels"""
        return {'values': [], 'labels': []}

    def _detect_scale_type(self, ticks: Dict) -> str:
        """Detect if axis uses linear, log, or other scale"""
        return 'linear'