"""
Chart Type Detection Module

Uses deep learning and traditional CV methods to accurately
identify chart types in scientific figures.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import logging

logger = logging.getLogger(__name__)


class ChartTypeDetector:
    """
    Advanced chart type detection using ensemble methods
    Combines CNN-based classification with rule-based heuristics
    """

    def __init__(self, model_path: Optional[str] = None, use_ensemble: bool = True):
        """
        Initialize chart type detector

        Args:
            model_path: Path to pre-trained model weights
            use_ensemble: Use ensemble of methods for better accuracy
        """
        self.use_ensemble = use_ensemble
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize deep learning model
        self.cnn_model = self._load_cnn_model(model_path)

        # Initialize preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Chart type labels
        self.chart_types = [
            'scatter_plot', 'line_chart', 'bar_chart', 'histogram',
            'heatmap', 'pie_chart', 'box_plot', 'violin_plot',
            'network_diagram', 'flowchart', 'contour_plot', '3d_plot'
        ]

        logger.info(f"ChartTypeDetector initialized on {self.device}")

    def _load_cnn_model(self, model_path: Optional[str]) -> nn.Module:
        """Load pre-trained CNN model for chart classification"""
        # Use ResNet50 as backbone
        model = models.resnet50(pretrained=True)

        # Modify final layer for chart classification
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.chart_types))

        if model_path:
            # Load pre-trained weights if available
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model weights: {e}")

        model.to(self.device)
        model.eval()

        return model

    def detect_chart_type(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect chart type with confidence scores

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary of chart types and confidence scores
        """
        results = {}

        if self.use_ensemble:
            # CNN-based prediction
            cnn_pred = self._cnn_prediction(image)

            # Rule-based prediction
            rule_pred = self._rule_based_prediction(image)

            # Feature-based prediction
            feature_pred = self._feature_based_prediction(image)

            # Ensemble predictions
            for chart_type in self.chart_types:
                scores = []
                if chart_type in cnn_pred:
                    scores.append(cnn_pred[chart_type] * 0.5)
                if chart_type in rule_pred:
                    scores.append(rule_pred[chart_type] * 0.3)
                if chart_type in feature_pred:
                    scores.append(feature_pred[chart_type] * 0.2)

                if scores:
                    results[chart_type] = np.mean(scores)
        else:
            results = self._cnn_prediction(image)

        # Sort by confidence
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

        return results

    def _cnn_prediction(self, image: np.ndarray) -> Dict[str, float]:
        """Get chart type prediction from CNN model"""
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.cnn_model(img_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

        # Create result dictionary
        predictions = {}
        for idx, prob in enumerate(probabilities):
            if idx < len(self.chart_types):
                predictions[self.chart_types[idx]] = float(prob)

        return predictions

    def _rule_based_prediction(self, image: np.ndarray) -> Dict[str, float]:
        """Rule-based chart type detection using CV techniques"""
        predictions = {}

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Detect features
        has_grid = self._detect_grid(gray)
        has_bars = self._detect_bars(gray)
        has_points = self._detect_points(gray)
        has_lines = self._detect_continuous_lines(gray)
        has_pie_segments = self._detect_pie_segments(gray)
        color_distribution = self._analyze_color_distribution(image)

        # Apply rules
        if has_points and not has_lines:
            predictions['scatter_plot'] = 0.8
        if has_lines and not has_bars:
            predictions['line_chart'] = 0.8
        if has_bars:
            predictions['bar_chart'] = 0.9
            predictions['histogram'] = 0.5
        if has_pie_segments:
            predictions['pie_chart'] = 0.9
        if color_distribution > 0.7:  # High color variation
            predictions['heatmap'] = 0.7
            predictions['contour_plot'] = 0.5

        return predictions

    def _feature_based_prediction(self, image: np.ndarray) -> Dict[str, float]:
        """Feature extraction based prediction"""
        features = self._extract_visual_features(image)
        predictions = {}

        # Simple feature matching (would be ML model in production)
        if features['aspect_ratio'] > 2:
            predictions['bar_chart'] = 0.6
        if features['edge_density'] > 0.3:
            predictions['network_diagram'] = 0.5
        if features['circular_features'] > 5:
            predictions['scatter_plot'] = 0.6
        if features['rectangular_features'] > 10:
            predictions['bar_chart'] = 0.7

        return predictions

    def _detect_grid(self, gray: np.ndarray) -> bool:
        """Detect if image has a grid pattern"""
        # Detect horizontal and vertical lines
        edges = cv2.Canny(gray, 50, 150)

        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                                minLineLength=100, maxLineGap=10)

        if lines is not None:
            horizontal_lines = 0
            vertical_lines = 0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                if angle < 10 or angle > 170:
                    horizontal_lines += 1
                elif 80 < angle < 100:
                    vertical_lines += 1

            return horizontal_lines > 2 and vertical_lines > 2

        return False

    def _detect_bars(self, gray: np.ndarray) -> bool:
        """Detect bar-like structures"""
        # Find contours
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangular_contours = 0
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if rectangular (4 vertices)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Typical bar aspect ratios
                if 0.2 < aspect_ratio < 5:
                    rectangular_contours += 1

        return rectangular_contours > 3

    def _detect_points(self, gray: np.ndarray) -> bool:
        """Detect point-like structures (for scatter plots)"""
        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30,
                                  minRadius=2, maxRadius=20)

        # Also detect blobs
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 500

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        num_points = 0
        if circles is not None:
            num_points += len(circles[0])
        num_points += len(keypoints)

        return num_points > 10

    def _detect_continuous_lines(self, gray: np.ndarray) -> bool:
        """Detect continuous lines (for line charts)"""
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Look for long, continuous contours
        long_contours = 0
        for contour in contours:
            if len(contour) > 100:  # Long contour
                # Check if it's roughly continuous (not too jagged)
                perimeter = cv2.arcLength(contour, False)
                if perimeter > gray.shape[1] * 0.5:  # At least half the width
                    long_contours += 1

        return long_contours > 0

    def _detect_pie_segments(self, gray: np.ndarray) -> bool:
        """Detect pie chart segments"""
        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0],
                                  param1=50, param2=30,
                                  minRadius=int(gray.shape[0] * 0.2),
                                  maxRadius=int(gray.shape[0] * 0.45))

        if circles is not None and len(circles[0]) > 0:
            # Check for radial lines from center
            center = (int(circles[0][0][0]), int(circles[0][0][1]))
            edges = cv2.Canny(gray, 50, 150)

            # Sample radial lines
            radial_lines = 0
            for angle in range(0, 360, 10):
                rad = np.radians(angle)
                x2 = int(center[0] + circles[0][0][2] * np.cos(rad))
                y2 = int(center[1] + circles[0][0][2] * np.sin(rad))

                # Check if line exists
                line_pixels = self._sample_line(edges, center, (x2, y2))
                if np.sum(line_pixels) > len(line_pixels) * 0.3:
                    radial_lines += 1

            return radial_lines > 5

        return False

    def _sample_line(self, image: np.ndarray,
                    pt1: Tuple[int, int],
                    pt2: Tuple[int, int]) -> np.ndarray:
        """Sample pixels along a line"""
        # Bresenham's line algorithm
        pixels = []
        x1, y1 = pt1
        x2, y2 = pt2

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]:
                pixels.append(image[y1, x1])

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return np.array(pixels)

    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution (for heatmaps)"""
        if len(image.shape) != 3:
            return 0.0

        # Calculate color histogram
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

        # Calculate entropy as measure of color distribution
        def entropy(hist):
            hist = hist.flatten() / hist.sum()
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))

        avg_entropy = (entropy(hist_b) + entropy(hist_g) + entropy(hist_r)) / 3

        # Normalize to 0-1 range
        return min(avg_entropy / 8, 1.0)  # 8 bits max entropy

    def _extract_visual_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract visual features for classification"""
        features = {}

        # Basic shape features
        features['aspect_ratio'] = image.shape[1] / image.shape[0]

        # Edge features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size

        # Circular features
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30)
        features['circular_features'] = len(circles[0]) if circles is not None else 0

        # Rectangular features
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rect_count = 0
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                rect_count += 1

        features['rectangular_features'] = rect_count

        return features

    def get_chart_properties(self, image: np.ndarray,
                           chart_type: str) -> Dict[str, Any]:
        """
        Get specific properties based on detected chart type

        Args:
            image: Input image
            chart_type: Detected chart type

        Returns:
            Dictionary of chart-specific properties
        """
        properties = {
            'type': chart_type,
            'has_legend': self._detect_legend(image),
            'has_title': self._detect_title(image),
            'has_grid': self._detect_grid(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                                        if len(image.shape) == 3 else image),
            'color_scheme': self._extract_color_scheme(image)
        }

        # Chart-specific properties
        if chart_type == 'scatter_plot':
            properties['num_points'] = self._count_scatter_points(image)
            properties['has_regression_line'] = self._detect_regression_line(image)

        elif chart_type == 'bar_chart':
            properties['num_bars'] = self._count_bars(image)
            properties['orientation'] = self._detect_bar_orientation(image)

        elif chart_type == 'heatmap':
            properties['has_colorbar'] = self._detect_colorbar(image)
            properties['matrix_size'] = self._estimate_matrix_size(image)

        return properties

    def _detect_legend(self, image: np.ndarray) -> bool:
        """Detect if chart has a legend"""
        # Look for small rectangular regions with text
        # Simplified implementation
        return False

    def _detect_title(self, image: np.ndarray) -> bool:
        """Detect if chart has a title"""
        # Check top region for text
        return False

    def _extract_color_scheme(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from chart"""
        if len(image.shape) != 3:
            return []

        # K-means clustering for dominant colors
        from sklearn.cluster import KMeans

        pixels = image.reshape(-1, 3)
        # Sample pixels for efficiency
        sample_size = min(1000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]

        # Cluster colors
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(sample_pixels)

        colors = kmeans.cluster_centers_.astype(int)
        return [tuple(color) for color in colors]

    def _count_scatter_points(self, image: np.ndarray) -> int:
        """Count approximate number of points in scatter plot"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Detect circles and blobs
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10,
                                  param1=50, param2=30,
                                  minRadius=2, maxRadius=20)

        count = 0
        if circles is not None:
            count = len(circles[0])

        return count

    def _detect_regression_line(self, image: np.ndarray) -> bool:
        """Detect if scatter plot has regression/trend line"""
        # Would detect continuous line among scatter points
        return False

    def _count_bars(self, image: np.ndarray) -> int:
        """Count number of bars in bar chart"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bar_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:  # Minimum bar size
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 5:  # Typical bar ratios
                    bar_count += 1

        return bar_count

    def _detect_bar_orientation(self, image: np.ndarray) -> str:
        """Detect if bars are vertical or horizontal"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vertical_bars = 0
        horizontal_bars = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                if h > w:
                    vertical_bars += 1
                else:
                    horizontal_bars += 1

        return 'vertical' if vertical_bars > horizontal_bars else 'horizontal'

    def _detect_colorbar(self, image: np.ndarray) -> bool:
        """Detect if heatmap has a colorbar"""
        # Look for vertical or horizontal gradient bar
        return False

    def _estimate_matrix_size(self, image: np.ndarray) -> Tuple[int, int]:
        """Estimate matrix dimensions for heatmap"""
        # Would detect grid structure
        return (10, 10)  # Placeholder