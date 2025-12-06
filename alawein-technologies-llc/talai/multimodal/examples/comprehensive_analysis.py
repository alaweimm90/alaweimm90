#!/usr/bin/env python3
"""
Comprehensive Multi-Modal Research Paper Analysis

This example demonstrates how to use all four multi-modal processing systems
together to analyze a complete research paper with figures, equations,
molecular structures, and supplementary videos.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Import all multi-modal modules
from TalAI.multimodal.figures import (
    FigureAnalyzer, DataExtractor, ChartTypeDetector,
    FigureOCR, CaptionProcessor, AccessibilityDescriptor
)
from TalAI.multimodal.equations import (
    EquationParser, LaTeXProcessor, HandwritingOCR,
    SymbolicEngine, DimensionalAnalyzer, EquationToCode
)
from TalAI.multimodal.molecules import (
    MolecularStructureParser, PropertyPredictor,
    BindingSiteAnalyzer, MolecularSimilarity,
    Molecular3DVisualizer, ChemicalDatabaseConnector
)
from TalAI.multimodal.av import (
    TranscriptionEngine, SpeakerDiarization,
    SlideExtractor, VideoAnalyzer, TimestampGenerator
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchPaperAnalyzer:
    """
    Comprehensive analyzer for research papers using all multi-modal systems
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize all processing systems"""
        self.config = config or {}

        # Initialize figure understanding
        self.figure_analyzer = FigureAnalyzer(
            model_backend=self.config.get('figure_model', 'claude'),
            gpu_enabled=self.config.get('use_gpu', True)
        )
        self.data_extractor = DataExtractor(precision_mode=True)
        self.chart_detector = ChartTypeDetector(use_ensemble=True)
        self.figure_ocr = FigureOCR(use_gpu=self.config.get('use_gpu', True))

        # Initialize equation processing
        self.equation_parser = EquationParser(notation_format="latex")
        self.symbolic_engine = SymbolicEngine()
        self.dimensional_analyzer = DimensionalAnalyzer()
        self.equation_to_code = EquationToCode()

        # Initialize molecular analysis
        self.mol_parser = MolecularStructureParser(cache_enabled=True)
        self.property_predictor = PropertyPredictor()
        self.binding_analyzer = BindingSiteAnalyzer()
        self.mol_similarity = MolecularSimilarity()

        # Initialize AV processing
        self.transcription_engine = TranscriptionEngine(
            model="whisper_large",
            enable_gpu=self.config.get('use_gpu', True)
        )
        self.slide_extractor = SlideExtractor()
        self.video_analyzer = VideoAnalyzer()

        logger.info("ResearchPaperAnalyzer initialized with all systems")

    def analyze_complete_paper(
        self,
        paper_dir: Path,
        output_dir: Path = None
    ) -> Dict[str, Any]:
        """
        Analyze a complete research paper with all supplementary materials

        Args:
            paper_dir: Directory containing paper and supplementary files
            output_dir: Optional output directory for results

        Returns:
            Comprehensive analysis results
        """
        paper_dir = Path(paper_dir)
        output_dir = output_dir or paper_dir / "analysis_output"
        output_dir.mkdir(exist_ok=True)

        logger.info(f"Starting comprehensive analysis of {paper_dir}")

        # Results container
        results = {
            'paper_path': str(paper_dir),
            'figures': [],
            'equations': [],
            'molecules': [],
            'videos': [],
            'integrated_insights': {},
            'quality_metrics': {}
        }

        # 1. Analyze Figures
        logger.info("Analyzing figures...")
        results['figures'] = self._analyze_figures(paper_dir)

        # 2. Process Equations
        logger.info("Processing equations...")
        results['equations'] = self._process_equations(paper_dir)

        # 3. Analyze Molecular Structures
        logger.info("Analyzing molecular structures...")
        results['molecules'] = self._analyze_molecules(paper_dir)

        # 4. Process Videos/Audio
        logger.info("Processing audio/video content...")
        results['videos'] = self._process_videos(paper_dir)

        # 5. Integrate and Cross-reference
        logger.info("Integrating multi-modal insights...")
        results['integrated_insights'] = self._integrate_insights(results)

        # 6. Calculate Quality Metrics
        results['quality_metrics'] = self._calculate_quality_metrics(results)

        # 7. Export Results
        self._export_results(results, output_dir)

        logger.info(f"Analysis complete. Results saved to {output_dir}")

        return results

    def _analyze_figures(self, paper_dir: Path) -> List[Dict]:
        """Analyze all figures in the paper"""
        figure_results = []

        # Find all figure files
        figure_patterns = ['*.png', '*.jpg', '*.jpeg', '*.pdf', '*.svg']
        figure_files = []
        for pattern in figure_patterns:
            figure_files.extend(paper_dir.glob(f"**/{pattern}"))

        for fig_path in figure_files:
            try:
                # Detect chart type
                img = self._load_image(fig_path)
                chart_type = self.chart_detector.detect_chart_type(img)

                # Analyze figure
                analysis = self.figure_analyzer.analyze_figure(
                    fig_path,
                    extract_data=True,
                    generate_accessibility=True
                )

                # Extract data points if applicable
                if analysis['figure_type'] in ['scatter_plot', 'line_chart', 'bar_chart']:
                    data_points = self.data_extractor.extract_from_scatter_plot(img)
                    analysis['extracted_data_points'] = [
                        {'x': p.x, 'y': p.y, 'error_y': p.error_y}
                        for p in data_points
                    ]

                # OCR text from figure
                ocr_results = self.figure_ocr.extract_all_text(img)
                analysis['ocr_text'] = ocr_results

                # Extract axis information
                axes_info = self.figure_ocr.extract_axis_labels(img)
                analysis['axes'] = axes_info

                figure_results.append({
                    'path': str(fig_path),
                    'analysis': analysis,
                    'chart_type_confidence': chart_type
                })

                logger.info(f"Analyzed figure: {fig_path.name}")

            except Exception as e:
                logger.error(f"Error analyzing figure {fig_path}: {e}")
                figure_results.append({
                    'path': str(fig_path),
                    'error': str(e)
                })

        return figure_results

    def _process_equations(self, paper_dir: Path) -> List[Dict]:
        """Process all equations found in the paper"""
        equation_results = []

        # Find LaTeX files or extract equations from PDFs
        tex_files = list(paper_dir.glob("**/*.tex"))

        for tex_path in tex_files:
            try:
                with open(tex_path, 'r') as f:
                    content = f.read()

                # Extract equations (simplified - would use regex in production)
                import re
                equation_pattern = r'\\begin\{equation\}(.*?)\\end\{equation\}'
                equations = re.findall(equation_pattern, content, re.DOTALL)

                for eq_text in equations:
                    # Parse equation
                    parsed = self.equation_parser.parse(eq_text)

                    # Symbolic analysis
                    if parsed.parsed_expr:
                        simplified = self.symbolic_engine.simplify(parsed.parsed_expr)
                        derivatives = self.symbolic_engine.calculate_derivatives(
                            parsed.parsed_expr,
                            parsed.variables
                        )

                        # Dimensional analysis for physics equations
                        if parsed.equation_type.value == 'physics':
                            dimensional_check = self.dimensional_analyzer.check_consistency(
                                parsed.parsed_expr
                            )
                        else:
                            dimensional_check = None

                        # Generate code
                        python_code = self.equation_to_code.generate(
                            parsed,
                            language='python'
                        )

                        equation_results.append({
                            'raw_latex': eq_text,
                            'parsed': {
                                'type': parsed.equation_type.value,
                                'variables': parsed.variables,
                                'complexity': parsed.complexity_score
                            },
                            'simplified': str(simplified),
                            'derivatives': {var: str(deriv) for var, deriv in derivatives.items()},
                            'dimensional_check': dimensional_check,
                            'generated_code': python_code
                        })

                        logger.info(f"Processed equation: {eq_text[:50]}...")

            except Exception as e:
                logger.error(f"Error processing equations in {tex_path}: {e}")

        return equation_results

    def _analyze_molecules(self, paper_dir: Path) -> List[Dict]:
        """Analyze molecular structure files"""
        molecule_results = []

        # Find molecular structure files
        mol_patterns = ['*.pdb', '*.sdf', '*.mol2', '*.cif', '*.xyz']
        mol_files = []
        for pattern in mol_patterns:
            mol_files.extend(paper_dir.glob(f"**/{pattern}"))

        # Also check for SMILES in text files
        smiles_files = list(paper_dir.glob("**/*smiles.txt"))

        for mol_path in mol_files:
            try:
                # Parse structure
                structure = self.mol_parser.parse(mol_path)

                # Predict properties
                properties = self.property_predictor.predict(structure)

                # Analyze binding sites for proteins
                binding_sites = []
                if structure.molecule_type.value == 'protein':
                    binding_sites = self.binding_analyzer.detect_sites(structure)

                # Find similar molecules
                similar = self.mol_similarity.find_similar(
                    structure,
                    database='pubchem',
                    threshold=0.7,
                    max_results=5
                )

                molecule_results.append({
                    'path': str(mol_path),
                    'type': structure.molecule_type.value,
                    'formula': structure.formula,
                    'molecular_weight': structure.molecular_weight,
                    'num_atoms': len(structure.atoms),
                    'num_bonds': len(structure.bonds),
                    'properties': properties,
                    'binding_sites': binding_sites,
                    'similar_molecules': similar
                })

                logger.info(f"Analyzed molecule: {mol_path.name}")

            except Exception as e:
                logger.error(f"Error analyzing molecule {mol_path}: {e}")

        # Process SMILES
        for smiles_path in smiles_files:
            try:
                with open(smiles_path, 'r') as f:
                    for line in f:
                        smiles = line.strip()
                        if smiles:
                            structure = self.mol_parser.parse_string(
                                smiles,
                                format='SMILES'
                            )

                            properties = self.property_predictor.predict(structure)

                            molecule_results.append({
                                'smiles': smiles,
                                'formula': structure.formula,
                                'molecular_weight': structure.molecular_weight,
                                'properties': properties
                            })

            except Exception as e:
                logger.error(f"Error processing SMILES {smiles_path}: {e}")

        return molecule_results

    def _process_videos(self, paper_dir: Path) -> List[Dict]:
        """Process video and audio supplementary materials"""
        video_results = []

        # Find video/audio files
        av_patterns = ['*.mp4', '*.avi', '*.mov', '*.webm', '*.mp3', '*.wav']
        av_files = []
        for pattern in av_patterns:
            av_files.extend(paper_dir.glob(f"**/{pattern}"))

        for av_path in av_files:
            try:
                # Transcribe content
                transcription = self.transcription_engine.transcribe(
                    av_path,
                    extract_key_points=True,
                    identify_speakers=True
                )

                # Extract slides if video
                slides = []
                if av_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.webm']:
                    slides = self.slide_extractor.extract_slides(
                        av_path,
                        ocr_text=True
                    )

                    # Analyze video content
                    video_analysis = self.video_analyzer.analyze(av_path)
                else:
                    video_analysis = None

                video_results.append({
                    'path': str(av_path),
                    'duration': transcription.duration,
                    'content_type': transcription.content_type.value,
                    'speakers': transcription.speakers,
                    'key_points': transcription.key_points[:5],  # Top 5
                    'technical_terms': transcription.technical_vocabulary[:10],  # Top 10
                    'transcript_preview': transcription.full_text[:500],
                    'num_slides': len(slides),
                    'video_analysis': video_analysis,
                    'quality_score': transcription.quality_score
                })

                # Export transcript
                transcript_path = av_path.with_suffix('.vtt')
                self.transcription_engine.export_transcript(
                    transcription,
                    transcript_path,
                    format='vtt'
                )

                logger.info(f"Processed video/audio: {av_path.name}")

            except Exception as e:
                logger.error(f"Error processing video {av_path}: {e}")

        return video_results

    def _integrate_insights(self, results: Dict) -> Dict:
        """Integrate insights across all modalities"""
        insights = {
            'cross_references': [],
            'data_consistency': [],
            'key_findings': [],
            'technical_concepts': set()
        }

        # Cross-reference equations with figures
        for eq in results['equations']:
            for fig in results['figures']:
                if 'ocr_text' in fig.get('analysis', {}):
                    # Check if equation variables appear in figure
                    ocr_text = ' '.join([t['text'] for t in fig['analysis']['ocr_text']])
                    for var in eq['parsed']['variables']:
                        if var in ocr_text:
                            insights['cross_references'].append({
                                'type': 'equation_figure',
                                'equation': eq['raw_latex'][:50],
                                'figure': fig['path'],
                                'variable': var
                            })

        # Check data consistency between figures
        for i, fig1 in enumerate(results['figures']):
            for fig2 in results['figures'][i+1:]:
                if 'extracted_data_points' in fig1.get('analysis', {}):
                    if 'extracted_data_points' in fig2.get('analysis', {}):
                        # Compare data ranges
                        data1 = fig1['analysis']['extracted_data_points']
                        data2 = fig2['analysis']['extracted_data_points']

                        if data1 and data2:
                            range1 = (min(d['x'] for d in data1), max(d['x'] for d in data1))
                            range2 = (min(d['x'] for d in data2), max(d['x'] for d in data2))

                            overlap = self._calculate_range_overlap(range1, range2)
                            if overlap > 0.5:
                                insights['data_consistency'].append({
                                    'figures': [fig1['path'], fig2['path']],
                                    'overlap': overlap
                                })

        # Extract key findings from transcripts
        for video in results['videos']:
            insights['key_findings'].extend(video.get('key_points', []))
            insights['technical_concepts'].update(video.get('technical_terms', []))

        # Add molecular insights
        for mol in results['molecules']:
            if 'properties' in mol:
                if mol['properties'].get('drug_likeness', 0) > 0.7:
                    insights['key_findings'].append(
                        f"Molecule {mol.get('formula', 'Unknown')} shows drug-like properties"
                    )

        insights['technical_concepts'] = list(insights['technical_concepts'])

        return insights

    def _calculate_quality_metrics(self, results: Dict) -> Dict:
        """Calculate overall quality metrics for the analysis"""
        metrics = {
            'figures_analyzed': len(results['figures']),
            'figures_with_data': sum(1 for f in results['figures']
                                    if 'extracted_data_points' in f.get('analysis', {})),
            'equations_parsed': len(results['equations']),
            'equations_simplified': sum(1 for e in results['equations']
                                       if 'simplified' in e),
            'molecules_analyzed': len(results['molecules']),
            'molecules_with_properties': sum(1 for m in results['molecules']
                                           if 'properties' in m),
            'videos_transcribed': len(results['videos']),
            'average_transcription_quality': np.mean([v['quality_score']
                                                     for v in results['videos']
                                                     if 'quality_score' in v]) if results['videos'] else 0,
            'total_technical_terms': len(results['integrated_insights'].get('technical_concepts', [])),
            'cross_references_found': len(results['integrated_insights'].get('cross_references', []))
        }

        # Calculate overall success rate
        total_items = sum([metrics['figures_analyzed'],
                          metrics['equations_parsed'],
                          metrics['molecules_analyzed'],
                          metrics['videos_transcribed']])

        successful_items = sum([metrics['figures_with_data'],
                               metrics['equations_simplified'],
                               metrics['molecules_with_properties'],
                               metrics['videos_transcribed']])

        metrics['overall_success_rate'] = successful_items / total_items if total_items > 0 else 0

        return metrics

    def _export_results(self, results: Dict, output_dir: Path):
        """Export results in multiple formats"""
        # JSON export
        json_path = output_dir / "analysis_results.json"
        with open(json_path, 'w') as f:
            # Convert non-serializable objects
            def default_serializer(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                else:
                    return str(obj)

            json.dump(results, f, indent=2, default=default_serializer)

        # Create summary report
        summary_path = output_dir / "analysis_summary.md"
        with open(summary_path, 'w') as f:
            f.write("# Multi-Modal Analysis Summary\n\n")

            # Quality metrics
            f.write("## Quality Metrics\n\n")
            for key, value in results['quality_metrics'].items():
                f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")

            # Key findings
            f.write("\n## Key Findings\n\n")
            for finding in results['integrated_insights'].get('key_findings', [])[:10]:
                f.write(f"- {finding}\n")

            # Technical concepts
            f.write("\n## Technical Concepts Identified\n\n")
            concepts = results['integrated_insights'].get('technical_concepts', [])
            f.write(", ".join(concepts[:20]))

            # Figure summary
            f.write("\n\n## Figures Analyzed\n\n")
            for fig in results['figures'][:5]:  # First 5
                if 'error' not in fig:
                    f.write(f"- {Path(fig['path']).name}: "
                           f"{fig.get('analysis', {}).get('figure_type', 'Unknown')}\n")

            # Equation summary
            f.write("\n## Equations Processed\n\n")
            for eq in results['equations'][:5]:  # First 5
                f.write(f"- Type: {eq['parsed']['type']}, "
                       f"Variables: {', '.join(eq['parsed']['variables'])}\n")

            # Molecule summary
            f.write("\n## Molecules Analyzed\n\n")
            for mol in results['molecules'][:5]:  # First 5
                f.write(f"- {mol.get('formula', 'Unknown')}: "
                       f"MW={mol.get('molecular_weight', 0):.2f}\n")

        # Create data export
        if results['figures']:
            # Export extracted data points to CSV
            all_data_points = []
            for fig in results['figures']:
                if 'extracted_data_points' in fig.get('analysis', {}):
                    points = fig['analysis']['extracted_data_points']
                    for p in points:
                        p['figure'] = Path(fig['path']).name
                        all_data_points.append(p)

            if all_data_points:
                df = pd.DataFrame(all_data_points)
                df.to_csv(output_dir / "extracted_data.csv", index=False)

        logger.info(f"Results exported to {output_dir}")

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from file"""
        from PIL import Image
        img = Image.open(path).convert('RGB')
        return np.array(img)

    def _calculate_range_overlap(self, range1: Tuple[float, float],
                                range2: Tuple[float, float]) -> float:
        """Calculate overlap between two ranges"""
        overlap_start = max(range1[0], range2[0])
        overlap_end = min(range1[1], range2[1])

        if overlap_start >= overlap_end:
            return 0.0

        overlap_length = overlap_end - overlap_start
        range1_length = range1[1] - range1[0]
        range2_length = range2[1] - range2[0]

        return overlap_length / min(range1_length, range2_length)


def main():
    """Example usage of comprehensive paper analyzer"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze research paper with multi-modal processing')
    parser.add_argument('paper_dir', type=str, help='Directory containing paper and supplementary files')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--figure_model', choices=['claude', 'gpt4v', 'local'],
                       default='claude', help='Vision model for figure analysis')

    args = parser.parse_args()

    # Configuration
    config = {
        'use_gpu': args.use_gpu,
        'figure_model': args.figure_model
    }

    # Run analysis
    analyzer = ResearchPaperAnalyzer(config)
    results = analyzer.analyze_complete_paper(
        Path(args.paper_dir),
        Path(args.output_dir) if args.output_dir else None
    )

    # Print summary
    print("\n=== Analysis Complete ===")
    print(f"Figures analyzed: {results['quality_metrics']['figures_analyzed']}")
    print(f"Equations processed: {results['quality_metrics']['equations_parsed']}")
    print(f"Molecules analyzed: {results['quality_metrics']['molecules_analyzed']}")
    print(f"Videos transcribed: {results['quality_metrics']['videos_transcribed']}")
    print(f"Overall success rate: {results['quality_metrics']['overall_success_rate']:.1%}")


if __name__ == "__main__":
    main()