import React, { useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface LaTeXRendererProps {
  content: string;
  displayMode?: boolean;
  className?: string;
}

export const LaTeXRenderer: React.FC<LaTeXRendererProps> = ({ 
  content, 
  displayMode = false, 
  className = '' 
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      try {
        // Render LaTeX content with KaTeX
        const renderedHtml = katex.renderToString(content, {
          displayMode,
          throwOnError: false,
          errorColor: '#cc0000',
          strict: 'warn',
          trust: true,
          maxSize: Infinity,
          maxExpand: 1000,
        });
        
        containerRef.current.innerHTML = renderedHtml;
      } catch (error) {
        console.error('LaTeX rendering error:', error);
        containerRef.current.innerHTML = `<span style="color: #cc0000;">LaTeX Error: ${content}</span>`;
      }
    }
  }, [content, displayMode]);

  return (
    <div 
      ref={containerRef} 
      className={`katex-container ${className}`}
      style={{
        fontSize: displayMode ? '1.2em' : '1em',
        textAlign: displayMode ? 'center' : 'inherit',
        margin: displayMode ? '1em 0' : '0',
      }}
    />
  );
};

// Enhanced LaTeX content processor for blog articles
export const processLaTeXContent = (content: string): string => {
  // Process block equations $$...$$
  content = content.replace(/\$\$([\s\S]*?)\$\$/g, (match, latex) => {
    try {
      return `<div class="latex-block">${katex.renderToString(latex.trim(), { 
        displayMode: true,
        throwOnError: false 
      })}</div>`;
    } catch (e) {
      return `<div class="latex-error">LaTeX Error: ${latex}</div>`;
    }
  });

  // Process inline equations $...$
  content = content.replace(/\$([^$\n]+?)\$/g, (match, latex) => {
    try {
      return `<span class="latex-inline">${katex.renderToString(latex.trim(), { 
        displayMode: false,
        throwOnError: false 
      })}</span>`;
    } catch (e) {
      return `<span class="latex-error">${latex}</span>`;
    }
  });

  return content;
};

// Common LaTeX macros for fitness and nutrition formulas
export const fitnessLaTeXMacros = {
  '\\BMI': '\\text{BMI}',
  '\\BMR': '\\text{BMR}',
  '\\TDEE': '\\text{TDEE}',
  '\\oneRM': '\\text{1RM}',
  '\\VO2max': '\\text{VO}_2\\text{max}',
  '\\RPE': '\\text{RPE}',
  '\\BF': '\\text{BF\\%}',
  '\\LBM': '\\text{LBM}',
  '\\RMR': '\\text{RMR}',
  '\\HR': '\\text{HR}',
  '\\MHR': '\\text{MHR}',
  '\\HRR': '\\text{HRR}',
  '\\kcal': '\\text{kcal}',
  '\\protein': '\\text{protein}',
  '\\carbs': '\\text{carbohydrates}',
  '\\fat': '\\text{fat}',
};

// LaTeX equation examples for fitness content
export const fitnessEquations = {
  bmi: 'BMI = \\frac{\\text{weight (kg)}}{\\text{height (m)}^2}',
  bmr_male: 'BMR_{\\text{male}} = 10 \\times \\text{weight} + 6.25 \\times \\text{height} - 5 \\times \\text{age} + 5',
  bmr_female: 'BMR_{\\text{female}} = 10 \\times \\text{weight} + 6.25 \\times \\text{height} - 5 \\times \\text{age} - 161',
  tdee: 'TDEE = BMR \\times \\text{Activity Factor}',
  oneRM: '1RM = \\frac{\\text{weight}}{1.0278 - (0.0278 \\times \\text{reps})}',
  vo2max: 'VO_2\\text{max} = 15.3 \\times \\frac{\\text{MHR}}{\\text{RHR}}',
  targetHR: 'THR = ((MHR - RHR) \\times \\text{\\% intensity}) + RHR',
  calorieDeficit: '\\text{Weight Loss} = \\frac{\\text{Caloric Deficit}}{7700 \\text{ kcal/kg}}',
  proteinIntake: '\\text{Protein} = \\text{LBM} \\times (1.6 \\text{ to } 2.2) \\text{ g/kg}',
  waterIntake: '\\text{Water} = \\text{Body Weight} \\times 35 \\text{ ml/kg}',
};

export default LaTeXRenderer;