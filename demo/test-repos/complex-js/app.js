const fs = require('fs');
const path = require('path');

function processFiles(directory, callback) {
  fs.readdir(directory, (err, files) => {
    if (err) return callback(err);
    let processed = 0;
    const results = [];
    if (files.length === 0) return callback(null, results);

    files.forEach((file, index) => {
      const filePath = path.join(directory, file);
      fs.stat(filePath, (err, stats) => {
        if (err) return callback(err);
        if (stats.isDirectory()) {
          processFiles(filePath, (err, subResults) => {
            if (err) return callback(err);
            results.push(...subResults);
            processed++;
            if (processed === files.length) {
              callback(null, results);
            }
          });
        } else {
          fs.readFile(filePath, 'utf8', (err, content) => {
            if (err) return callback(err);
            const lines = content.split('\n');
            const wordCount = content.split(/\s+/).length;
            const charCount = content.length;
            const complexity = calculateComplexity(content);
            results.push({
              file: filePath,
              lines: lines.length,
              words: wordCount,
              chars: charCount,
              complexity: complexity,
              size: stats.size
            });
            processed++;
            if (processed === files.length) {
              callback(null, results);
            }
          });
        }
      });
    });
  });
}

function calculateComplexity(content) {
  let complexity = 1;
  const lines = content.split('\n');
  lines.forEach(line => {
    const trimmed = line.trim();
    if (trimmed.includes('if ') || trimmed.includes('else') || trimmed.includes('for ') || trimmed.includes('while ')) {
      complexity++;
    }
    if (trimmed.includes('&&') || trimmed.includes('||')) {
      complexity += 0.5;
    }
  });
  return complexity;
}

function analyzeResults(results, callback) {
  const summary = {
    totalFiles: results.length,
    totalLines: 0,
    totalWords: 0,
    totalChars: 0,
    avgComplexity: 0,
    maxComplexity: 0,
    minComplexity: Infinity
  };

  results.forEach(result => {
    summary.totalLines += result.lines;
    summary.totalWords += result.words;
    summary.totalChars += result.chars;
    summary.avgComplexity += result.complexity;
    if (result.complexity > summary.maxComplexity) {
      summary.maxComplexity = result.complexity;
    }
    if (result.complexity < summary.minComplexity) {
      summary.minComplexity = result.complexity;
    }
  });

  if (results.length > 0) {
    summary.avgComplexity /= results.length;
  } else {
    summary.minComplexity = 0;
  }

  const categorized = {
    simple: results.filter(r => r.complexity < 5),
    moderate: results.filter(r => r.complexity >= 5 && r.complexity < 10),
    complex: results.filter(r => r.complexity >= 10)
  };

  callback(null, { summary, categorized, details: results });
}

function generateReport(analysis, callback) {
  const report = `
Analysis Report
===============

Total Files: ${analysis.summary.totalFiles}
Total Lines: ${analysis.summary.totalLines}
Total Words: ${analysis.summary.totalWords}
Total Characters: ${analysis.summary.totalChars}

Complexity Metrics:
- Average: ${analysis.summary.avgComplexity.toFixed(2)}
- Maximum: ${analysis.summary.maxComplexity}
- Minimum: ${analysis.summary.minComplexity}

File Categories:
- Simple (< 5): ${analysis.categorized.simple.length}
- Moderate (5-10): ${analysis.categorized.moderate.length}
- Complex (>= 10): ${analysis.categorized.complex.length}

Detailed Results:
${analysis.details.map(d => `- ${d.file}: ${d.complexity} complexity`).join('\n')}
  `;

  fs.writeFile('report.txt', report, (err) => {
    if (err) return callback(err);
    callback(null, report);
  });
}

function main() {
  const directory = process.argv[2] || '.';
  processFiles(directory, (err, results) => {
    if (err) {
      console.error('Error processing files:', err);
      process.exit(1);
    }
    analyzeResults(results, (err, analysis) => {
      if (err) {
        console.error('Error analyzing results:', err);
        process.exit(1);
      }
      generateReport(analysis, (err, report) => {
        if (err) {
          console.error('Error generating report:', err);
          process.exit(1);
        }
        console.log('Report generated successfully');
        console.log(report);
      });
    });
  });
}

main();