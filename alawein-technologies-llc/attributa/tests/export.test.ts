import { Report } from '../src/types';

// Mock the export functions since they use browser APIs
const mockExport = {
  exportReportAsJSON: jest.fn(),
  exportReportAsCSV: jest.fn(),
};

jest.mock('../src/lib/export', () => mockExport);

describe('Export functionality', () => {
  const mockReport: Report = {
    docId: 'test-doc-123',
    createdAt: Date.now(),
    summary: {
      totalChars: 1500,
      types: { prose: 2, code: 1 }
    },
    segments: [
      {
        segmentId: 'seg1',
        type: 'prose',
        content: 'Test content',
        lengthChars: 1000,
        startIndex: 0,
        endIndex: 1000
      },
      {
        segmentId: 'seg2', 
        type: 'code',
        content: 'console.log("test");',
        lengthChars: 500,
        startIndex: 1000,
        endIndex: 1500
      }
    ],
    signals: {
      seg1: {
        gltr: { tailTokenShare: 0.3, rankVariance: 0.8 },
        detectgpt: { curvature: -0.2, numPerturbations: 80 }
      },
      seg2: {
        gltr: { tailTokenShare: 0.4, rankVariance: 0.6 }
      }
    },
    scores: {
      seg1: { value: 0.75, confidence: 0.85, rationale: ['negative curvature', 'high variance'] },
      seg2: { value: 0.45, confidence: 0.60, rationale: ['moderate signals'] }
    }
  };

  const mockWeights = {
    gltr: 0.22,
    detectgpt: 0.22,
    watermark: 0.18,
    citations: 0.25,
    cwe: 0.10,
    shortPenalty: 0.03
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should have required export functions', () => {
    expect(mockExport.exportReportAsJSON).toBeDefined();
    expect(mockExport.exportReportAsCSV).toBeDefined();
  });

  it('should call exportReportAsJSON with correct parameters', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { exportReportAsJSON } = require('../src/lib/export');
    exportReportAsJSON(mockReport, mockWeights);
    
    expect(mockExport.exportReportAsJSON).toHaveBeenCalledWith(mockReport, mockWeights);
  });

  it('should call exportReportAsCSV with correct parameters', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { exportReportAsCSV } = require('../src/lib/export');
    exportReportAsCSV(mockReport);
    
    expect(mockExport.exportReportAsCSV).toHaveBeenCalledWith(mockReport);
  });
});