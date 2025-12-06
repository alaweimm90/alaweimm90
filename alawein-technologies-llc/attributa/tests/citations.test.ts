import { extractDOI, resolveDOI } from '../src/lib/citations/doi';
import { parseCitation, validateCitation } from '../src/lib/citations/crossref';
import { parseBibTeX, formatBibTeXEntry, suggestPatchedEntry } from '../src/lib/citations/bibtex';

describe('Citation Validation', () => {
  describe('DOI extraction and resolution', () => {
    it('should extract DOI from various formats', () => {
      const texts = [
        'doi:10.1234/example.2023.001',
        'https://doi.org/10.1234/example.2023.001',
        'See paper at 10.1234/example.2023.001 for details',
        'Available at https://dx.doi.org/10.1234/example.2023.001'
      ];
      
      for (const text of texts) {
        const doi = extractDOI(text);
        expect(doi).toBe('10.1234/example.2023.001');
      }
    });

    it('should return null for text without DOI', () => {
      const text = 'This text has no DOI reference';
      const doi = extractDOI(text);
      expect(doi).toBeNull();
    });

    it('should return resolves=false for fake DOI', async () => {
      const fakeDOI = '10.9999/nonexistent.2023.999';
      const result = await resolveDOI(fakeDOI);
      
      expect(result.resolves).toBe(false);
      expect(result.meta).toBeUndefined();
    });

    // Note: Real DOI resolution would require network access
    // In unit tests, we're testing the logic, not the actual API calls
  });

  describe('Citation parsing', () => {
    it('should parse citation components correctly', () => {
      const citation = 'Smith, J. and Jones, M. (2023). "Understanding Machine Learning". Nature, 123(4), 567-589.';
      const parsed = parseCitation(citation);
      
      expect(parsed.year).toBe(2023);
      expect(parsed.title).toBe('Understanding Machine Learning');
      expect(parsed.authors).toBeDefined();
      expect(parsed.authors!.length).toBeGreaterThan(0);
    });

    it('should extract year from citation', () => {
      const citations = [
        'Published in 2022',
        'Conference proceedings (2021)',
        'Journal article, 2020'
      ];
      
      const years = [2022, 2021, 2020];
      
      citations.forEach((citation, i) => {
        const parsed = parseCitation(citation);
        expect(parsed.year).toBe(years[i]);
      });
    });

    it('should handle citations without clear structure', () => {
      const citation = 'Some random text without proper citation format';
      const parsed = parseCitation(citation);
      
      // Should return empty object or partial data
      expect(parsed).toBeDefined();
    });
  });

  describe('BibTeX parsing', () => {
    it('should parse BibTeX entries correctly', () => {
      const bibtex = `
        @article{smith2023,
          title = {Machine Learning Advances},
          author = {Smith, John and Doe, Jane},
          year = {2023},
          journal = {AI Journal},
          doi = {10.1234/example.2023}
        }
      `;
      
      const entries = parseBibTeX(bibtex);
      
      expect(entries).toHaveLength(1);
      expect(entries[0].type).toBe('article');
      expect(entries[0].key).toBe('smith2023');
      expect(entries[0].fields.title).toBe('Machine Learning Advances');
      expect(entries[0].fields.year).toBe('2023');
      expect(entries[0].fields.doi).toBe('10.1234/example.2023');
    });

    it('should format BibTeX entry back to string', () => {
      const entry = {
        type: 'article',
        key: 'test2023',
        fields: {
          title: 'Test Article',
          author: 'Test Author',
          year: '2023'
        }
      };
      
      const formatted = formatBibTeXEntry(entry);
      
      expect(formatted).toContain('@article{test2023,');
      expect(formatted).toContain('title = {Test Article}');
      expect(formatted).toContain('author = {Test Author}');
      expect(formatted).toContain('year = {2023}');
    });

    it('should suggest patches for incomplete BibTeX entries', () => {
      const entry = {
        type: 'article',
        key: 'incomplete2023',
        fields: {
          title: 'Incomplete Entry',
          author: 'Author Name'
        }
      };
      
      const metadata = {
        title: 'Complete Title for Entry',
        doi: '10.1234/complete.2023',
        year: 2023,
        venue: 'Journal Name'
      };
      
      const patched = suggestPatchedEntry(entry, metadata);
      
      expect(patched.fields.doi).toBe('10.1234/complete.2023');
      expect(patched.fields.year).toBe('2023');
      expect(patched.fields.journal).toBe('Journal Name');
    });
  });

  describe('Citation validation with rate limiting', () => {
    it('should validate citation and return suggestions', async () => {
      const citation = 'Fictional paper about AI (2023)';
      const result = await validateCitation(citation);
      
      expect(result.raw).toBe(citation);
      expect(result.resolves).toBe(false);
      expect(result.suggestions).toBeDefined();
      expect(Array.isArray(result.suggestions)).toBe(true);
    });

    it('should detect DOI in citation and attempt resolution', async () => {
      const citationWithDOI = 'Smith et al. (2023). Test paper. doi:10.9999/fake.2023.001';
      const result = await validateCitation(citationWithDOI);
      
      expect(result.raw).toBe(citationWithDOI);
      expect(result.resolves).toBe(false); // Fake DOI won't resolve
    });
  });
});