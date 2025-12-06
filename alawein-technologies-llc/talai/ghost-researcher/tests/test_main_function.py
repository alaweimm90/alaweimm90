"""
Direct tests for main() function to increase coverage
"""
import pytest
import sys
from pathlib import Path
from io import StringIO
from unittest.mock import patch
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ghost_researcher.main import main


class TestMainFunctionDirect:
    """Test main() function directly"""

    def test_main_list_command(self, capsys):
        """Test main() with list command"""
        with patch('sys.argv', ['ghost_researcher', 'list']):
            main()
            captured = capsys.readouterr()
            assert 'einstein' in captured.out.lower()
            assert 'AVAILABLE SCIENTISTS' in captured.out

    def test_main_info_command(self, capsys):
        """Test main() with info command"""
        with patch('sys.argv', ['ghost_researcher', 'info', '--scientist', 'curie']):
            main()
            captured = capsys.readouterr()
            assert 'Marie Curie' in captured.out or 'MARIE CURIE' in captured.out
            assert '1867-1934' in captured.out

    def test_main_consult_command(self, capsys):
        """Test main() with consult command"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            with patch('sys.argv', [
                'ghost_researcher', 'consult',
                '--scientist', 'feynman',
                '--problem', 'machine learning',
                '--domain', 'computer_science'
            ]):
                main()
                captured = capsys.readouterr()

                # Should show consultation output
                assert 'Feynman' in captured.out or 'FEYNMAN' in captured.out
                assert 'machine learning' in captured.out.lower()
        finally:
            # Clean up the default ghost.json file
            Path('ghost.json').unlink(missing_ok=True)

    def test_main_consult_creates_output(self):
        """Test that consult command produces output"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            with patch('sys.argv', [
                'ghost_researcher', 'consult',
                '--scientist', 'darwin',
                '--problem', 'neural networks evolution',
                '--domain', 'biology'
            ]):
                # Capture output
                output = StringIO()
                with patch('sys.stdout', output):
                    main()

                result = output.getvalue()
                # Check for key sections
                assert len(result) > 500  # Should be substantial
                assert 'Darwin' in result or 'DARWIN' in result
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_main_info_all_scientists(self):
        """Test info command for multiple scientists"""
        scientists = ['einstein', 'newton', 'lovelace', 'franklin']

        for scientist in scientists:
            with patch('sys.argv', ['ghost_researcher', 'info', '--scientist', scientist]):
                output = StringIO()
                with patch('sys.stdout', output):
                    main()

                result = output.getvalue()
                assert len(result) > 100
                assert 'KNOWN FOR' in result
                assert 'PERSONALITY' in result

    def test_main_list_shows_all_scientists(self):
        """Test list command shows all scientists"""
        with patch('sys.argv', ['ghost_researcher', 'list']):
            output = StringIO()
            with patch('sys.stdout', output):
                main()

            result = output.getvalue()
            # Should list all 8 scientists
            assert result.count('•') >= 8  # Each scientist listed with bullet

    def test_main_consult_with_different_domains(self):
        """Test consult with different problem domains"""
        domains = [
            ('einstein', 'quantum entanglement', 'physics'),
            ('darwin', 'genetic engineering', 'biology'),
            ('turing', 'deep learning', 'computer_science'),
        ]

        for scientist, problem, domain in domains:
            with patch('sys.argv', [
                'ghost_researcher', 'consult',
                '--scientist', scientist,
                '--problem', problem,
                '--domain', domain
            ]):
                output = StringIO()
                with patch('sys.stdout', output):
                    main()

                result = output.getvalue()
                assert problem in result.lower()
                assert domain in result.lower() or domain.replace('_', ' ') in result.lower()
