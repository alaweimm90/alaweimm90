// Demo datasets for testing different analyzers

export const demoContent = {
  humanText: {
    title: "Human-written Sample (University Essay)",
    content: `The concept of artificial intelligence has captivated human imagination for decades, but I find myself wondering whether we're approaching it from the right angle. When I was young, my grandmother used to tell me stories about how she learned to cook – not from following exact recipes, but by understanding the essence of flavors, the way ingredients interact, and developing an intuitive sense of what works.

This reminds me of how humans actually think. We don't process information in neat, logical sequences like computers do. Instead, we make leaps, have hunches, change our minds halfway through sentences, and somehow still arrive at meaningful conclusions. Last week, while struggling with a programming assignment, I realized that my best insights came not from methodically working through the problem, but from taking a shower and letting my mind wander.

So here's my question: if human intelligence is messy, intuitive, and wonderfully unpredictable, why are we trying to create AI that's the opposite? Maybe we're missing something fundamental about what makes intelligence truly intelligent.`,
    type: "prose" as const
  },

  aiText: {
    title: "AI-generated Sample (GPT-style)",
    content: `Artificial intelligence represents one of the most significant technological developments of our time, with profound implications for various sectors of society. The rapid advancement of machine learning algorithms and neural networks has enabled unprecedented capabilities in data processing, pattern recognition, and automated decision-making.

Furthermore, the integration of AI systems across industries has demonstrated remarkable potential for enhancing efficiency and productivity. These sophisticated technologies leverage vast amounts of data to generate insights and predictions that were previously unattainable through traditional computational methods.

Moreover, the continued evolution of artificial intelligence presents both opportunities and challenges that require careful consideration. The development of robust AI frameworks necessitates collaboration between researchers, policymakers, and industry stakeholders to ensure responsible implementation and ethical guidelines.

In conclusion, artificial intelligence will undoubtedly continue to shape the future of human civilization, transforming how we work, communicate, and interact with technology in increasingly sophisticated ways.`,
    type: "prose" as const
  },

  latexWithFakeDOIs: {
    title: "Academic Paper with Invalid Citations",
    content: `\\documentclass{article}
\\title{Novel Approaches in Machine Learning}
\\author{Demo Author}

\\begin{document}
\\section{Introduction}

Recent advances in machine learning have shown promising results in various applications. This paper reviews current methodologies and proposes new approaches for improving model performance.

The transformer architecture has revolutionized natural language processing tasks, enabling models to capture long-range dependencies effectively. However, computational requirements remain a significant challenge for deployment in resource-constrained environments.

\\section{Related Work}

Several studies have contributed to this field. The foundational work by Smith et al. demonstrates the effectiveness of attention mechanisms~\\cite{smith2023attention}. Additionally, the research by Johnson and Brown provides insights into optimization techniques~\\cite{johnson2023optimization}.

\\section{Methodology}

Our approach builds upon existing frameworks while introducing novel architectural modifications. We employ a hybrid model that combines convolutional and attention-based components to achieve better performance.

\\bibliographystyle{plain}
\\begin{thebibliography}{2}

\\bibitem{smith2023attention}
Smith, J., Davis, M., and Wilson, K.
\\newblock Advanced attention mechanisms in neural networks.
\\newblock \\emph{Journal of Machine Learning}, 45(3):123-145, 2023.
\\newblock doi:10.1234/nonexistent.2023.001

\\bibitem{johnson2023optimization}
Johnson, A. and Brown, S.
\\newblock Optimization techniques for large-scale neural networks.
\\newblock In \\emph{Proceedings of ICML}, pages 567-582, 2023.
\\newblock doi:10.5555/fake.doi.2022.456

\\end{thebibliography}

\\end{document}`,
    type: "latex" as const
  },

  vulnerablePython: {
    title: "Vulnerable Python Code (Multiple Security Issues)",
    content: `#!/usr/bin/env python3
"""
Demo web application with intentional security vulnerabilities
This is for educational/demo purposes only - DO NOT use in production!
"""

import subprocess
import os
import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# CWE-798: Hardcoded credentials
DATABASE_PASSWORD = "admin123"
API_KEY = "sk-1234567890abcdef"
SECRET_TOKEN = "super_secret_token_hardcoded"

# CWE-327: Weak cryptographic algorithms
import hashlib
def weak_hash(password):
    return hashlib.md5(password.encode()).hexdigest()

@app.route('/search')
def search_logs():
    query = request.args.get('q', '')
    # CWE-78: OS Command Injection
    cmd = f"grep {query} /var/log/app.log"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

@app.route('/file')
def read_file():
    filename = request.args.get('name', '')
    # CWE-22: Path Traversal  
    file_path = '/uploads/' + filename
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        return str(e)

@app.route('/admin')
def admin_panel():
    # CWE-89: SQL Injection
    user_id = request.args.get('id', '1')
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return str(results)

@app.route('/template')
def render_template():
    # CWE-94: Code Injection via template
    template = request.args.get('tmpl', 'Hello')
    return render_template_string(template)

if __name__ == '__main__':
    # CWE-489: Debug mode in production
    app.run(debug=True, host='0.0.0.0', port=5000)`,
    type: "code" as const
  }
};

export type DemoContentKey = keyof typeof demoContent;