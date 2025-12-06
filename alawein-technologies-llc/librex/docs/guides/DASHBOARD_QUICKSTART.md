# Librex.QAP Dashboard v2.0 - Quick Start Guide

## 🚀 Getting Started (2 minutes)

### Step 1: Install Dependencies
```bash
cd Librex.QAP-new
pip install -r requirements.txt
```

### Step 2: Start the API Server
```bash
python server.py
```
Keep this terminal running. You should see:
```
Librex.QAP-new API Server Starting
Documentation: http://localhost:8000/docs
```

### Step 3: Launch the Dashboard (New Terminal)
```bash
streamlit run dashboard.py
```

The dashboard will open automatically at: **http://localhost:8501**

---

## 🎯 First-Time User Walkthrough

### 1. Overview Page (🏠)
**What you'll see:**
- API connection status (should show ✅ Connected)
- Key metrics: optimizations count, quality, runtime
- Available optimization methods

**Try this:**
- Check that the API is connected
- Expand a method to see its details

### 2. Solve Problem Page (🔧)
**What to do:**
- Keep default settings (Problem Size: 20, Method: fft_laplace)
- Click "🚀 Solve Now"
- Watch the progress bar
- See your results!

**Your first optimization:**
```
Problem Size: 20
Method: FFT Laplace
Iterations: 500
Click: 🚀 Solve Now
```

**Expected result:** ~1-2 seconds completion time

### 3. Try Dark Mode (🌓)
**In the sidebar:**
1. Look for the Theme section
2. Click the 🌓 button
3. Enjoy the sleek dark interface!

---

## ⚡ Power User Tips

### Real-Time Updates
1. Sidebar → Auto-Refresh → Enable
2. Set interval to 5 seconds
3. Go to Analytics page
4. Watch live updates!

### Compare Methods
1. Go to Benchmarks page (🏃)
2. Select instance: "nug20"
3. Choose methods: fft_laplace, reverse_time, genetic_algorithm
4. Set runs: 5
5. Click "🏃 Run Benchmark"
6. Compare results in beautiful charts!

### Export Your Work
**After any optimization:**
- Click "📥 CSV" or "📥 JSON"
- Save results for later analysis
- Import into Excel, Python, or R

### Save for Comparison
1. After solving a problem
2. Click "➕ Add to Comparison History"
3. Go to History page (💾)
4. See all your saved results
5. Export everything with one click

---

## 🎨 Customization Quick Tips

### Change API URL
**Sidebar → Configuration → API Base URL**
```
Default: http://localhost:8000
Remote:  http://your-server:8000
```

### Adjust Auto-Refresh
**Sidebar → Auto-Refresh**
- Enable: ✓
- Interval: 1-60 seconds
- Perfect for monitoring long benchmarks!

---

## 🔧 Common Tasks

### Task 1: Run a Quick Test
```
1. Go to Methods page (⚙️)
2. Select any method tab
3. Scroll to "Quick Test"
4. Click "▶️ Run Test"
5. See instant results!
```

### Task 2: Benchmark Multiple Methods
```
1. Benchmarks page (🏃)
2. Instance: nug20
3. Methods: Select 3-4 methods
4. Runs: 5
5. Advanced Options → Visualization: All
6. Run and compare!
```

### Task 3: Track Your Work
```
1. Solve several problems
2. Save each to history
3. Go to History page
4. View comparison chart
5. Export all results
```

---

## 📊 Understanding the Pages

| Page | Icon | Purpose | Key Features |
|------|------|---------|--------------|
| Overview | 🏠 | System status | Metrics, health checks |
| Solve | 🔧 | Run optimizations | Configure & solve problems |
| Benchmarks | 🏃 | Compare methods | Multi-method analysis |
| Methods | ⚙️ | Explore algorithms | Method details & testing |
| Analytics | 📊 | Monitor performance | Historical trends, stats |
| History | 💾 | Review results | Saved comparisons |

---

## 🎯 Example Workflows

### Workflow 1: Quick Optimization
```
1. Solve Problem page
2. Adjust problem size (try 30)
3. Select method
4. Click Solve
5. Export results
Total time: < 1 minute
```

### Workflow 2: Method Comparison
```
1. Benchmarks page
2. Choose instance
3. Select 3+ methods
4. Set 10 runs each
5. Review statistics
6. Export Excel file
Total time: 2-3 minutes
```

### Workflow 3: Research Session
```
1. Enable auto-refresh
2. Run multiple optimizations
3. Save each to history
4. Go to History page
5. Analyze trends
6. Export complete dataset
Total time: 10-15 minutes
```

---

## 💡 Pro Tips

### Tip 1: Use Keyboard Shortcuts
- **Tab**: Navigate between fields
- **Enter**: Submit forms
- **Space**: Toggle checkboxes
- **Arrow Keys**: Adjust sliders

### Tip 2: Optimize Performance
- Disable auto-refresh when not needed
- Use smaller problem sizes for testing
- Reduce benchmark runs for quick checks

### Tip 3: Export Early, Export Often
- Save results as you go
- Use JSON for complete data
- Use CSV for spreadsheet analysis
- Use Excel for formatted reports

### Tip 4: Leverage Search & Filter
- Methods page: Search by keyword
- Filter by category
- Find the perfect algorithm quickly

### Tip 5: Monitor in Real-Time
- Enable auto-refresh
- Watch Analytics page
- Track system performance
- Identify trends

---

## ❓ Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot connect to API" | Start server with `python server.py` |
| Page is blank | Refresh browser (Ctrl+R or Cmd+R) |
| Dark mode stuck | Toggle theme button again |
| Export not working | Check browser download settings |
| Slow performance | Disable auto-refresh |
| Charts not showing | Check internet (for Plotly CDN) |

---

## 🎓 Learning Resources

### Understand the Algorithms
1. Go to Methods page (⚙️)
2. Read each method description
3. Check complexity analysis
4. Run quick tests
5. Compare performance

### Explore the Data
1. Run benchmarks
2. Download CSV exports
3. Open in Excel/Python
4. Create custom analyses
5. Share insights

---

## 🌟 Cool Features to Try

### 1. Dark Mode
Sidebar → Click 🌓 → Instant dark theme!

### 2. Interactive Charts
Hover over any chart point → See details!

### 3. Real-Time Updates
Enable auto-refresh → Live data streaming!

### 4. Multiple Export Formats
CSV, JSON, Excel → Choose your favorite!

### 5. Custom Problem Input
Advanced Options → Paste your own matrix!

### 6. Method Testing
Methods page → Quick Test → Instant results!

---

## 📱 Mobile & Tablet Support

The dashboard is responsive! Access from:
- Desktop (best experience)
- Laptop
- Tablet (works well)
- Mobile (limited, but functional)

---

## 🎉 You're Ready!

**Start exploring with confidence:**
1. ✅ Dashboard is running
2. ✅ API is connected
3. ✅ You know the basics
4. ✅ Ready to optimize!

**Next Steps:**
- Run your first optimization
- Try dark mode
- Compare some methods
- Export your results

**Happy Optimizing! 🚀**

---

## 💬 Need Help?

- Check the full guide: `DASHBOARD_V2_FEATURES.md`
- API Documentation: http://localhost:8000/docs
- GitHub Issues: [Report bugs](https://github.com/AlaweinOS/AlaweinOS/issues)

---

**Librex.QAP Dashboard v2.0** - Professional optimization made simple
