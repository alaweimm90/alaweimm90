# MEZAN Metrics Dashboard

A production-quality real-time monitoring dashboard for the MEZAN (Modular Enterprise Zonal Automation Network) system.

## Features

### Real-Time Monitoring
- **Live Metrics Updates**: Active jobs, throughput, success rate, and average response time
- **Performance Charts**: Time-series visualization of system performance
- **Solver Comparison**: Bar charts comparing different solver algorithms
- **System Health Gauges**: CPU, Memory, and Queue depth monitoring

### Visual Components
- **Animated Gauges**: SVG-based circular progress indicators
- **Interactive Charts**: Custom canvas-based line and bar charts
- **Progress Bars**: Visual job completion tracking
- **Status Badges**: Color-coded status indicators

### User Experience
- **Dark/Light Theme**: Toggle between dark and light modes (persisted in localStorage)
- **Responsive Design**: Adapts to desktop, tablet, and mobile viewports
- **Smooth Animations**: CSS transitions and animations for better UX
- **Real-time Updates**: Simulated WebSocket connection with automatic data refresh

## Usage

### Option 1: Direct Browser Access
Simply open the `dashboard.html` file directly in your web browser:
```bash
# From the command line
open dashboard.html  # macOS
xdg-open dashboard.html  # Linux
start dashboard.html  # Windows

# Or navigate to file in browser
file:///path/to/MEZAN/ORCHEX/ORCHEX-core/dashboard.html
```

### Option 2: Python Server
Use the included Python server for API endpoints and proper HTTP serving:
```bash
# Navigate to the directory
cd /home/user/AlaweinOS/MEZAN/ORCHEX/ORCHEX-core

# Run the server (default port 8080)
python serve_dashboard.py

# Or specify a custom port
python serve_dashboard.py --port 3000
```

Then open your browser to: `http://localhost:8080`

### Option 3: Integration with MEZAN API
The dashboard can be integrated with the existing MEZAN Flask API server:
```python
# In mezan_api.py, add:
@app.route('/dashboard')
def dashboard():
    return send_file('dashboard.html')
```

## Dashboard Components

### 1. Key Metrics Cards
- **Active Jobs**: Current number of running jobs
- **Throughput**: Jobs processed per second (K/s)
- **Success Rate**: Percentage of successful job completions
- **Avg Response**: Average response time in milliseconds

### 2. Performance Timeline
- Real-time line chart showing system performance over time
- Auto-updating every 2 seconds
- Gradient fill with animated data points

### 3. Solver Comparison
- Bar chart comparing different optimization solvers
- Includes: Genetic, Tabu, Simulated Annealing, Random, Local Search
- Color-coded bars with value labels

### 4. System Health Gauges
- **CPU Usage**: Circular gauge showing CPU utilization
- **Memory Usage**: Memory consumption indicator
- **Queue Depth**: Number of jobs in the queue

### 5. System Alerts
- Real-time alert notifications
- Color-coded by severity (success, warning, danger)
- Auto-scrolling with newest alerts at top
- Maximum 5 alerts displayed

### 6. Active Jobs Table
- Live updating table of current jobs
- Columns: Job ID, Type, Status, Progress, Duration, Agent
- Visual progress bars for job completion
- Hover effects for better readability

## Technical Details

### Architecture
- **Single Page Application**: All code embedded in one HTML file
- **No External Dependencies**: All CSS and JavaScript included inline
- **Canvas-based Charts**: Custom chart rendering without external libraries
- **SVG Gauges**: Scalable vector graphics for system metrics

### Update Intervals
- Metrics refresh: Every 2 seconds
- Alert generation: Random, ~30% chance every 5 seconds
- Connection status: Every 10 seconds
- Charts redraw: On data update and window resize

### Browser Compatibility
- Modern browsers (Chrome, Firefox, Safari, Edge)
- ES6+ JavaScript features used
- CSS Grid and Flexbox layout
- LocalStorage for theme persistence

## Customization

### Modify Update Frequency
In the JavaScript section, adjust the intervals:
```javascript
// Update metrics every 2 seconds (change to desired milliseconds)
setInterval(updateMetrics, 2000);

// Generate alerts (change frequency)
setInterval(() => {
    if (Math.random() > 0.7) {  // 30% chance
        generateAlert();
    }
}, 5000);  // Every 5 seconds
```

### Change Color Scheme
Modify the CSS variables in `:root`:
```css
:root {
    --accent-primary: #6366f1;  /* Primary accent color */
    --accent-secondary: #22d3ee;  /* Secondary accent */
    --accent-success: #10b981;   /* Success indicators */
    --accent-warning: #f59e0b;   /* Warning indicators */
    --accent-danger: #ef4444;    /* Error/danger indicators */
}
```

### Add New Metrics
Add new metric cards by duplicating the card structure:
```html
<div class="card">
    <div class="card-header">
        <span class="card-title">New Metric</span>
    </div>
    <div class="metric-value" id="newMetric">0</div>
    <div class="metric-change positive">↑ 0%</div>
</div>
```

## API Endpoints (when using serve_dashboard.py)

- `GET /` - Serves the dashboard HTML
- `GET /api/metrics` - Returns current metrics in JSON format
- `GET /api/jobs` - Returns active jobs list in JSON format

## Performance Considerations

- Canvas charts are hardware-accelerated
- CSS animations use `transform` for better performance
- Minimal DOM manipulation for updates
- Efficient data structures for real-time updates

## Security Notes

- No sensitive data displayed in demo mode
- CORS headers enabled for API endpoints (configure for production)
- No authentication in demo (add for production use)

## Future Enhancements

Potential improvements for production deployment:
1. WebSocket integration for true real-time updates
2. Historical data persistence and retrieval
3. User authentication and role-based access
4. Export functionality for charts and metrics
5. Advanced filtering and search for jobs table
6. Integration with actual MEZAN backend APIs
7. Configurable alert thresholds
8. Multi-dashboard support for different system components

## License

Part of the AlaweinOS MEZAN project under Apache 2.0 License

---

For questions or issues, contact: meshal@berkeley.edu