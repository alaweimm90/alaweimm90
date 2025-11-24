# Custom Stationery Templates

A web-based application for creating customizable stationery templates including notebooks, journals, and planners with professional layouts and print optimization.

## Features

- **Notebook Templates**: Ruled, dotted, and blank pages
- **Journal Templates**: Dated and undated entries with decorative elements
- **Planner Templates**: Weekly and monthly calendar layouts
- **Multiple Themes**: Minimalist, floral, geometric, vintage, and modern
- **Customization Options**:
  - Color schemes (primary and secondary colors)
  - Custom headers and footers
  - Branding text
  - Page count adjustment
- **Print Optimization**: High-resolution printing with proper page breaks
- **Digital Preview**: Real-time preview of templates
- **Responsive Design**: Works on both desktop and mobile devices

## File Structure

```
templates/stationery/
├── index.html              # Main application interface
├── css/
│   ├── styles.css          # Main styling and layout
│   ├── themes.css          # Theme definitions and color schemes
│   └── print.css           # Print-specific optimizations
├── js/
│   └── app.js              # Application logic and interactivity
├── templates/
│   ├── notebook-ruled.html     # Ruled notebook template
│   ├── notebook-dotted.html    # Dotted notebook template
│   ├── notebook-blank.html     # Blank notebook template
│   ├── journal-dated.html      # Dated journal template
│   ├── journal-undated.html    # Undated journal template
│   ├── planner-weekly.html     # Weekly planner template
│   └── planner-monthly.html    # Monthly planner template
└── assets/
    ├── floral-border.svg       # Floral decorative elements
    └── geometric-pattern.svg   # Geometric decorative elements
```

## Usage

1. **Open the Application**: Open `index.html` in a web browser
2. **Select Template Type**: Choose from notebooks, journals, or planners
3. **Customize**: Adjust theme, colors, headers, footers, and page count
4. **Preview**: View the template in real-time
5. **Print or Download**: Use browser's print function to save as PDF

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Print Settings

For best results when printing:

- Use "Print to PDF" for digital versions
- Set paper size to A4 or Letter
- Enable background graphics/colors
- Set margins to default
- Use high-quality/color printing for themes with colors

## Customization

### Adding New Themes

1. Add theme variables to `css/themes.css`
2. Update the theme selector in `index.html`
3. Add theme-specific styles if needed

### Creating New Templates

1. Create new HTML file in `templates/` directory
2. Follow the existing template structure
3. Add URL parameter handling for customization
4. Update `js/app.js` to include the new template

## Technical Details

- **CSS Grid & Flexbox**: Modern layout techniques
- **CSS Custom Properties**: Dynamic theming
- **JavaScript ES6 Classes**: Organized application structure
- **Local Storage**: Settings persistence
- **Responsive Images**: SVG assets for crisp graphics at any size

## Development

To modify or extend the application:

1. Edit HTML files for structure changes
2. Modify CSS files for styling updates
3. Update JavaScript for new functionality
4. Test printing across different browsers

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

For questions or issues, please check the browser console for errors and ensure all files are properly loaded.