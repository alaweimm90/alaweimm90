# Live It Iconic - Logo Design Specifications

**Version:** 1.0
**Date:** November 2024
**Status:** Production Ready

---

## Logo Files Overview

This document provides complete technical specifications for creating all Live It Iconic logo variants. These specifications should be used by graphic designers to create the actual logo files.

---

## 1. Primary Wordmark Logo

### Design Specifications

**Typography:**
- Font: Montserrat Bold (Weight: 700)
- Text: "LIVE IT ICONIC" (all uppercase)
- Letter spacing: 0.15em (150 tracking in Adobe apps)
- Baseline alignment: Optical, not metric

**Proportions:**
```
Total width: 100 units (proportional)
Letter heights:
- L: 10 units tall
- I: 10 units tall (same as L)
- V, E, T: 10 units tall
- Space between words: 2.5 units

Minimum clear space around logo: 2 units on all sides
```

**Color Variations:**

1. **Primary Black** (`LII_Logo_Primary_Black.svg`)
   - Fill: #0A0A0A
   - Use on: White/light backgrounds

2. **Primary White** (`LII_Logo_Primary_White.svg`)
   - Fill: #FFFFFF
   - Use on: Dark backgrounds, photography

3. **Primary Gold** (`LII_Logo_Primary_Gold.svg`)
   - Fill: #D4AF37
   - Use on: Premium materials, special occasions

**Export Formats:**
```
SVG (vector):
- LII_Logo_Primary_Black.svg
- LII_Logo_Primary_White.svg
- LII_Logo_Primary_Gold.svg

PNG (raster, 300 DPI):
- Sizes: 500px, 1000px, 2000px, 4000px width
- LII_Logo_Primary_Black_[width]px.png
- LII_Logo_Primary_White_[width]px.png
- LII_Logo_Primary_Gold_[width]px.png

PDF (print-ready):
- LII_Logo_Primary_Black_Print.pdf (CMYK)
- LII_Logo_Primary_White_Print.pdf (CMYK)
```

**Usage Guidelines:**
- Minimum width: 120px digital, 1.5" print
- Never stretch or compress disproportionately
- Never apply effects (shadows, gradients, 3D)
- Never outline the letters
- Never change letter spacing

---

## 2. Stacked Logo

### Design Specifications

**Layout:**
```
LIVE IT
ICONIC

Line 1: "LIVE IT"
Line 2: "ICONIC"
Both lines: Montserrat Bold, same size
Center-aligned
Line spacing: 0.8em (tight but readable)
```

**Proportions:**
```
Width: 60 units
Height: 22 units (approximately)
Each line height: 10 units
Gap between lines: 2 units

Minimum clear space: 2 units on all sides
```

**Color Variations:**
- Same as Primary Wordmark (Black, White, Gold)

**File Names:**
```
LII_Logo_Stacked_Black.svg
LII_Logo_Stacked_White.svg
LII_Logo_Stacked_Gold.svg

(Plus PNG at 500px, 1000px, 2000px, 4000px widths)
```

**Usage Guidelines:**
- Use when: Square format required (social media avatars, small spaces)
- Minimum width: 80px digital, 1" print
- Best for: Instagram profile, app icons, tight layouts

---

## 3. Monogram (LII)

### Design Specifications

**Typography:**
- Font: Montserrat Bold
- Letters: "LII" (all uppercase)
- Letter spacing: 0.1em (tighter than wordmark)
- Optical kerning

**Optional Design Enhancement:**
- Consider connecting the three letters with a subtle ligature
- OR place inside a subtle circular or square frame (2pt stroke)

**Proportions:**
```
Width: 25 units
Height: 10 units
Letter width: ~7 units each
Gaps: ~1 unit between letters

If framed:
Frame diameter/width: 15 units
Frame stroke: 0.5 units
Logo centered within frame
```

**Color Variations:**
```
LII_Logo_Monogram_Black.svg
LII_Logo_Monogram_White.svg
LII_Logo_Monogram_Gold.svg

Special variations:
LII_Logo_Monogram_GoldOnBlack.svg (gold letters on black background)
LII_Logo_Monogram_BlackOnGold.svg (black letters on gold background)
```

**Export Formats:**
- SVG (vector)
- PNG: 256px, 512px, 1024px, 2048px (square format)
- ICO: 16px, 32px, 64px, 128px, 256px (for favicons)

**Usage Guidelines:**
- Use when: Very small space, favicon, product tags, subtle branding
- Minimum size: 32px digital, 0.5" print
- Best for: Favicon, app icon, embroidered patches, subtle watermarks

---

## 4. Icon Mark (Stylized "I")

### Design Specifications

**Concept:**
Create an elegant, minimalist icon based on the letter "I" with sophisticated serif details.

**Design Elements:**
```
Base: Vertical stem (rectangle)
- Width: 2 units
- Height: 10 units

Top serif: Elegant triangular cap
- Width at base: 6 units
- Height: 2 units
- Centered on stem
- Slight curve or angle for elegance

Bottom serif: Horizontal base
- Width: 5 units
- Height: 1 unit
- Centered on stem

Optional: Subtle taper in the stem (narrower at center, wider at ends)
```

**Proportions:**
```
Total width: 6 units
Total height: 13 units (including serifs)
Stem: 2 units wide × 10 units tall
Top serif: 6 units wide × 2 units tall
Bottom base: 5 units wide × 1 unit tall

Clear space: 2 units on all sides
```

**Color Variations:**
```
LII_Logo_IconMark_Black.svg
LII_Logo_IconMark_White.svg
LII_Logo_IconMark_Gold.svg

Special variations:
LII_Logo_IconMark_BlackGoldGradient.svg (subtle gradient for premium use)
```

**Export Formats:**
- SVG (vector)
- PNG: 256px, 512px, 1024px, 2048px (square canvas)
- ICO: 16px, 32px, 64px, 128px, 256px
- Apple Touch Icon: 180px × 180px

**Usage Guidelines:**
- Use when: Brand needs to be subtle, modern, iconic
- Best for: Social media profile pictures, app icons, loading animations
- Works well: Animated (can pulse, fade, or elegantly reveal)
- Minimum size: 24px digital, 0.25" print

---

## 5. Color Specifications (Technical)

### Iconic Black
```
Hex: #0A0A0A
RGB: 10, 10, 10
CMYK: 0, 0, 0, 96
Pantone: Black 6 C
```

### Iconic Gold
```
Hex: #D4AF37
RGB: 212, 175, 55
CMYK: 0, 17, 74, 17
Pantone: 7501 C (closest match)
```

### Pure White
```
Hex: #FFFFFF
RGB: 255, 255, 255
CMYK: 0, 0, 0, 0
```

---

## 6. File Structure

```
.brand/
└── assets/
    └── logos/
        ├── svg/
        │   ├── primary/
        │   │   ├── LII_Logo_Primary_Black.svg
        │   │   ├── LII_Logo_Primary_White.svg
        │   │   └── LII_Logo_Primary_Gold.svg
        │   ├── stacked/
        │   │   ├── LII_Logo_Stacked_Black.svg
        │   │   ├── LII_Logo_Stacked_White.svg
        │   │   └── LII_Logo_Stacked_Gold.svg
        │   ├── monogram/
        │   │   ├── LII_Logo_Monogram_Black.svg
        │   │   ├── LII_Logo_Monogram_White.svg
        │   │   └── LII_Logo_Monogram_Gold.svg
        │   └── icon/
        │       ├── LII_Logo_IconMark_Black.svg
        │       ├── LII_Logo_IconMark_White.svg
        │       └── LII_Logo_IconMark_Gold.svg
        ├── png/
        │   ├── primary/ (500px, 1000px, 2000px, 4000px)
        │   ├── stacked/ (500px, 1000px, 2000px, 4000px)
        │   ├── monogram/ (256px, 512px, 1024px, 2048px)
        │   └── icon/ (256px, 512px, 1024px, 2048px)
        ├── favicon/
        │   ├── favicon.ico (multi-size)
        │   ├── favicon-16x16.png
        │   ├── favicon-32x32.png
        │   ├── apple-touch-icon.png (180×180)
        │   └── android-chrome-512x512.png
        └── print/
            ├── LII_Logo_Primary_Black_Print.pdf (CMYK)
            ├── LII_Logo_Primary_White_Print.pdf (CMYK)
            └── LII_Logo_Primary_Gold_Print.pdf (CMYK)
```

---

## 7. Usage Matrix

| Context | Logo Type | Color | Size |
|---------|-----------|-------|------|
| Website header | Primary Wordmark | Black | 180px width |
| Website header (dark mode) | Primary Wordmark | White | 180px width |
| Favicon | Monogram | Black | 32×32px |
| Instagram profile | Stacked | Black or White | 320×320px |
| YouTube channel art | Primary Wordmark | White (on photo) | 2560px width |
| Business card | Stacked or Monogram | Gold | 1" width |
| T-shirt | Primary Wordmark | White | 10" width |
| Cap embroidery | Monogram | Black thread | 1.5" width |
| Email signature | Primary Wordmark | Black | 150px width |
| Loading spinner | Icon Mark | Gold | 64px |
| Product tag | Monogram | Gold foil | 0.75" |
| Letterhead | Primary Wordmark | Black | 3" width |

---

## 8. Logo Don'ts (Never Do These)

❌ Never stretch or compress disproportionately
❌ Never rotate at odd angles
❌ Never apply drop shadows or outer glows
❌ Never place on busy backgrounds without sufficient contrast
❌ Never use colors outside the brand palette
❌ Never rasterize logos below minimum sizes
❌ Never add effects (emboss, bevel, 3D)
❌ Never outline the letters
❌ Never change the font
❌ Never rearrange the letters
❌ Never use low-resolution files for print

---

## 9. Designer Instructions

### Creating the Files

**Software:**
- Adobe Illustrator (preferred)
- Affinity Designer (acceptable)
- Figma (acceptable for web assets)

**Process:**

1. **Set up artboard:**
   - Units: 100 × 100 (proportional)
   - Color mode: RGB for digital, CMYK for print

2. **Type the wordmark:**
   - Font: Montserrat Bold
   - Text: "LIVE IT ICONIC"
   - Size: Scale to fit artboard (approximately 10 units tall)
   - Tracking: +150

3. **Convert to outlines:**
   - Type → Create Outlines
   - This ensures font compatibility

4. **Clean up paths:**
   - Remove unnecessary anchor points
   - Ensure smooth curves
   - Optical alignment adjustments

5. **Create color variations:**
   - Duplicate artboards
   - Change fill colors
   - Ensure proper color profiles

6. **Export:**
   - SVG: Save as SVG, optimize for web
   - PNG: Export at specified sizes, 300 DPI, transparent background
   - PDF: Save as PDF/X-4 for print

### Quality Checklist

- [ ] All paths are closed and clean
- [ ] No stray points or hidden elements
- [ ] Text converted to outlines
- [ ] Colors match specifications exactly
- [ ] Clear space preserved in all exports
- [ ] File names follow naming convention
- [ ] Transparent backgrounds (except where specified)
- [ ] All sizes exported correctly
- [ ] SVG code is clean (no unnecessary metadata)
- [ ] Print files are CMYK, digital files are RGB

---

## 10. Approval Process

Before finalizing logo files:

1. **Design Review:** Ensure all proportions match specifications
2. **Color Accuracy:** Verify hex codes match exactly
3. **Size Testing:** Test at minimum and maximum sizes
4. **Context Testing:** Place on various backgrounds (white, black, photo)
5. **Format Testing:** Verify all file formats open correctly
6. **Print Testing:** Output at actual size to check quality

**Approval Required From:** Brand manager, founder

---

## 11. Future Considerations

As the brand evolves, consider creating:

- **Animated logo:** For video intros/outros
- **Seasonal variations:** Special editions for collaborations
- **Pattern versions:** Logo as repeating pattern for packaging
- **Outlined versions:** Outline-only for special applications
- **Embossed/debossed specs:** For leather, metal products

---

## Contact

**Questions about logo usage or creation?**
Email: brand@liveiticonic.com

**File requests:**
Request logo files from the brand assets library.

---

**This document should be used by graphic designers to create the actual logo files according to these specifications.**
