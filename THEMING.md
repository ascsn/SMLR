# SMLR Documentation Theming Guide

This document explains the available documentation themes and how to switch between them.

## Available Themes

### 1. Default Theme
The original Material theme with a clean, professional look.
- **Colors**: Indigo primary, Deep Purple accent
- **Style**: Clean, minimal, professional
- **Best for**: Production, formal documentation

### 2. Neobrutalism Theme
A bold, visually striking theme with high-contrast colors and playful design.
- **Colors**: Purple, Pink, Yellow, Green accents
- **Style**: Bold borders, hard shadows, vibrant colors
- **Best for**: Making a statement, developer-friendly sites, projects that want to stand out

#### Neobrutalism Design Characteristics:
- **Hard offset shadows** (no blur/soft shadows)
- **Thick black borders** on interactive elements
- **High-contrast color palette** (cream, purple, pink, yellow, green)
- **Bold typography** with uppercase headings
- **Raw, unpolished aesthetic** that feels authentic
- **Playful hover effects** with transform animations

## How to Switch Themes

### Using the Script (Recommended)

We provide a convenience script to switch between themes:

```bash
# Show current theme and available options
./switch_theme.sh

# Switch to default theme
./switch_theme.sh default

# Switch to neobrutalism theme
./switch_theme.sh neobrutalism

# Check current status
./switch_theme.sh status
```

### Manual Switching

#### To enable Neobrutalism:
Add `neobrutalism.css` to your `mkdocs.yml`:

```yaml
extra_css:
  - stylesheets/extra.css
  - stylesheets/neobrutalism.css
```

#### To disable Neobrutalism:
Remove the neobrutalism line or restore from backup:

```bash
cp mkdocs.default.yml mkdocs.yml
```

## File Structure

```
SMLR/
├── mkdocs.yml              # Active configuration
├── mkdocs.default.yml      # Backup of default configuration
├── switch_theme.sh         # Theme switching script
└── docs/
    └── stylesheets/
        ├── extra.css           # Base custom styles
        ├── extra.default.css   # Backup of base styles
        └── neobrutalism.css    # Neobrutalism theme
```

## Customizing the Neobrutalism Theme

The neobrutalism theme uses CSS custom properties that can be easily modified:

```css
:root {
  /* Primary colors */
  --neo-black: #1a1a2e;
  --neo-white: #fffef7;
  --neo-cream: #fef6e4;
  
  /* Accent colors - modify these to change the color scheme */
  --neo-yellow: #ffd803;
  --neo-pink: #ff6b9d;
  --neo-purple: #9b5de5;
  --neo-blue: #00b4d8;
  --neo-green: #06d6a0;
  --neo-orange: #ff9f1c;
  --neo-red: #ef476f;
  
  /* Layout settings */
  --neo-border-width: 3px;
  --neo-shadow-offset: 5px;
}
```

### Color Scheme Variations

You can create different color moods by changing the accent colors:

#### Warm Palette
```css
--neo-yellow: #ffbe0b;
--neo-pink: #fb5607;
--neo-purple: #ff006e;
```

#### Cool Palette
```css
--neo-blue: #3a86ff;
--neo-purple: #8338ec;
--neo-pink: #ff006e;
```

#### Pastel Palette
```css
--neo-yellow: #fdffb6;
--neo-pink: #ffafcc;
--neo-purple: #bdb2ff;
--neo-green: #caffbf;
```

## Custom Components

The neobrutalism theme includes several custom CSS classes you can use in your Markdown:

### Hero Section
```html
<div class="hero">
  <h1>Welcome to SMLR</h1>
  <p>Fast, accurate emulation of strength functions</p>
</div>
```

### Feature Cards
```html
<div class="feature-card">
  <h3>Fast Emulation</h3>
  <p>Get results in milliseconds instead of hours.</p>
</div>
```

### Stat Boxes
```html
<div class="stat-box">
  <span class="number">1000x</span>
  <span class="label">Faster</span>
</div>
```

### Badges
```html
<span class="badge badge-success">New</span>
<span class="badge badge-warning">Beta</span>
<span class="badge badge-danger">Deprecated</span>
<span class="badge badge-info">Info</span>
```

## Dark Mode Support

The neobrutalism theme fully supports dark mode. It automatically adjusts:
- Background colors become darker
- Text colors invert for readability
- Shadow colors adjust to use light shadows on dark backgrounds
- Primary accent colors shift (purple → yellow)

Toggle dark mode using the sun/moon icon in the header.

## Troubleshooting

### Theme not applying?
1. Make sure `neobrutalism.css` is listed in `mkdocs.yml` under `extra_css`
2. Clear your browser cache (Cmd+Shift+R or Ctrl+Shift+R)
3. Restart the mkdocs server

### Styles look broken?
1. Check that `extra.css` is loaded before `neobrutalism.css`
2. Verify the CSS file paths are correct
3. Check browser console for 404 errors

### Want to modify specific elements?
The neobrutalism theme uses specific selectors. To override:
1. Add your custom CSS file after `neobrutalism.css` in `mkdocs.yml`
2. Or add custom styles directly to `extra.css`

## Development

To preview changes while developing themes:

```bash
# Start the dev server
mkdocs serve

# Or with uv
uv run mkdocs serve
```

The server will hot-reload when you save CSS changes.
