# PathSnap Live Telemetry & Map-Matching HUD — Design System

**Art Direction:** Liquid Glass & Cyberpunk Telemetry HUD  
**Inspiration:** High-performance aerospace telemetry, Autonomous vehicle perception decks, Dark glassmorphism.

---

## 1. Color Palette (60-30-10 Rule)

### Dominant (60% — Canvas & Atmospheric Surfaces)
- **Deep Space Black:** `#070a12` (Base application canvas)
- **Obsidian Glass:** `rgba(13, 19, 33, 0.72)` (Primary glass container fill)
- **Sub-surface Slate:** `#0f172a` (Secondary panels, table headers)

### Structural & Secondary (30% — Cards, Dividers, Neutral Content)
- **Card Background:** `rgba(18, 27, 46, 0.65)` (Translucent card backdrop with `backdrop-filter: blur(16px)`)
- **Border Specular Highlight:** `rgba(255, 255, 255, 0.09)` (Glass rim edge highlight)
- **Muted Telemetry Text:** `#94a3b8` (Labels, coordinates, metric names)
- **High-Contrast Text:** `#f8fafc` (Primary values, headings, titles)

### Accent & Signal (10% — Visual Hierarchy & Real-Time States)
- **Cyber Emerald:** `#10b981` (Model-predicted route, matched states, Top-1 accuracy)
- **Electric Cyan:** `#00f0ff` (Raw GPS track points, active vehicle beacon, selected states)
- **Solar Amber:** `#f59e0b` (Ground truth road route, warnings, candidate markers)
- **Crimson Alert:** `#ef4444` (Misclassification, off-road deviations)

---

## 2. Typography Hierarchy

- **Primary UI Sans:** `Inter`, system sans-serif (`-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `sans-serif`)
  - Headings: `font-weight: 700 / 800`, tracking: `-0.02em`
  - Body: `font-size: 14px`, `font-weight: 400`, line-height: `1.5`
- **Telemetry Monospace:** `JetBrains Mono`, `SF Mono`, `ui-monospace`, monospace
  - Metric numbers: `font-size: 20px / 28px`, `font-weight: 700`, tabular figures (`font-variant-numeric: tabular-nums`)
  - Coordinates & Arc IDs: `font-size: 12px`, uppercase, letter-spacing: `0.05em`

---

## 3. Elevation, Glassmorphism & Radii

```css
/* Glass Specular Panel */
.telemetry-card {
  background: rgba(13, 20, 36, 0.75);
  backdrop-filter: blur(16px) saturate(160%);
  -webkit-backdrop-filter: blur(16px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.38);
}

/* Radiant Beacon Pulse */
.beacon-cyan {
  box-shadow: 0 0 16px rgba(0, 240, 255, 0.5), inset 0 0 8px rgba(0, 240, 255, 0.8);
}
.beacon-emerald {
  box-shadow: 0 0 16px rgba(16, 185, 129, 0.5), inset 0 0 8px rgba(16, 185, 129, 0.8);
}
```

---

## 4. UI Icons & Visual Assets
- **Strict Rule:** Zero emojis used as UI icons.
- All icons rendered via clean, inline SVG vectors (Lucide icon style: Play, Pause, Navigation, Layers, Cpu, ShieldCheck, Activity, BarChart3, Crosshair, ChevronDown).

---

## 5. Motion & Interaction Tokens
- **Hover Transitions:** `all 0.2s cubic-bezier(0.4, 0, 0.2, 1)`
- **Map Polyline Animation:** Smooth dash-array progression on initial route load.
- **Micro-Interactions:** Subtle scale (`scale: 1.02`) and border luminance on card hover.
- **Accessibility:** `prefers-reduced-motion` respected across all GSAP and CSS animations.
