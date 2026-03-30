# SpindL GUI Design System

**Purpose:** Canonical reference for visual design decisions. Future sessions MUST adhere to these specifications. Do not deviate without explicit User approval.

---

## Color Palette

### Background
- **Primary background:** Pure black `oklch(0 0 0)` — NOT dark gray, NOT near-black, actual `#000000`
- **Card/elevated surfaces:** `oklch(0.12 0 0)` — subtle lift from pure black
- **Sidebar:** `oklch(0.08 0 0)` — slightly darker than cards

### Accent (Purple)
- **Primary purple:** `oklch(0.65 0.25 295)` — vibrant, saturated purple
- **Secondary purple:** `oklch(0.55 0.20 295)` — slightly muted for hover states
- **Sidebar accent:** `oklch(0.18 0.05 295)` — subtle purple tint for active states

### Text
- **Primary text:** `oklch(0.95 0 0)` — near-white
- **Muted text:** `oklch(0.65 0 0)` — for secondary information
- **On primary:** `oklch(0.98 0 0)` — text on purple buttons

### Borders
- **Default border:** `oklch(1 0 0 / 12%)` — subtle white at 12% opacity
- **Input border:** `oklch(1 0 0 / 15%)` — slightly more visible for inputs

### Status Colors
- **Success/healthy:** Green (use sparingly, for health indicators)
- **Warning:** Yellow/amber
- **Error/destructive:** `oklch(0.65 0.25 25)` — red-orange

---

## Typography

- **Font family:** Geist Sans (system default from Next.js)
- **Monospace:** Geist Mono (for code, transcriptions, technical data)
- **No custom fonts** — keep it system-native for performance

---

## Spacing & Layout

- **Sidebar width:** Fixed, not collapsible (unless explicitly requested)
- **Content padding:** Consistent with shadcn defaults
- **Card radius:** `0.625rem` base, scaled variants for different sizes
- **Dense information display** — this is a control panel, not a marketing site

---

## Component Guidelines

### Buttons
- Primary actions: Purple background, white text
- Secondary actions: Dark gray background, light text
- Destructive actions: Red-orange, use sparingly with confirmation

### Cards
- Subtle elevation from background
- No heavy shadows — rely on background color difference
- Border optional, use sparingly

### Status Indicators
- Small badges or dots for connection/health status
- Green = connected/healthy
- Red = disconnected/error
- Yellow = warning/degraded
- Gray = unknown/loading

### Data Display
- Progress bars for token usage (purple fill)
- Timestamps in muted text
- State names in badges
- Live text (transcription/response) in monospace

---

## What NOT To Do

1. **No light mode** — User explicitly prefers dark. No toggle needed.
2. **No gray backgrounds** — Pure black only for main background.
3. **No blue accents** — Purple is the brand color.
4. **No rounded-full buttons** — Keep consistent with shadcn defaults.
5. **No animations unless functional** — Subtle transitions okay, no decorative motion.
6. **No gradients** — Flat colors only.

---

## Reference Files

- Theme variables: `src/app/globals.css` (`:root` and `.dark` sections)
- Tailwind config: Uses CSS variables defined in globals.css
- shadcn components: `src/components/ui/` — do not modify base styles

---

*Document created to prevent style drift across sessions. If a future session suggests changing these values, request explicit User approval first.*
