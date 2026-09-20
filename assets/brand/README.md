# GapMigrate brand assets

The icon depicts a transfer across a gap: an open purple source structure, a coral handoff arrow and an open destination structure. The small detached block represents the part being adapted. The custom GapMigrate wordmark uses original rounded geometric paths, with coral dots linking it visually to the transfer mark.

- `logo.svg` — horizontal lockup for light backgrounds, transparent.
- `logo-dark.svg` — horizontal lockup for dark backgrounds, transparent.
- `icon.svg` — standalone transfer icon.
- `icon-mono.svg` — single-color icon for monochrome printing.
- `wordmark.svg` — custom path-based lettering without the icon.
- PNG equivalents and `brand-preview.png` — convenient previews.

All SVG artwork is editable vector geometry, not an embedded bitmap or an external font dependency. Keep the aspect ratio and leave at least one arrow-stroke width of clear space around the icon. Purple marks retained structure; coral marks adaptation/transfer; dark ink anchors the destination and wordmark.

Regenerate using `python scripts/render_brand.py`. PNG output additionally requires PyMuPDF. The project has not selected a redistribution license for code or artwork.
