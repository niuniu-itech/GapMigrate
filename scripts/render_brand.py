"""Render original path-based branding. Optional PNG export requires PyMuPDF."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/brand"
OUT.mkdir(parents=True, exist_ok=True)
PURPLE, CORAL, NAVY = "#7443A8", "#ED7055", "#213343"

# Original geometric lettering; no external font outlines or raster images.
GLYPHS = {
    "G": (64, 'M53 17 C43 4 20 4 11 19 C2 35 6 59 24 65 C38 70 53 62 53 48 V40 H34'),
    "a": (54, 'M43 33 C33 20 9 24 9 45 C9 68 43 71 43 45 M43 28 V66'),
    "p": (55, 'M10 29 V87 M10 45 C10 19 45 21 45 45 C45 70 10 69 10 45'),
    "M": (72, 'M9 66 V9 L35 40 L61 9 V66'),
    "i": (26, 'M13 30 V66'),
    "g": (56, 'M44 32 C29 19 9 27 9 46 C9 67 43 69 43 44 M44 28 V70 C44 91 18 96 10 80'),
    "r": (43, 'M10 66 V29 M10 44 C10 30 23 24 35 29'),
    "t": (43, 'M19 12 V52 C19 62 23 66 35 66 M6 30 H35'),
    "o": (55, 'M46 46 C46 20 9 20 9 46 C9 73 46 73 46 46'),
    "e": (54, 'M10 44 H44 C44 20 9 21 9 45 C9 67 32 74 44 60'),
    "n": (56, 'M10 66 V29 M10 43 C10 21 45 23 45 43 V66'),
}


def icon(dark=False, mono=False):
    ink = "#F4F1ED" if dark else NAVY
    purple, coral = (ink, ink) if mono else (PURPLE, CORAL)
    return f'''<g fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path d="M86 37 H57 Q24 37 24 70 V112 Q24 145 57 145 H86 V105 H64" stroke="{purple}" stroke-width="14"/>
    <path d="M136 145 H163 Q196 145 196 112 V70 Q196 37 163 37 H136" stroke="{ink}" stroke-width="14"/>
    <path d="M77 79 H140" stroke="{coral}" stroke-width="11"/>
    <path d="M126 63 L145 79 L126 95" stroke="{coral}" stroke-width="11"/>
    <rect x="100" y="112" width="18" height="18" rx="4" fill="{coral}" stroke="none"/>
    </g>'''


def wordmark(dark=False, mono=False):
    x = 0
    items = []
    for index, char in enumerate("GapMigrate"):
        advance, path = GLYPHS[char]
        ink = "#F4F1ED" if dark else NAVY
        color = ink if mono or index >= 3 else PURPLE
        items.append(f'<g transform="translate({x},0)"><path d="{path}" fill="none" stroke="{color}" stroke-width="6.8" stroke-linecap="round" stroke-linejoin="round"/>')
        if char == "i":
            items.append(f'<circle cx="13" cy="12" r="4.2" fill="{ink if mono else CORAL}"/>')
        items.append('</g>')
        x += advance + 3
    return ''.join(items), x


def svg(body, width, height, title):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>{title}</title>{body}</svg>'


for dark, filename in [(False, 'logo.svg'), (True, 'logo-dark.svg')]:
    letters, width = wordmark(dark)
    body = '<g transform="translate(5,14)">' + icon(dark) + '</g>'
    body += '<g transform="translate(255,56) scale(1.18)">' + letters + '</g>'
    (OUT/filename).write_text(svg(body, round(255 + width * 1.18 + 15), 210, 'GapMigrate - transfer structure across the gap'), encoding='utf-8')
(OUT/'icon.svg').write_text(svg(icon(),220,182,'GapMigrate transfer icon'),encoding='utf-8')
(OUT/'icon-mono.svg').write_text(svg(icon(mono=True),220,182,'GapMigrate monochrome transfer icon'),encoding='utf-8')
letters,width=wordmark()
(OUT/'wordmark.svg').write_text(svg('<g transform="translate(5,5)">'+letters+'</g>',width+10,105,'GapMigrate custom wordmark'),encoding='utf-8')
try:
    import fitz
    for name in ['logo','icon','wordmark']:
        doc=fitz.open(OUT/(name+'.svg'));pdf=fitz.open('pdf',doc.convert_to_pdf())
        pdf[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=True).save(OUT/(name+'.png'))
    body='<rect width="1120" height="420" rx="24" fill="#FBF7F1"/>'
    body+='<g transform="translate(40,10)">'+icon()+'</g>'
    body+='<g transform="translate(290,58) scale(1.18)">'+letters+'</g>'
    body+='<path d="M55 227 H1065" stroke="#DED5CA"/>'
    body+='<text x="58" y="283" font-family="sans-serif" font-size="22" fill="#213343">PRESERVE  /  ADAPT  /  TRANSFER</text>'
    body+='<g transform="translate(845,248) scale(.68)">'+icon(mono=True)+'</g>'
    preview=svg(body,1120,420,'GapMigrate brand preview')
    doc=fitz.open(stream=preview.encode(),filetype='svg');pdf=fitz.open('pdf',doc.convert_to_pdf())
    pdf[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/'brand-preview.png')
except ImportError:
    print('SVG files ready. Install PyMuPDF to render PNG assets.')
