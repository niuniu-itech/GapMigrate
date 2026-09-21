#!/usr/bin/env python3
"""Copy four manuscript figures and render matching README previews.

This exports existing figure PDFs; it does not run or reconstruct experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

FIGURES={
    'example':('migration_choices','method'),
    'overview':('framework_overview','method'),
    'baseline':('tvm_latency_comparison','result'),
    'gain_yield':('qualified_gain_yield','result'),
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tex',type=Path,required=True)
    parser.add_argument('--version',required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'assets/paper')
    args=parser.parse_args()
    import fitz
    tex=args.tex.resolve(); text=tex.read_text(encoding='utf-8')
    text=text.split(r'\begin{document}',1)[1]
    found={}
    for number,match in enumerate(re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',text,re.S),1):
        block=match.group()
        label=re.search(r'\\label\{fig:([^}]+)\}',block)
        graphic=re.search(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',block)
        if label and graphic and label[1] in FIGURES:
            source=tex.parent/'figures'/graphic[1]
            if not source.is_file():raise FileNotFoundError(source)
            found[label[1]]=(number,source)
    if set(found)!=set(FIGURES):raise ValueError('Required figure labels missing from manuscript')
    args.output.mkdir(parents=True,exist_ok=True); records=[]
    for label,(name,kind) in FIGURES.items():
        number,source=found[label];dst=args.output/(name+'.pdf')
        shutil.copy2(source,dst)
        with fitz.open(source) as doc:
            if len(doc)!=1:raise ValueError('Each figure must be a one-page PDF')
            doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(args.output/(name+'.png'))
        records.append(dict(file=dst.name,source_figure=source.stem,paper_figure_number=number,
            source_label='fig:'+label,source_version=args.version,kind=kind,
            sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
            preview_sha256=hashlib.sha256((args.output/(name+'.png')).read_bytes()).hexdigest(),
            preview_method='PyMuPDF rendering of the same one-page PDF at 144 dpi'))
    (args.output/'provenance.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print('Synchronized four manuscript PDFs and matching PNG previews from '+args.version)

if __name__=='__main__':
    main()
