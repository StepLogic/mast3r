#!/usr/bin/env python3
"""Gradio viewer for articulated models built by build_model.py / run_articulation.py --build_model."""
import argparse
import json
import os
import shutil
import tempfile

import gradio as gr

from artsfm.model_io import load_model
from artsfm.urdf import check_urdf, write_urdf
from artsfm.viewer import info_markdown, list_models
from artsfm.webviewer import PAGE_VERSION, build_html

VIEWER_DIR = os.path.join(tempfile.gettempdir(), 'artsfm_viewer')   # served to the iframe via allowed_paths


def _result_for(model_dir):
    path = os.path.join(os.path.dirname(model_dir), 'result.json')
    return json.load(open(path)) if os.path.isfile(path) else None


def export_package(model, extend=False):
    """Write the URDF (observed or extended limits), run the MuJoCo check, zip the model folder."""
    j = model['joint']
    lower = upper = None
    if extend:
        r = j['upper'] - j['lower']
        lower, upper = j['lower'] - 0.5 * r, j['upper'] + 0.5 * r
    try:
        urdf = write_urdf(model['dir'], lower=lower, upper=upper)
    except ValueError as e:
        return None, f'URDF export refused: {e}'
    err = check_urdf(urdf)
    msg = f'Wrote {os.path.basename(urdf)}. MuJoCo check: max relative error {err:.1e} ({"PASS" if err < 1e-6 else "FAIL"}).'
    base = os.path.join(tempfile.mkdtemp(), model.get('object', 'model'))
    return shutil.make_archive(base, 'zip', model['dir']), msg


def viewer_iframe(model):
    """Write the model's three.js page and return an iframe that loads it from the gradio file route."""
    os.makedirs(VIEWER_DIR, exist_ok=True)
    stamp = int(os.path.getmtime(os.path.join(model['dir'], 'model.json')))
    path = os.path.join(VIEWER_DIR, f"{model.get('object', 'model')}_{stamp}_{PAGE_VERSION}.html")
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            f.write(build_html(model))
    return (f'<iframe src="/gradio_api/file={path}" title="Articulated model" '
            'style="width:100%;height:680px;border:0;border-radius:8px"></iframe>')


def build_app(root='outputs/artsfm'):
    models = list_models(root)
    with gr.Blocks(title='Articulated model viewer') as app:
        gr.Markdown('## Articulated model viewer\nMove the recovered joint of a reconstructed object, '
                    'then export it as a URDF for simulation.')
        if not models:
            gr.Markdown(f'No articulated models found under `{root}`. Build one with '
                        '`python3 build_model.py --out_dir <run dir>` or '
                        '`python3 run_articulation.py ... --build_model`.')
            return app
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                picker = gr.Dropdown(choices=models, value=models[0], label='Model')
                extend = gr.Checkbox(value=False, label='URDF limits beyond observed range')
                export_btn = gr.Button('Export URDF')
                package = gr.File(label='URDF package')
                export_msg = gr.Markdown()
                info = gr.Markdown()
            with gr.Column(scale=2):
                view = gr.HTML()

        def load(path):
            m = load_model(path)
            return m, viewer_iframe(m), info_markdown(m, _result_for(path))

        app.load(load, picker, [state, view, info])
        picker.change(load, picker, [state, view, info])
        export_btn.click(export_package, [state, extend], [package, export_msg])
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='outputs/artsfm')
    p.add_argument('--port', type=int, default=7861)
    p.add_argument('--server_name', default='127.0.0.1')
    a = p.parse_args()
    build_app(a.root).launch(server_name=a.server_name, server_port=a.port, allowed_paths=[VIEWER_DIR])


if __name__ == '__main__':
    main()
