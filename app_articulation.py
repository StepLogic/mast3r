#!/usr/bin/env python3
"""Gradio viewer for articulated models built by build_model.py / run_articulation.py --build_model."""
import argparse
import json
import os
import shutil
import tempfile
import time

import gradio as gr

from artsfm.model_io import load_model
from artsfm.urdf import check_urdf, write_urdf
from artsfm.viewer import (build_glb, info_markdown, list_models, range_update, slider_spec, sweep_values,
                           to_joint_value)


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
                slider = gr.Slider(minimum=0, maximum=1, value=0, label='Joint value')
                extend = gr.Checkbox(value=False, label='Allow beyond observed range')
                colour = gr.Radio(['RGB', 'By part'], value='RGB', label='Colour')
                hulls = gr.Checkbox(value=False, label='Show collision hulls')
                play = gr.Button('Play')
                export_btn = gr.Button('Export URDF')
                package = gr.File(label='URDF package')
                export_msg = gr.Markdown()
                info = gr.Markdown()
            with gr.Column(scale=2):
                view = gr.Model3D(label='Model', height=620, clear_color=[0.0, 0.0, 0.0, 0.0])

        def render(m, v, c, h):
            return build_glb(m, to_joint_value(m['joint'], v), 'part' if c == 'By part' else 'rgb', True, h)

        def load(path):
            m = load_model(path)
            lo, hi, unit = slider_spec(m['joint'])
            return (m, gr.update(minimum=lo, maximum=hi, value=0, label=f'Joint value ({unit})'),
                    render(m, 0, 'RGB', False), info_markdown(m, _result_for(path)))

        def set_range(m, e, v):
            lo, hi, v = range_update(m['joint'], e, v)
            return gr.update(minimum=lo, maximum=hi, value=v)

        def play_fn(m, c, h):
            for v in sweep_values(m['joint']):
                yield build_glb(m, v, 'part' if c == 'By part' else 'rgb', True, h)
                time.sleep(0.15)

        app.load(load, picker, [state, slider, view, info])
        picker.change(load, picker, [state, slider, view, info])
        slider.release(render, [state, slider, colour, hulls], view)
        colour.change(render, [state, slider, colour, hulls], view)
        hulls.change(render, [state, slider, colour, hulls], view)
        extend.change(set_range, [state, extend, slider], slider)
        play.click(play_fn, [state, colour, hulls], view)
        export_btn.click(export_package, [state, extend], [package, export_msg])
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='outputs/artsfm')
    p.add_argument('--port', type=int, default=7861)
    p.add_argument('--server_name', default='127.0.0.1')
    a = p.parse_args()
    build_app(a.root).launch(server_name=a.server_name, server_port=a.port)


if __name__ == '__main__':
    main()
