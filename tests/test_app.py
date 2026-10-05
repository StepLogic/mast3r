from tests.test_urdf import _make_model


def test_build_app_with_models(tmp_path):
    import gradio as gr
    from app_articulation import build_app
    _make_model(tmp_path / 'toy')
    app = build_app(str(tmp_path))
    assert isinstance(app, gr.Blocks)


def test_build_app_empty_root(tmp_path):
    import gradio as gr
    from app_articulation import build_app
    assert isinstance(build_app(str(tmp_path)), gr.Blocks)


def test_export_fn_produces_zip_and_check(tmp_path):
    from app_articulation import export_package
    from artsfm.model_io import load_model
    m = load_model(_make_model(tmp_path / 'toy'))
    zip_path, msg = export_package(m, extend=False)
    assert zip_path.endswith('.zip') and 'PASS' in msg
