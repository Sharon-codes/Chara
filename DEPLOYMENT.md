# Chara Survival deployment

## GitHub

Create a public repository named `chara-survival`, then push the package, `app.py`, frozen model artifact, and benchmark scripts:

```bash
git init
git add chara app.py scripts setup.py pyproject.toml requirements.txt DEPLOYMENT.md
git commit -m "Initial Chara Survival release"
git branch -M main
git remote add origin https://github.com/<your-account>/chara-survival.git
git push -u origin main
```

## PyPI

Build from a clean checkout and publish the artifacts with a configured PyPI token:

```bash
python -m pip install --upgrade build twine
python setup.py sdist bdist_wheel
twine upload dist/*
```

## Hugging Face Spaces

Create a new Space with the **Gradio** SDK, connect it to the GitHub repository, and set the Space secrets needed for any private model artifacts. The Space entrypoint is `app.py`; keep `chara_model_4337.pkl` beside it or fetch it from a release asset during the build.
