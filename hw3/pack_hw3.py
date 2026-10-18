"""Build the small, credential-free upload archive for HW3."""

from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


root = Path(__file__).resolve().parent
source = root / "handout"
files = (
    "lora.py", "model.py", "dataloader.py", "train.py", "generate.py",
    "requirements.txt", "test_lora.py", "data.pt", "run_hw3.sh",
    "README_DEPLOY.md", "run_in_colab.ipynb",
)
target = root / "hw3_upload.zip"
temporary = root / "hw3_upload.new.zip"

with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(source / name, name)

with ZipFile(temporary) as archive:
    assert archive.namelist() == list(files)
    assert archive.testzip() is None

temporary.replace(target)
print(f"{target}: {target.stat().st_size} bytes, SHA256 {sha256(target.read_bytes()).hexdigest()}")
