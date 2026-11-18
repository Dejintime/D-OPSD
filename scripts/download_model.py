"""Download a pinned model snapshot, preserving partial files for retries."""
import argparse
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download


def verified_range_download(root):
    """Parallel mirror transfer, verified against the pinned HF blob SHA256."""
    import requests
    expected = "08faf1e950173bb44c11d26df28c55da7237dc55c77dbc1cbae4a5d7584540d9"
    size, block = 3087467144, 16 * 1024**2
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "model.safetensors"
    def digest(path):
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(8 * 1024**2), b""):
                sha.update(data)
        return sha.hexdigest()
    if target.exists() and target.stat().st_size == size and digest(target) == expected:
        return
    parts = root / ".download_parts"
    parts.mkdir(exist_ok=True)
    legacy = root / "model.safetensors.modelscope-part"
    if legacy.exists():
        with legacy.open("rb") as stream:
            for index in range(legacy.stat().st_size // block):
                data = stream.read(block)
                path = parts / str(index)
                if not path.exists():
                    path.write_bytes(data)
    url = "https://modelscope.cn/models/Qwen/Qwen2.5-Math-1.5B-Instruct/resolve/master/model.safetensors"
    count = (size + block - 1) // block
    def fetch(index):
        start, end = index * block, min(size, (index + 1) * block) - 1
        path = parts / str(index)
        if path.exists() and path.stat().st_size == end - start + 1:
            return index
        for attempt in range(6):
            try:
                with requests.get(url + f"?part={start}", headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                        raise ValueError("Mirror did not honor requested byte range")
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(1024**2):
                            stream.write(chunk)
                    if temporary.stat().st_size != end - start + 1:
                        raise ValueError("Incomplete range")
                    temporary.replace(path)
                    return index
            except (requests.RequestException, ValueError) as error:
                if attempt == 5:
                    raise RuntimeError(f"Failed block {index}: {error}") from error
                time.sleep(min(2**attempt, 16))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 8 == 0 or completed == count:
                print(f"Verified transfer blocks {completed}/{count}", flush=True)
    temporary = root / "model.safetensors.assembling"
    with temporary.open("wb") as output:
        for index in range(count):
            with (parts / str(index)).open("rb") as stream:
                for data in iter(lambda: stream.read(1024**2), b""):
                    output.write(data)
    if digest(temporary) != expected:
        raise ValueError("Weight SHA256 mismatch; refusing to install")
    temporary.replace(target)
    print("Weight SHA256 verified", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--revision", default="aafeb0fc6f22cbf0eaeed126eff8be45b0360a35")
    parser.add_argument("--local-dir", default="model/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--range-mirror", action="store_true", help="Known Qwen checkpoint only, SHA256-verified parallel transfer")
    args = parser.parse_args()
    if args.range_mirror:
        if args.repo != "Qwen/Qwen2.5-Math-1.5B-Instruct" or args.revision != "aafeb0fc6f22cbf0eaeed126eff8be45b0360a35":
            parser.error("Parallel mirror hashes only cover the default pinned checkpoint")
        verified_range_download(args.local_dir)
        path = args.local_dir
        required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        if not all(Path(path, name).exists() for name in required):
            raise ValueError("Missing model/tokenizer metadata: download the pinned snapshot metadata first")
    else:
        path = snapshot_download(args.repo, revision=args.revision, local_dir=args.local_dir,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
                                 max_workers=2)
    Path(path, "download_manifest.json").write_text(json.dumps({
        "repo": args.repo, "revision": args.revision, "status": "complete"}, indent=2))
    print(path)


if __name__ == "__main__":
    main()
