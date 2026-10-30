#!/usr/bin/env python3
"""Normalize the pinned publisher manifest without changing config or layer blobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

SOURCE = "ghcr.io/steveseguin/vllm-openai-xpu-qwen38-int4@sha256:ea61e69834d02b4abfe435eaaf56b2eda7b7b7c5ac78fffa3740779d8f27353a"
CONFIG = "sha256:2bf6dd4287e60cbe156f418fe4b25ff14209a61ba212053a3ee06d19cfc6daea"
DOCKER_GZIP = "application/vnd.docker.image.rootfs.diff.tar.gzip"
OCI_GZIP = "application/vnd.oci.image.layer.v1.tar+gzip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="containers-storage:localhost/b70-fp8-base:normalized")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="b70-fp8-manifest-", dir=args.work_dir) as directory:
        subprocess.run(["skopeo", "copy", "--preserve-digests", "docker://" + SOURCE, "dir:" + directory], check=True)
        path = Path(directory) / "manifest.json"
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == SOURCE.split("sha256:")[1], "Publisher manifest does not match pin"
        manifest = json.loads(raw)
        assert manifest["mediaType"] == "application/vnd.oci.image.manifest.v1+json"
        assert manifest["config"]["digest"] == CONFIG
        identities = [(layer["digest"], layer["size"]) for layer in manifest["layers"]]
        for layer in manifest["layers"]:
            assert layer["mediaType"] in (DOCKER_GZIP, OCI_GZIP), "Unexpected layer type"
            layer["mediaType"] = OCI_GZIP
        assert identities == [(layer["digest"], layer["size"]) for layer in manifest["layers"]]
        path.write_text(json.dumps(manifest, separators=(",", ":")))
        subprocess.run(["skopeo", "copy", "dir:" + directory, args.destination], check=True)
        print(json.dumps({"source": SOURCE, "config_digest": CONFIG, "layer_count": len(identities), "destination": args.destination, "change": "layer media types only; config and compressed layer digests unchanged"}), flush=True)


if __name__ == "__main__":
    main()
