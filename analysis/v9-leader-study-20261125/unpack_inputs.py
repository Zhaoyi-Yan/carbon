"""Verify and restore the two frozen public replay cohorts. No policy execution."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath


def digest(data):
    return hashlib.sha256(data).hexdigest()


def extract(data, expected, target):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Archive member list differs from its frozen manifest")
        verified = []
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unexpected archive entry: {member.name}")
            content = archive.extractfile(member).read()
            if digest(content) != expected[member.name]:
                raise ValueError(f"SHA-256 mismatch: {member.name}")
            verified.append((path, content))
    target.mkdir(parents=True)
    for path, content in verified:
        destination = target.joinpath(*path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return len(verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New directory for current/ and history/")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    if args.destination.exists():
        parser.error("destination must be a new directory")
    parts = json.loads((source / "archive_parts.json").read_text())
    buffers = []
    for part in parts["parts"]:
        content = (source / part["file"]).read_bytes()
        if len(content) != part["bytes"] or digest(content) != part["sha256"]:
            raise ValueError(f"Part checksum mismatch: {part['file']}")
        buffers.append(content)
    current = b"".join(buffers)
    if len(current) != parts["archive_bytes"] or digest(current) != parts["archive_sha256"]:
        raise ValueError("Combined current archive checksum mismatch")
    current_meta = json.loads((source / "archive.json").read_text())
    if digest(current) != current_meta["sha256"]:
        raise ValueError("Current manifests disagree")
    history_meta = json.loads((source / "history_archive.json").read_text())
    history = (source / history_meta["file"]).read_bytes()
    if len(history) != history_meta["size"] or digest(history) != history_meta["sha256"]:
        raise ValueError("Historical archive checksum mismatch")
    current_count = extract(current, current_meta["member_sha256"], args.destination / "current")
    historical_count = extract(history, {m["file"]: m["sha256"] for m in history_meta["members"]}, args.destination / "history")
    print(f"Verified and restored {current_count} current and {historical_count} historical files to {args.destination}")


if __name__ == "__main__":
    main()
