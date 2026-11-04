"""Plan and assemble selected members from a remote ZIP archive.

The archive's central-directory metadata and each member's local header are
downloaded separately.  This script calculates byte ranges for parallel HTTP
requests and reconstructs a DEFLATE-compressed member with CRC validation.
"""

from __future__ import annotations

import argparse
import binascii
import json
import struct
import zlib
from pathlib import Path


LOCAL_HEADER = struct.Struct("<4s5H3I2H")


def make_plan(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    members = manifest.get("members", manifest)
    member = next((m for m in members if m["name"] == args.member), None)
    if member is None:
        raise SystemExit(f"Member not found: {args.member}")

    header = Path(args.header).read_bytes()
    if len(header) < LOCAL_HEADER.size:
        raise SystemExit("Local-header download is too short")
    fields = LOCAL_HEADER.unpack_from(header)
    signature = fields[0]
    if signature != b"PK\x03\x04":
        raise SystemExit(f"Invalid local-header signature: {signature!r}")
    filename_len, extra_len = fields[-2:]
    encoded_name = header[LOCAL_HEADER.size : LOCAL_HEADER.size + filename_len]
    archive_name = encoded_name.decode("utf-8")
    if archive_name != args.member:
        raise SystemExit(f"Header member mismatch: {archive_name!r}")

    data_start = member["local_header_offset"] + LOCAL_HEADER.size + filename_len + extra_len
    compressed_size = member["compressed_size"]
    chunk_size = max(1, int(args.chunk_size_mb * 1024 * 1024))
    chunks = []
    start = data_start
    idx = 0
    while start < data_start + compressed_size:
        end = min(start + chunk_size - 1, data_start + compressed_size - 1)
        chunks.append({"index": idx, "start": start, "end": end, "size": end - start + 1})
        start = end + 1
        idx += 1

    plan = {
        "member": member,
        "data_start": data_start,
        "data_end": data_start + compressed_size - 1,
        "chunks": chunks,
    }
    Path(args.output).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({
        "member": args.member,
        "compressed_size": compressed_size,
        "uncompressed_size": member["uncompressed_size"],
        "data_start": data_start,
        "data_end": data_start + compressed_size - 1,
        "n_chunks": len(chunks),
    }))


def assemble(args: argparse.Namespace) -> None:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    member = plan["member"]
    part_dir = Path(args.part_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    crc = 0
    total_out = 0
    if member["method"] == 8:
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    elif member["method"] == 0:
        decompressor = None
    else:
        raise SystemExit(f"Unsupported ZIP method: {member['method']}")

    with output.open("wb") as dst:
        for chunk in plan["chunks"]:
            part = part_dir / f"part_{chunk['index']:03d}.bin"
            payload = part.read_bytes()
            if len(payload) != chunk["size"]:
                raise SystemExit(
                    f"Wrong size for {part.name}: {len(payload)} != {chunk['size']}"
                )
            decoded = decompressor.decompress(payload) if decompressor else payload
            if decoded:
                dst.write(decoded)
                crc = binascii.crc32(decoded, crc)
                total_out += len(decoded)
        if decompressor:
            decoded = decompressor.flush()
            dst.write(decoded)
            crc = binascii.crc32(decoded, crc)
            total_out += len(decoded)

    expected_crc = int(member["crc32"], 16)
    if total_out != member["uncompressed_size"]:
        raise SystemExit(f"Wrong output size: {total_out} != {member['uncompressed_size']}")
    if (crc & 0xFFFFFFFF) != expected_crc:
        raise SystemExit(f"CRC mismatch: {crc & 0xFFFFFFFF:08x} != {expected_crc:08x}")
    print(json.dumps({"output": str(output), "bytes": total_out, "crc32": f"{crc & 0xFFFFFFFF:08x}"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--member", required=True)
    plan.add_argument("--header", required=True)
    plan.add_argument("--chunk-size-mb", type=float, default=8.0)
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=make_plan)

    assemble_parser = sub.add_parser("assemble")
    assemble_parser.add_argument("--plan", required=True)
    assemble_parser.add_argument("--part-dir", required=True)
    assemble_parser.add_argument("--output", required=True)
    assemble_parser.set_defaults(func=assemble)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
