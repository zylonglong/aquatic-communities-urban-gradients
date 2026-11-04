from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


EOCD = b"PK\x05\x06"
CDH = b"PK\x01\x02"


def parse_tail(path: Path, total_size: int) -> dict:
    tail = path.read_bytes()
    tail_start = total_size - len(tail)
    eocd_pos = tail.rfind(EOCD)
    if eocd_pos < 0:
        raise RuntimeError("ZIP end-of-central-directory record was not found")
    if eocd_pos + 22 > len(tail):
        raise RuntimeError("Truncated ZIP EOCD")
    _, disk, cd_disk, n_disk, n_total, cd_size, cd_offset, comment_len = struct.unpack_from(
        "<4s4H2IH", tail, eocd_pos
    )
    if disk != 0 or cd_disk != 0 or n_disk != n_total:
        raise RuntimeError("Multi-disk ZIP archives are not supported")
    cd_pos = cd_offset - tail_start
    if cd_pos < 0 or cd_pos + cd_size > len(tail):
        raise RuntimeError(
            f"Central directory is outside downloaded tail; need range {cd_offset}-{cd_offset + cd_size - 1}"
        )
    members = []
    pos = cd_pos
    for _ in range(n_total):
        if tail[pos : pos + 4] != CDH:
            raise RuntimeError(f"Invalid central-directory signature at relative offset {pos}")
        fields = struct.unpack_from("<4s6H3I5H2I", tail, pos)
        (
            _, _made, _needed, flags, method, _mtime, _mdate,
            crc32, compressed_size, uncompressed_size,
            name_len, extra_len, member_comment_len, _disk_start, _internal_attr,
            _external_attr, local_header_offset,
        ) = fields
        name_raw = tail[pos + 46 : pos + 46 + name_len]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_raw.decode(encoding)
        members.append(
            {
                "name": name,
                "flags": flags,
                "method": method,
                "crc32": f"{crc32:08x}",
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_header_offset": local_header_offset,
            }
        )
        pos += 46 + name_len + extra_len + member_comment_len
    return {
        "archive_size": total_size,
        "central_directory_offset": cd_offset,
        "central_directory_size": cd_size,
        "comment_length": comment_len,
        "members": members,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tail", type=Path)
    parser.add_argument("--total-size", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = parse_tail(args.tail, args.total_size)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
