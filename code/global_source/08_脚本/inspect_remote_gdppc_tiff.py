"""Inspect a remote TIFF using HTTP byte ranges without downloading all data."""

from __future__ import annotations

import io
import json
import subprocess
from collections import OrderedDict

import tifffile


URL = "https://zenodo.org/api/records/18429133/files/rast_adm2_gdp_perCapita_1990_2024.tif/content"
SIZE = 130266414


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int, block_size: int = 1 << 16, max_blocks: int = 24):
        self.url = url
        self.size = size
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self.position + offset
        elif whence == io.SEEK_END:
            new = self.size + offset
        else:
            raise ValueError(whence)
        self.position = max(0, min(self.size, int(new)))
        return self.position

    def _block(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block_size
        end = min(self.size - 1, start + self.block_size - 1)
        completed = subprocess.run(
            [
                "curl.exe", "-L", "--silent", "--show-error", "--fail",
                "--retry", "4", "--retry-delay", "2",
                "-r", f"{start}-{end}", self.url,
            ],
            check=True,
            stdout=subprocess.PIPE,
        )
        data = completed.stdout
        expected = end - start + 1
        if len(data) != expected:
            raise IOError(f"Range {start}-{end}: expected {expected} bytes, received {len(data)}")
        self.cache[index] = data
        self.cache.move_to_end(index)
        while len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(int(size), self.size - self.position)
        if size <= 0:
            return b""
        start = self.position
        stop = start + size
        chunks = []
        cursor = start
        while cursor < stop:
            block_index = cursor // self.block_size
            block = self._block(block_index)
            block_start = block_index * self.block_size
            local_start = cursor - block_start
            local_stop = min(len(block), stop - block_start)
            chunks.append(block[local_start:local_stop])
            cursor = block_start + local_stop
        self.position = stop
        return b"".join(chunks)

    def close(self) -> None:
        super().close()


def main() -> None:
    reader = HTTPRangeReader(URL, SIZE)
    with tifffile.TiffFile(reader) as tif:
        page = tif.pages[0]
        summary = {
            "n_pages": len(tif.pages),
            "series": [{"shape": series.shape, "axes": series.axes, "dtype": str(series.dtype)} for series in tif.series],
            "page_shape": page.shape,
            "page_axes": page.axes,
            "dtype": str(page.dtype),
            "compression": str(page.compression),
            "is_tiled": bool(page.is_tiled),
            "rows_per_strip": int(page.rowsperstrip),
            "samples_per_pixel": int(page.samplesperpixel),
            "planar_config": str(page.planarconfig),
            "n_offsets": len(page.dataoffsets),
            "first_offsets": list(map(int, page.dataoffsets[:5])),
            "last_offsets": list(map(int, page.dataoffsets[-5:])),
            "first_bytecounts": list(map(int, page.databytecounts[:5])),
            "last_bytecounts": list(map(int, page.databytecounts[-5:])),
            "description": page.description,
            "cached_blocks": list(reader.cache.keys()),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
