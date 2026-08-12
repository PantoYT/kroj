#!/usr/bin/env python3
"""
Kroj — a TrueType font engine written from scratch on the Python standard library.

Parses .ttf/.ttc/.otf(glyf) files, resolves outlines (including composite glyphs),
maps characters through cmap, applies kerning from either `kern` or GPOS, flattens
quadratic Beziers, rasterizes with analytic horizontal coverage + vertical
supersampling, and writes PNG files by hand (zlib + struct).

No third-party dependencies. No PIL, no freetype, no fontTools.

Commands:
    text      render a string to PNG (and optionally SVG)
    glyph     render a single glyph as an annotated inspector diagram
    info      dump font metadata, tables, metrics and coverage
    term      render a string as ASCII art in the terminal
    selftest  build a synthetic font in memory and verify the whole pipeline
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import zlib

# ---------------------------------------------------------------------------
# binary reading
# ---------------------------------------------------------------------------


class FontError(Exception):
    """Anything malformed or unsupported in a font file."""


class Reader:
    """Big-endian cursor over a bytes buffer. All sfnt data is big-endian."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def seek(self, pos: int) -> "Reader":
        self.pos = pos
        return self

    def skip(self, n: int) -> "Reader":
        self.pos += n
        return self

    def _take(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.data):
            raise FontError(f"read past end of data at {self.pos} (+{n})")
        chunk = self.data[self.pos:end]
        self.pos = end
        return chunk

    def uint8(self) -> int:
        return self._take(1)[0]

    def int8(self) -> int:
        return struct.unpack(">b", self._take(1))[0]

    def uint16(self) -> int:
        return struct.unpack(">H", self._take(2))[0]

    def int16(self) -> int:
        return struct.unpack(">h", self._take(2))[0]

    def uint32(self) -> int:
        return struct.unpack(">I", self._take(4))[0]

    def int32(self) -> int:
        return struct.unpack(">i", self._take(4))[0]

    def f2dot14(self) -> float:
        return self.int16() / 16384.0

    def tag(self) -> str:
        return self._take(4).decode("latin-1")

    def array_uint16(self, n: int) -> list[int]:
        return list(struct.unpack(f">{n}H", self._take(2 * n)))

    def array_int16(self, n: int) -> list[int]:
        return list(struct.unpack(f">{n}h", self._take(2 * n)))

    def array_uint32(self, n: int) -> list[int]:
        return list(struct.unpack(f">{n}I", self._take(4 * n)))


# ---------------------------------------------------------------------------
# outlines
# ---------------------------------------------------------------------------

# A contour is a list of path commands in font units:
#   ("m", (x, y))            move to (always first, exactly once)
#   ("l", (x, y))            line to
#   ("q", (cx, cy), (x, y))  quadratic Bezier with one control point
# Contours are implicitly closed back to their move-to point.


def _decode_contour(points: list[tuple[float, float, bool]]) -> list[tuple]:
    """Turn TrueType (x, y, on_curve) points into path commands.

    TrueType stores contours as an alternating-ish sequence of on-curve and
    off-curve points. Two consecutive off-curve points imply an on-curve point
    exactly halfway between them; a contour may also start off-curve.
    """
    if not points:
        return []

    if points[0][2]:
        start = (points[0][0], points[0][1])
        rest = points[1:] + points[:1]
    elif points[-1][2]:
        # Last point is on-curve: rotate so the contour starts there.
        start = (points[-1][0], points[-1][1])
        rest = points[:]
    else:
        # No on-curve point at either end: synthesise the implied midpoint.
        start = ((points[0][0] + points[-1][0]) / 2.0,
                 (points[0][1] + points[-1][1]) / 2.0)
        rest = points[:] + [(start[0], start[1], True)]

    path: list[tuple] = [("m", start)]
    pending_ctrl: tuple[float, float] | None = None

    for x, y, on_curve in rest:
        if on_curve:
            if pending_ctrl is None:
                path.append(("l", (x, y)))
            else:
                path.append(("q", pending_ctrl, (x, y)))
                pending_ctrl = None
        else:
            if pending_ctrl is not None:
                # Implied on-curve point between two controls.
                mid = ((pending_ctrl[0] + x) / 2.0, (pending_ctrl[1] + y) / 2.0)
                path.append(("q", pending_ctrl, mid))
            pending_ctrl = (x, y)

    if pending_ctrl is not None:
        path.append(("q", pending_ctrl, start))

    return path


def transform_contour(contour: list[tuple], a: float, b: float, c: float,
                      d: float, e: float, f: float) -> list[tuple]:
    """Apply the 2x2 matrix [a b; c d] plus offset (e, f) to every point."""

    def tp(p):
        return (a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f)

    out = []
    for cmd in contour:
        if cmd[0] == "q":
            out.append(("q", tp(cmd[1]), tp(cmd[2])))
        else:
            out.append((cmd[0], tp(cmd[1])))
    return out


def flatten_contour(contour: list[tuple], scale: float, tx: float, ty: float,
                    tolerance: float = 0.12) -> list[tuple[float, float]]:
    """Flatten one contour into a device-space polygon.

    Font units are y-up; device space is y-down, so y is negated. The number of
    line segments per curve is chosen so the flattening error stays under
    `tolerance` pixels: for a quadratic, max deviation is |p0 - 2*p1 + p2| / 8,
    and splitting into n segments divides that by n^2.
    """
    if not contour:
        return []

    def dev(p):
        return (tx + p[0] * scale, ty - p[1] * scale)

    poly: list[tuple[float, float]] = []
    cur = dev(contour[0][1])
    poly.append(cur)

    for cmd in contour[1:]:
        if cmd[0] == "l":
            cur = dev(cmd[1])
            poly.append(cur)
        else:
            ctrl = dev(cmd[1])
            end = dev(cmd[2])
            dx = cur[0] - 2.0 * ctrl[0] + end[0]
            dy = cur[1] - 2.0 * ctrl[1] + end[1]
            deviation = math.hypot(dx, dy) / 8.0
            n = 1 if deviation <= tolerance else int(math.ceil(math.sqrt(deviation / tolerance)))
            n = max(1, min(n, 64))
            for i in range(1, n + 1):
                t = i / n
                mt = 1.0 - t
                px = mt * mt * cur[0] + 2.0 * mt * t * ctrl[0] + t * t * end[0]
                py = mt * mt * cur[1] + 2.0 * mt * t * ctrl[1] + t * t * end[1]
                poly.append((px, py))
            cur = end

    return poly


def contour_bbox(contours: list[list[tuple]]) -> tuple[float, float, float, float] | None:
    """Bounding box over all points, control points included."""
    xs: list[float] = []
    ys: list[float] = []
    for contour in contours:
        for cmd in contour:
            for p in cmd[1:]:
                xs.append(p[0])
                ys.append(p[1])
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# rasterizer
# ---------------------------------------------------------------------------


def _add_span(row: list[float], x0: float, x1: float, weight: float, width: int) -> None:
    """Accumulate horizontal coverage for [x0, x1) with exact partial pixels."""
    if x1 <= x0:
        return
    if x0 < 0.0:
        x0 = 0.0
    if x1 > width:
        x1 = float(width)
    if x1 <= x0:
        return

    ix0 = int(x0)
    ix1 = int(x1)
    if ix0 == ix1:
        row[ix0] += (x1 - x0) * weight
        return

    row[ix0] += (ix0 + 1 - x0) * weight
    for i in range(ix0 + 1, ix1):
        row[i] += weight
    if ix1 < width:
        row[ix1] += (x1 - ix1) * weight


def rasterize(polygons: list[list[tuple[float, float]]], width: int, height: int,
              samples: int = 15) -> list[float]:
    """Scanline-fill polygons with the nonzero winding rule.

    Anti-aliasing is analytic horizontally (exact fractional span coverage) and
    supersampled vertically (`samples` sub-scanlines per pixel row). Returns a
    flat width*height list of coverage values in [0, 1].
    """
    coverage = [0.0] * (width * height)
    if width <= 0 or height <= 0:
        return coverage

    # Edge = (y_top, y_bottom, x_at_y_top, dx/dy, winding_direction)
    edges: list[tuple[float, float, float, float, int]] = []
    for poly in polygons:
        n = len(poly)
        if n < 3:
            continue
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            if y0 == y1:
                continue
            direction = 1 if y1 > y0 else -1
            if y1 < y0:
                x0, y0, x1, y1 = x1, y1, x0, y0
            edges.append((y0, y1, x0, (x1 - x0) / (y1 - y0), direction))

    if not edges:
        return coverage

    edges.sort(key=lambda e: e[0])
    max_y = max(e[1] for e in edges)
    min_y = edges[0][0]

    first_row = max(0, int(min_y))
    last_row = min(height - 1, int(math.ceil(max_y)))

    weight = 1.0 / samples
    next_edge = 0
    active: list[tuple[float, float, float, float, int]] = []

    # Edges are added in y_top order and never revisited, so a single forward
    # sweep over sub-scanlines keeps the active list correct.
    for row in range(first_row, last_row + 1):
        rowbuf = [0.0] * width
        touched = False

        for j in range(samples):
            y = row + (j + 0.5) * weight

            while next_edge < len(edges) and edges[next_edge][0] <= y:
                active.append(edges[next_edge])
                next_edge += 1
            if active:
                active = [e for e in active if e[1] > y]

            if not active:
                continue

            crossings = []
            for y_top, y_bot, x_top, slope, direction in active:
                if y_top <= y < y_bot:
                    crossings.append((x_top + (y - y_top) * slope, direction))
            if not crossings:
                continue

            crossings.sort()
            winding = 0
            span_start = 0.0
            for x, direction in crossings:
                if winding == 0:
                    span_start = x
                winding += direction
                if winding == 0:
                    _add_span(rowbuf, span_start, x, weight, width)
                    touched = True

        if touched:
            base = row * width
            for i, v in enumerate(rowbuf):
                if v:
                    total = coverage[base + i] + v
                    coverage[base + i] = 1.0 if total > 1.0 else total

    return coverage


# ---------------------------------------------------------------------------
# canvas + PNG output
# ---------------------------------------------------------------------------


class Canvas:
    """An RGBA image with float channels, composited from coverage masks."""

    def __init__(self, width: int, height: int, background=(0, 0, 0, 0)):
        self.width = width
        self.height = height
        r, g, b, a = background
        self.pixels = [r / 255.0, g / 255.0, b / 255.0, a / 255.0] * (width * height)

    def composite(self, coverage: list[float], color, opacity: float = 1.0) -> None:
        """Source-over blend of a solid colour through a coverage mask."""
        sr, sg, sb = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        sa_base = (color[3] / 255.0 if len(color) > 3 else 1.0) * opacity
        px = self.pixels
        for i, cov in enumerate(coverage):
            if cov <= 0.0:
                continue
            sa = sa_base * cov
            if sa <= 0.0:
                continue
            o = i * 4
            da = px[o + 3]
            out_a = sa + da * (1.0 - sa)
            if out_a <= 0.0:
                continue
            px[o] = (sr * sa + px[o] * da * (1.0 - sa)) / out_a
            px[o + 1] = (sg * sa + px[o + 1] * da * (1.0 - sa)) / out_a
            px[o + 2] = (sb * sa + px[o + 2] * da * (1.0 - sa)) / out_a
            px[o + 3] = out_a

    def to_rgba_bytes(self) -> bytes:
        out = bytearray(len(self.pixels))
        for i, v in enumerate(self.pixels):
            n = int(v * 255.0 + 0.5)
            out[i] = 0 if n < 0 else (255 if n > 255 else n)
        return bytes(out)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def write_png(path: str, rgba: bytes, width: int, height: int, text: str | None = None) -> None:
    """Write an 8-bit RGBA PNG. Filter type 0 on every row; zlib does the rest."""
    if len(rgba) != width * height * 4:
        raise ValueError("pixel buffer size does not match dimensions")

    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)
        raw += rgba[y * stride:(y + 1) * stride]

    chunks = [b"\x89PNG\r\n\x1a\n",
              _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))]
    if text:
        chunks.append(_png_chunk(b"tEXt", b"Software\x00" + text.encode("latin-1", "replace")))
    chunks.append(_png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
    chunks.append(_png_chunk(b"IEND", b""))

    with open(path, "wb") as fh:
        fh.write(b"".join(chunks))


def read_png_header(path: str) -> tuple[int, int, int, int]:
    """Minimal PNG reader used by the self-test: (width, height, depth, colour)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    length = struct.unpack(">I", data[8:12])[0]
    if data[12:16] != b"IHDR" or length != 13:
        raise ValueError("malformed IHDR")
    w, h, depth, colour = struct.unpack(">IIBB", data[16:26])
    return w, h, depth, colour


def png_idat_pixels(path: str) -> tuple[int, int, bytes]:
    """Decompress a PNG written by write_png back into raw RGBA bytes."""
    with open(path, "rb") as fh:
        data = fh.read()
    width, height, depth, colour = read_png_header(path)
    if depth != 8 or colour != 6:
        raise ValueError("only 8-bit RGBA is supported")

    pos = 8
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        if tag == b"IDAT":
            idat += data[pos + 8:pos + 8 + length]
        pos += 12 + length

    raw = zlib.decompress(bytes(idat))
    stride = width * 4
    out = bytearray()
    for y in range(height):
        offset = y * (stride + 1)
        if raw[offset] != 0:
            raise ValueError("unexpected PNG row filter")
        out += raw[offset + 1:offset + 1 + stride]
    return width, height, bytes(out)


# ---------------------------------------------------------------------------
# the font
# ---------------------------------------------------------------------------

_VALUE_FORMAT_BITS = (0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080)


def _value_record_size(value_format: int) -> int:
    return 2 * sum(1 for bit in _VALUE_FORMAT_BITS if value_format & bit)


class TrueTypeFont:
    """A parsed sfnt font with glyf outlines."""

    def __init__(self, data: bytes, face_index: int = 0):
        self.data = data
        self.tables: dict[str, tuple[int, int]] = {}
        self._glyph_cache: dict[int, list[list[tuple]]] = {}
        self._read_table_directory(face_index)
        self._read_head()
        self._read_maxp()
        self._read_hhea_hmtx()
        self._read_loca()
        self._read_cmap()
        self._read_kern()
        self._read_gpos_kern()

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, path: str, face_index: int = 0) -> "TrueTypeFont":
        with open(path, "rb") as fh:
            data = fh.read()
        font = cls(data, face_index)
        font.path = path
        return font

    def _read_table_directory(self, face_index: int) -> None:
        r = Reader(self.data)
        version = r.uint32()

        if version == 0x74746366:  # 'ttcf' — a font collection
            r.skip(4)
            num_fonts = r.uint32()
            offsets = r.array_uint32(num_fonts)
            if face_index >= num_fonts:
                raise FontError(f"face {face_index} out of range (collection has {num_fonts})")
            r.seek(offsets[face_index])
            version = r.uint32()

        if version not in (0x00010000, 0x74727565, 0x4F54544F):  # 1.0, 'true', 'OTTO'
            raise FontError(f"unrecognised sfnt version 0x{version:08X}")
        self.is_cff = version == 0x4F54544F

        num_tables = r.uint16()
        r.skip(6)
        for _ in range(num_tables):
            tag = r.tag()
            r.skip(4)  # checksum
            offset = r.uint32()
            length = r.uint32()
            self.tables[tag] = (offset, length)

        if "glyf" not in self.tables or "loca" not in self.tables:
            if self.is_cff or "CFF " in self.tables:
                raise FontError(
                    "this is a CFF/PostScript-flavoured OpenType font; Kroj reads "
                    "glyf outlines only (try a .ttf)")
            raise FontError("font has no glyf/loca tables")

    def table(self, tag: str) -> bytes:
        if tag not in self.tables:
            raise FontError(f"missing table '{tag}'")
        offset, length = self.tables[tag]
        return self.data[offset:offset + length]

    def has(self, tag: str) -> bool:
        return tag in self.tables

    # -- header tables ------------------------------------------------------

    def _read_head(self) -> None:
        r = Reader(self.table("head"))
        r.skip(12)
        if r.uint32() != 0x5F0F3CF5:
            raise FontError("bad magic number in head")
        r.skip(2)
        self.units_per_em = r.uint16()
        if not 16 <= self.units_per_em <= 16384:
            raise FontError(f"implausible unitsPerEm {self.units_per_em}")
        r.skip(16)
        self.x_min, self.y_min = r.int16(), r.int16()
        self.x_max, self.y_max = r.int16(), r.int16()
        self.mac_style = r.uint16()
        r.skip(4)
        self.index_to_loc_format = r.int16()

    def _read_maxp(self) -> None:
        r = Reader(self.table("maxp"), 4)
        self.num_glyphs = r.uint16()

    def _read_hhea_hmtx(self) -> None:
        r = Reader(self.table("hhea"), 4)
        self.ascender = r.int16()
        self.descender = r.int16()
        self.line_gap = r.int16()
        r = Reader(self.table("hhea"), 34)
        num_h_metrics = r.uint16()

        self.advances: list[int] = []
        self.lsbs: list[int] = []
        if self.has("hmtx") and num_h_metrics:
            hmtx = Reader(self.table("hmtx"))
            last = 0
            for _ in range(min(num_h_metrics, self.num_glyphs)):
                last = hmtx.uint16()
                self.advances.append(last)
                self.lsbs.append(hmtx.int16())
            while len(self.advances) < self.num_glyphs:
                self.advances.append(last)
                try:
                    self.lsbs.append(hmtx.int16())
                except FontError:
                    self.lsbs.append(0)

    def _read_loca(self) -> None:
        loca = self.table("loca")
        n = self.num_glyphs + 1
        r = Reader(loca)
        if self.index_to_loc_format == 0:
            available = min(n, len(loca) // 2)
            self.loca = [v * 2 for v in r.array_uint16(available)]
        else:
            available = min(n, len(loca) // 4)
            self.loca = r.array_uint32(available)
        if len(self.loca) < 2:
            raise FontError("loca table is too short to describe any glyph")

    # -- cmap ---------------------------------------------------------------

    def _read_cmap(self) -> None:
        self.cmap: dict[int, int] = {}
        self.cmap_encoding = None
        if not self.has("cmap"):
            return

        data = self.table("cmap")
        r = Reader(data, 2)
        num_subtables = r.uint16()
        subtables = []
        for _ in range(num_subtables):
            platform = r.uint16()
            encoding = r.uint16()
            offset = r.uint32()
            subtables.append((platform, encoding, offset))

        # Preference order: full Unicode first, then BMP, then Mac Roman/symbol.
        def rank(entry):
            platform, encoding, _ = entry
            order = [(3, 10), (0, 6), (0, 4), (3, 1), (0, 3), (0, 2), (0, 1), (0, 0), (3, 0), (1, 0)]
            return order.index((platform, encoding)) if (platform, encoding) in order else 99

        for platform, encoding, offset in sorted(subtables, key=rank):
            if offset >= len(data):
                continue
            try:
                table = self._parse_cmap_subtable(data, offset)
            except FontError:
                continue
            if table:
                self.cmap = table
                self.cmap_encoding = (platform, encoding)
                # Symbol fonts map into the 0xF000 private-use block.
                if (platform, encoding) == (3, 0):
                    for code in list(table):
                        if 0xF000 <= code <= 0xF0FF:
                            self.cmap.setdefault(code - 0xF000, table[code])
                break

    def _parse_cmap_subtable(self, data: bytes, offset: int) -> dict[int, int]:
        r = Reader(data, offset)
        fmt = r.uint16()
        table: dict[int, int] = {}

        if fmt == 0:
            r.skip(4)
            for code in range(256):
                gid = r.uint8()
                if gid:
                    table[code] = gid

        elif fmt == 4:
            r.skip(4)
            seg_count = r.uint16() // 2
            r.skip(6)
            end_codes = r.array_uint16(seg_count)
            r.skip(2)
            start_codes = r.array_uint16(seg_count)
            id_deltas = r.array_int16(seg_count)
            range_offset_pos = r.pos
            id_range_offsets = r.array_uint16(seg_count)

            for i in range(seg_count):
                start, end = start_codes[i], end_codes[i]
                if start > end or (start == 0xFFFF and end == 0xFFFF):
                    continue
                for code in range(start, min(end, 0xFFFF) + 1):
                    if id_range_offsets[i] == 0:
                        gid = (code + id_deltas[i]) & 0xFFFF
                    else:
                        addr = range_offset_pos + i * 2 + id_range_offsets[i] + (code - start) * 2
                        if addr + 2 > len(data):
                            continue
                        gid = struct.unpack(">H", data[addr:addr + 2])[0]
                        if gid:
                            gid = (gid + id_deltas[i]) & 0xFFFF
                    if gid:
                        table[code] = gid

        elif fmt == 6:
            r.skip(4)
            first = r.uint16()
            count = r.uint16()
            for i, gid in enumerate(r.array_uint16(count)):
                if gid:
                    table[first + i] = gid

        elif fmt == 12:
            r.skip(10)
            n_groups = r.uint32()
            for _ in range(min(n_groups, 200000)):
                start = r.uint32()
                end = r.uint32()
                start_gid = r.uint32()
                if end - start > 0x10FFFF:
                    continue
                for k in range(end - start + 1):
                    table[start + k] = start_gid + k

        else:
            raise FontError(f"unsupported cmap format {fmt}")

        return table

    def glyph_id(self, char: str) -> int:
        return self.cmap.get(ord(char), 0)

    # -- kerning ------------------------------------------------------------

    def _read_kern(self) -> None:
        self.kern_pairs: dict[tuple[int, int], int] = {}
        if not self.has("kern"):
            return
        data = self.table("kern")
        if len(data) < 4:
            return

        r = Reader(data)
        version = r.uint16()
        if version != 0:
            return  # Apple's 1.0 kern layout; skip rather than misread it.
        num_subtables = r.uint16()

        for _ in range(num_subtables):
            base = r.pos
            try:
                r.uint16()  # subtable version
                length = r.uint16()
                coverage = r.uint16()
            except FontError:
                return
            fmt = coverage >> 8
            horizontal = coverage & 0x0001
            minimum = coverage & 0x0002
            if fmt == 0 and horizontal and not minimum:
                n_pairs = r.uint16()
                r.skip(6)
                for _ in range(n_pairs):
                    try:
                        left = r.uint16()
                        right = r.uint16()
                        value = r.int16()
                    except FontError:
                        break
                    self.kern_pairs[(left, right)] = value
            if length <= 0:
                return
            r.seek(base + length)

    def _read_gpos_kern(self) -> None:
        """Extract horizontal pair kerning from GPOS 'kern' features.

        Modern fonts often ship no `kern` table at all. This handles lookup
        type 2 (pair adjustment) formats 1 and 2, plus type 9 extensions.
        """
        self.gpos_subtables: list[dict] = []
        if not self.has("GPOS"):
            return
        try:
            data = self.table("GPOS")
            r = Reader(data)
            r.skip(4)
            r.uint16()  # scriptList — every script gets the same kerning here
            feature_list = r.uint16()
            lookup_list = r.uint16()

            wanted: set[int] = set()
            fr = Reader(data, feature_list)
            for _ in range(fr.uint16()):
                tag = fr.tag()
                offset = fr.uint16()
                if tag != "kern":
                    continue
                f = Reader(data, feature_list + offset)
                f.skip(2)
                for idx in f.array_uint16(f.uint16()):
                    wanted.add(idx)
            if not wanted:
                return

            lr = Reader(data, lookup_list)
            lookup_offsets = lr.array_uint16(lr.uint16())
            for idx in sorted(wanted):
                if idx >= len(lookup_offsets):
                    continue
                self._parse_gpos_lookup(data, lookup_list + lookup_offsets[idx])
        except FontError:
            self.gpos_subtables = []

    def _parse_gpos_lookup(self, data: bytes, offset: int, depth: int = 0) -> None:
        if depth > 2:
            return
        r = Reader(data, offset)
        lookup_type = r.uint16()
        r.uint16()  # lookupFlag
        sub_offsets = r.array_uint16(r.uint16())

        for sub in sub_offsets:
            base = offset + sub
            if lookup_type == 9:  # extension: redirect to the real subtable
                er = Reader(data, base)
                er.uint16()
                real_type = er.uint16()
                real_offset = base + er.uint32()
                if real_type == 2:
                    self._parse_pair_pos(data, real_offset)
            elif lookup_type == 2:
                self._parse_pair_pos(data, base)

    def _parse_coverage(self, data: bytes, offset: int) -> dict[int, int]:
        r = Reader(data, offset)
        fmt = r.uint16()
        table: dict[int, int] = {}
        if fmt == 1:
            for i, gid in enumerate(r.array_uint16(r.uint16())):
                table[gid] = i
        elif fmt == 2:
            for _ in range(r.uint16()):
                start = r.uint16()
                end = r.uint16()
                start_index = r.uint16()
                for k in range(min(end - start + 1, 0xFFFF)):
                    table[start + k] = start_index + k
        return table

    def _parse_classdef(self, data: bytes, offset: int) -> dict[int, int]:
        r = Reader(data, offset)
        fmt = r.uint16()
        table: dict[int, int] = {}
        if fmt == 1:
            start = r.uint16()
            for i, cls in enumerate(r.array_uint16(r.uint16())):
                if cls:
                    table[start + i] = cls
        elif fmt == 2:
            for _ in range(r.uint16()):
                start = r.uint16()
                end = r.uint16()
                cls = r.uint16()
                if not cls:
                    continue
                for k in range(min(end - start + 1, 0xFFFF)):
                    table[start + k] = cls
        return table

    def _parse_pair_pos(self, data: bytes, offset: int) -> None:
        r = Reader(data, offset)
        fmt = r.uint16()
        coverage_offset = r.uint16()
        value_format1 = r.uint16()
        value_format2 = r.uint16()
        size1 = _value_record_size(value_format1)
        size2 = _value_record_size(value_format2)
        # We only care about XAdvance of the first glyph in the pair.
        if not value_format1 & 0x0004:
            return
        x_advance_shift = 2 * sum(1 for bit in (0x0001, 0x0002) if value_format1 & bit)

        coverage = self._parse_coverage(data, offset + coverage_offset)
        if not coverage:
            return

        if fmt == 1:
            pair_set_offsets = r.array_uint16(r.uint16())
            pairs: dict[tuple[int, int], int] = {}
            index_to_glyph = {v: k for k, v in coverage.items()}
            for i, pso in enumerate(pair_set_offsets):
                first_glyph = index_to_glyph.get(i)
                if first_glyph is None:
                    continue
                pr = Reader(data, offset + pso)
                try:
                    count = pr.uint16()
                    for _ in range(count):
                        second = pr.uint16()
                        record_start = pr.pos
                        value = struct.unpack(
                            ">h", data[record_start + x_advance_shift:
                                       record_start + x_advance_shift + 2])[0]
                        pr.skip(size1 + size2)
                        if value:
                            pairs[(first_glyph, second)] = value
                except (FontError, struct.error):
                    continue
            if pairs:
                self.gpos_subtables.append({"kind": 1, "pairs": pairs})

        elif fmt == 2:
            class_def1 = r.uint16()
            class_def2 = r.uint16()
            class1_count = r.uint16()
            class2_count = r.uint16()
            records_start = r.pos
            classes1 = self._parse_classdef(data, offset + class_def1)
            classes2 = self._parse_classdef(data, offset + class_def2)
            self.gpos_subtables.append({
                "kind": 2,
                "coverage": coverage,
                "classes1": classes1,
                "classes2": classes2,
                "class2_count": class2_count,
                "class1_count": class1_count,
                "records": records_start,
                "stride": size1 + size2,
                "shift": x_advance_shift,
                "data": data,
            })

    def kerning(self, left_gid: int, right_gid: int) -> int:
        """Kerning adjustment in font units (negative pulls glyphs together)."""
        value = self.kern_pairs.get((left_gid, right_gid))
        if value is not None:
            return value

        for sub in self.gpos_subtables:
            if sub["kind"] == 1:
                value = sub["pairs"].get((left_gid, right_gid))
                if value:
                    return value
            else:
                if left_gid not in sub["coverage"]:
                    continue
                c1 = sub["classes1"].get(left_gid, 0)
                c2 = sub["classes2"].get(right_gid, 0)
                if c1 >= sub["class1_count"] or c2 >= sub["class2_count"]:
                    continue
                pos = (sub["records"] + (c1 * sub["class2_count"] + c2) * sub["stride"]
                       + sub["shift"])
                data = sub["data"]
                if pos + 2 > len(data):
                    continue
                found = struct.unpack(">h", data[pos:pos + 2])[0]
                if found:
                    return found
        return 0

    # -- glyphs -------------------------------------------------------------

    def advance(self, gid: int) -> int:
        if not self.advances:
            return self.units_per_em // 2
        if gid < len(self.advances):
            return self.advances[gid]
        return self.advances[-1]

    def left_side_bearing(self, gid: int) -> int:
        if gid < len(self.lsbs):
            return self.lsbs[gid]
        return 0

    def glyph_contours(self, gid: int, depth: int = 0) -> list[list[tuple]]:
        """Outline of a glyph in font units, composites resolved."""
        if gid in self._glyph_cache:
            return self._glyph_cache[gid]
        if gid < 0 or gid + 1 >= len(self.loca):
            return []

        start, end = self.loca[gid], self.loca[gid + 1]
        if end <= start:
            return []  # blank glyph (space and friends)

        glyf_offset, glyf_length = self.tables["glyf"]
        if end > glyf_length:
            return []
        data = self.data[glyf_offset + start:glyf_offset + end]

        r = Reader(data)
        num_contours = r.int16()
        r.skip(8)  # per-glyph bbox, recomputed from points when needed

        if num_contours >= 0:
            contours = self._parse_simple_glyph(r, num_contours)
        else:
            contours = self._parse_composite_glyph(r, depth)

        if depth == 0:
            self._glyph_cache[gid] = contours
        return contours

    def _parse_simple_glyph(self, r: Reader, num_contours: int) -> list[list[tuple]]:
        if num_contours == 0:
            return []
        end_points = r.array_uint16(num_contours)
        num_points = end_points[-1] + 1
        if num_points > 10000:
            raise FontError(f"implausible point count {num_points}")
        r.skip(r.uint16())  # hinting instructions — Kroj does not hint

        flags: list[int] = []
        while len(flags) < num_points:
            flag = r.uint8()
            flags.append(flag)
            if flag & 0x08:  # REPEAT
                repeat = r.uint8()
                flags.extend([flag] * min(repeat, num_points - len(flags)))

        xs: list[float] = []
        x = 0
        for flag in flags:
            if flag & 0x02:  # X_SHORT
                delta = r.uint8()
                x += delta if flag & 0x10 else -delta
            elif not flag & 0x10:  # not X_SAME
                x += r.int16()
            xs.append(float(x))

        ys: list[float] = []
        y = 0
        for flag in flags:
            if flag & 0x04:  # Y_SHORT
                delta = r.uint8()
                y += delta if flag & 0x20 else -delta
            elif not flag & 0x20:  # not Y_SAME
                y += r.int16()
            ys.append(float(y))

        contours: list[list[tuple]] = []
        first = 0
        for last in end_points:
            points = [(xs[i], ys[i], bool(flags[i] & 0x01))
                      for i in range(first, min(last + 1, num_points))]
            path = _decode_contour(points)
            if path:
                contours.append(path)
            first = last + 1
        return contours

    def _parse_composite_glyph(self, r: Reader, depth: int) -> list[list[tuple]]:
        if depth > 5:
            raise FontError("composite glyph nesting is too deep")
        contours: list[list[tuple]] = []

        while True:
            flags = r.uint16()
            component_gid = r.uint16()

            if flags & 0x0001:  # ARG_1_AND_2_ARE_WORDS
                arg1, arg2 = (r.int16(), r.int16()) if flags & 0x0002 else (r.uint16(), r.uint16())
            else:
                arg1, arg2 = (r.int8(), r.int8()) if flags & 0x0002 else (r.uint8(), r.uint8())

            a = d = 1.0
            b = c = 0.0
            if flags & 0x0008:  # WE_HAVE_A_SCALE
                a = d = r.f2dot14()
            elif flags & 0x0040:  # X_AND_Y_SCALE
                a = r.f2dot14()
                d = r.f2dot14()
            elif flags & 0x0080:  # TWO_BY_TWO
                a, b, c, d = r.f2dot14(), r.f2dot14(), r.f2dot14(), r.f2dot14()

            # Point-matching placement (ARGS_ARE_XY_VALUES clear) is vanishingly
            # rare and needs the parent's points; fall back to no offset.
            dx, dy = (float(arg1), float(arg2)) if flags & 0x0002 else (0.0, 0.0)

            for contour in self.glyph_contours(component_gid, depth + 1):
                contours.append(transform_contour(contour, a, b, c, d, dx, dy))

            if not flags & 0x0020:  # MORE_COMPONENTS
                break

        return contours

    # -- names --------------------------------------------------------------

    def names(self) -> dict[int, str]:
        result: dict[int, str] = {}
        if not self.has("name"):
            return result
        data = self.table("name")
        r = Reader(data)
        r.uint16()
        count = r.uint16()
        string_offset = r.uint16()
        for _ in range(count):
            try:
                platform = r.uint16()
                encoding = r.uint16()
                r.uint16()  # language
                name_id = r.uint16()
                length = r.uint16()
                offset = r.uint16()
            except FontError:
                break
            raw = data[string_offset + offset:string_offset + offset + length]
            if platform == 3 or (platform == 0):
                try:
                    text = raw.decode("utf-16-be")
                except UnicodeDecodeError:
                    continue
            elif platform == 1 and encoding == 0:
                text = raw.decode("mac-roman", "replace")
            else:
                continue
            result.setdefault(name_id, text)
        return result

    @property
    def family(self) -> str:
        return self.names().get(1, "(unknown)")

    @property
    def subfamily(self) -> str:
        return self.names().get(2, "")


# ---------------------------------------------------------------------------
# font lookup on disk
# ---------------------------------------------------------------------------


def font_directories() -> list[str]:
    dirs = []
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(os.path.join(windir, "Fonts"))
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
    else:
        dirs += ["/usr/share/fonts", "/usr/local/share/fonts",
                 os.path.expanduser("~/.local/share/fonts"), os.path.expanduser("~/.fonts"),
                 "/System/Library/Fonts", "/Library/Fonts"]
    return [d for d in dirs if os.path.isdir(d)]


def resolve_font(name: str) -> str:
    """Accept a path, a filename, or a loose family name."""
    if os.path.isfile(name):
        return name

    target = name.lower()
    candidates: list[str] = []
    for directory in font_directories():
        for root, _dirs, files in os.walk(directory):
            for filename in files:
                if not filename.lower().endswith((".ttf", ".ttc")):
                    continue
                path = os.path.join(root, filename)
                stem = os.path.splitext(filename)[0].lower()
                if filename.lower() == target or stem == target:
                    return path
                if target in stem:
                    candidates.append(path)

    if candidates:
        candidates.sort(key=len)
        return candidates[0]
    raise FontError(f"could not find a font matching {name!r}; pass a full path to a .ttf")


def default_font() -> str:
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf", "LiberationSans-Regular.ttf",
                 "Helvetica.ttc", "verdana.ttf"):
        try:
            return resolve_font(name)
        except FontError:
            continue
    raise FontError("no default font found; pass --font")


# ---------------------------------------------------------------------------
# text layout
# ---------------------------------------------------------------------------


class Shaped:
    """Positioned glyphs for one line of text."""

    def __init__(self):
        self.glyphs: list[tuple[int, float]] = []  # (gid, pen x in pixels)
        self.width = 0.0
        self.missing: list[str] = []


def shape_line(font: TrueTypeFont, text: str, size: float, kerning: bool = True,
               tracking: float = 0.0) -> Shaped:
    scale = size / font.units_per_em
    out = Shaped()
    pen = 0.0
    previous: int | None = None

    for char in text:
        gid = font.glyph_id(char)
        if gid == 0 and char != "\u0000":
            out.missing.append(char)
        if kerning and previous is not None:
            pen += font.kerning(previous, gid) * scale
        out.glyphs.append((gid, pen))
        pen += font.advance(gid) * scale + tracking
        previous = gid

    out.width = pen
    return out


def text_polygons(font: TrueTypeFont, shaped: Shaped, size: float, origin_x: float,
                  baseline_y: float) -> list[list[tuple[float, float]]]:
    scale = size / font.units_per_em
    polygons: list[list[tuple[float, float]]] = []
    for gid, pen in shaped.glyphs:
        for contour in font.glyph_contours(gid):
            poly = flatten_contour(contour, scale, origin_x + pen, baseline_y)
            if len(poly) >= 3:
                polygons.append(poly)
    return polygons


def parse_color(text: str) -> tuple[int, int, int, int]:
    value = text.strip().lstrip("#")
    named = {
        "black": "000000", "white": "ffffff", "red": "e5484d", "green": "30a46c",
        "blue": "3e63dd", "amber": "ffb224", "grey": "8b8d98", "gray": "8b8d98",
        "ink": "11131a", "paper": "faf8f4", "none": "00000000",
    }
    value = named.get(value.lower(), value)
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    if len(value) == 6:
        value += "ff"
    if len(value) != 8:
        raise ValueError(f"cannot parse colour {text!r}")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4, 6))


# ---------------------------------------------------------------------------
# stroking (used by the inspector)
# ---------------------------------------------------------------------------


def stroke_polyline(points: list[tuple[float, float]], width: float,
                    closed: bool = False) -> list[list[tuple[float, float]]]:
    """Turn a polyline into fillable quads, one per segment plus round-ish joins."""
    half = width / 2.0
    quads: list[list[tuple[float, float]]] = []
    n = len(points)
    if n < 2:
        return quads

    limit = n if closed else n - 1
    for i in range(limit):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        nx, ny = -dy / length * half, dx / length * half
        quads.append([(x0 + nx, y0 + ny), (x1 + nx, y1 + ny),
                      (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)])

    # Square patches at the joints so corners do not show gaps.
    joint_range = range(n) if closed else range(1, n - 1)
    for i in joint_range:
        x, y = points[i]
        quads.append([(x - half, y - half), (x + half, y - half),
                      (x + half, y + half), (x - half, y + half)])
    return quads


def circle_polygon(cx: float, cy: float, radius: float, segments: int = 20) -> list[tuple[float, float]]:
    return [(cx + radius * math.cos(2 * math.pi * i / segments),
             cy + radius * math.sin(2 * math.pi * i / segments)) for i in range(segments)]


def square_polygon(cx: float, cy: float, radius: float) -> list[tuple[float, float]]:
    return [(cx - radius, cy - radius), (cx + radius, cy - radius),
            (cx + radius, cy + radius), (cx - radius, cy + radius)]


def dashed_line(x0: float, y0: float, x1: float, y1: float, dash: float = 8.0,
                gap: float = 6.0) -> list[list[tuple[float, float]]]:
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-9:
        return []
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    segments = []
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        segments.append([(x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end)])
        pos = end + gap
    return segments


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_text(args) -> int:
    font = TrueTypeFont.load(resolve_font(args.font), args.face)
    lines = args.text.split("\\n") if "\\n" in args.text else args.text.split("\n")
    size = args.size
    scale = size / font.units_per_em

    shaped = [shape_line(font, line, size, kerning=not args.no_kerning, tracking=args.tracking)
              for line in lines]

    ascent = font.ascender * scale
    descent = -font.descender * scale
    line_height = (font.ascender - font.descender + font.line_gap) * scale * args.line_height

    pad = args.pad
    text_width = max((s.width for s in shaped), default=0.0)
    width = int(math.ceil(text_width + 2 * pad)) or 1
    height = int(math.ceil(ascent + descent + line_height * (len(shaped) - 1) + 2 * pad)) or 1

    polygons: list[list[tuple[float, float]]] = []
    for i, line in enumerate(shaped):
        if args.align == "center":
            x = pad + (text_width - line.width) / 2.0
        elif args.align == "right":
            x = pad + text_width - line.width
        else:
            x = pad
        polygons += text_polygons(font, line, size, x, pad + ascent + i * line_height)

    if args.tight and polygons:
        xs = [p[0] for poly in polygons for p in poly]
        ys = [p[1] for poly in polygons for p in poly]
        off_x = min(xs) - pad
        off_y = min(ys) - pad
        polygons = [[(px - off_x, py - off_y) for px, py in poly] for poly in polygons]
        width = int(math.ceil(max(xs) - min(xs) + 2 * pad)) or 1
        height = int(math.ceil(max(ys) - min(ys) + 2 * pad)) or 1

    coverage = rasterize(polygons, width, height, samples=args.samples)
    canvas = Canvas(width, height, parse_color(args.bg))
    canvas.composite(coverage, parse_color(args.color))
    write_png(args.out, canvas.to_rgba_bytes(), width, height, text="Kroj")

    missing = sorted({c for s in shaped for c in s.missing})
    print(f"{args.out}  {width}x{height}px  {sum(len(s.glyphs) for s in shaped)} glyphs  "
          f"{len(polygons)} contours  [{font.family} {font.subfamily}]".rstrip())
    if missing:
        print(f"  warning: no glyph for {' '.join(repr(c) for c in missing)}")

    if args.svg:
        write_svg(args.svg, font, shaped, size, pad, pad + ascent, line_height, width, height,
                  args.color, args.align, text_width)
        print(f"{args.svg}  vector outlines")
    return 0


def write_svg(path: str, font: TrueTypeFont, shaped: list[Shaped], size: float,
              pad: float, baseline: float, line_height: float, width: int, height: int,
              color: str, align: str, text_width: float) -> None:
    """Export the same outlines as SVG paths — exact quadratics, no flattening."""
    scale = size / font.units_per_em
    fill = "#" + "".join(f"{c:02x}" for c in parse_color(color)[:3])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">']

    for i, line in enumerate(shaped):
        if align == "center":
            origin = pad + (text_width - line.width) / 2.0
        elif align == "right":
            origin = pad + text_width - line.width
        else:
            origin = pad
        y0 = baseline + i * line_height
        for gid, pen in line.glyphs:
            commands = []
            for contour in font.glyph_contours(gid):
                for cmd in contour:
                    pts = [(origin + pen + p[0] * scale, y0 - p[1] * scale) for p in cmd[1:]]
                    if cmd[0] == "m":
                        commands.append(f"M{pts[0][0]:.2f} {pts[0][1]:.2f}")
                    elif cmd[0] == "l":
                        commands.append(f"L{pts[0][0]:.2f} {pts[0][1]:.2f}")
                    else:
                        commands.append(f"Q{pts[0][0]:.2f} {pts[0][1]:.2f} "
                                        f"{pts[1][0]:.2f} {pts[1][1]:.2f}")
                commands.append("Z")
            if commands:
                parts.append(f'<path fill="{fill}" fill-rule="nonzero" d="{" ".join(commands)}"/>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


PALETTE = {
    "bg": (18, 19, 24, 255),
    "grid": (40, 43, 54, 255),
    "em": (70, 75, 95, 255),
    "baseline": (229, 72, 77, 255),
    "advance": (255, 178, 36, 255),
    "fill": (226, 232, 240, 255),
    "outline": (99, 179, 237, 255),
    "handle": (120, 130, 160, 255),
    "on_curve": (48, 164, 108, 255),
    "off_curve": (255, 122, 89, 255),
    "label": (150, 158, 178, 255),
}


def cmd_glyph(args) -> int:
    """Render one glyph as an annotated diagram: outline, points, metrics."""
    font = TrueTypeFont.load(resolve_font(args.font), args.face)

    if args.gid is not None:
        gid = args.gid
        label = f"gid {gid}"
    else:
        if not args.char:
            raise SystemExit("pass --char or --gid")
        char = args.char[0]
        gid = font.glyph_id(char)
        label = f"U+{ord(char):04X} '{char}' -> gid {gid}"
        if gid == 0:
            print(f"warning: {char!r} is not in this font's cmap; rendering .notdef")

    contours = font.glyph_contours(gid)
    size = args.size
    scale = size / font.units_per_em
    margin = int(size * 0.28)

    ascent = font.ascender * scale
    descent = -font.descender * scale
    advance = font.advance(gid) * scale

    # The caption is set in the font under inspection, so it has to be measured
    # before the canvas is sized or it would run off the right edge.
    caption_size = max(11.0, size / 26.0)
    caption = (f"{font.family} {font.subfamily}  |  {label}  |  "
               f"{len(contours)} contours  |  adv {font.advance(gid)} lsb "
               f"{font.left_side_bearing(gid)}  |  upem {font.units_per_em}")
    caption_shaped = shape_line(font, caption, caption_size)

    width = int(max(advance + 2 * margin, caption_shaped.width + 24.0))
    height = int(ascent + descent + 2 * margin)
    origin_x = float(margin)
    baseline_y = margin + ascent

    # Widen if the outline spills outside the advance width (common for italics).
    box = contour_bbox(contours)
    if box:
        right = origin_x + box[2] * scale
        left = origin_x + box[0] * scale
        if right + margin > width:
            width = int(right + margin)
        if left < margin / 2:
            shift = margin / 2 - left
            origin_x += shift
            width += int(shift)

    canvas = Canvas(width, height, PALETTE["bg"])

    def draw(polys, color, opacity=1.0):
        if polys:
            canvas.composite(rasterize(polys, width, height, samples=args.samples), color, opacity)

    # Em-square grid every 1/8 em.
    step = size / 8.0
    grid = []
    x = origin_x
    while x <= width:
        grid += stroke_polyline([(x, 0), (x, height)], 1.0)
        x += step
    x = origin_x - step
    while x >= 0:
        grid += stroke_polyline([(x, 0), (x, height)], 1.0)
        x -= step
    y = baseline_y
    while y <= height:
        grid += stroke_polyline([(0, y), (width, y)], 1.0)
        y += step
    y = baseline_y - step
    while y >= 0:
        grid += stroke_polyline([(0, y), (width, y)], 1.0)
        y -= step
    draw(grid, PALETTE["grid"])

    # Em box: from the baseline up one ascender and down one descender.
    em_lines = []
    for yy in (baseline_y - ascent, baseline_y + descent):
        for seg in dashed_line(0, yy, width, yy):
            em_lines += stroke_polyline(seg, 1.6)
    draw(em_lines, PALETTE["em"])

    # Baseline and advance width.
    draw(stroke_polyline([(0, baseline_y), (width, baseline_y)], 2.0), PALETTE["baseline"])
    advance_lines = []
    for xx in (origin_x, origin_x + advance):
        for seg in dashed_line(xx, 0, xx, height):
            advance_lines += stroke_polyline(seg, 1.6)
    draw(advance_lines, PALETTE["advance"])

    polygons = [flatten_contour(c, scale, origin_x, baseline_y, tolerance=0.05) for c in contours]
    polygons = [p for p in polygons if len(p) >= 3]

    if not args.no_fill:
        draw(polygons, PALETTE["fill"], opacity=0.20)

    outline = []
    for poly in polygons:
        outline += stroke_polyline(poly, 2.0, closed=True)
    draw(outline, PALETTE["outline"])

    if not args.no_points:
        def dev(p):
            return (origin_x + p[0] * scale, baseline_y - p[1] * scale)

        handles = []
        on_points = []
        off_points = []
        for contour in contours:
            current = dev(contour[0][1])
            on_points.append(current)
            for cmd in contour[1:]:
                if cmd[0] == "q":
                    ctrl = dev(cmd[1])
                    end = dev(cmd[2])
                    handles += stroke_polyline([current, ctrl], 1.0)
                    handles += stroke_polyline([ctrl, end], 1.0)
                    off_points.append(ctrl)
                    on_points.append(end)
                    current = end
                else:
                    current = dev(cmd[1])
                    on_points.append(current)
        draw(handles, PALETTE["handle"])
        r = max(2.2, size / 150.0)
        draw([circle_polygon(x, y, r * 1.15) for x, y in off_points], PALETTE["off_curve"])
        draw([square_polygon(x, y, r) for x, y in on_points], PALETTE["on_curve"])

    draw(text_polygons(font, caption_shaped, caption_size, 12.0, height - 12.0),
         PALETTE["label"])

    write_png(args.out, canvas.to_rgba_bytes(), width, height, text="Kroj glyph inspector")
    print(f"{args.out}  {width}x{height}px  {label}  {len(contours)} contours  "
          f"advance {font.advance(gid)} units")
    return 0


RAMP = " .:-=+*#%@"


def cmd_term(args) -> int:
    font = TrueTypeFont.load(resolve_font(args.font), args.face)
    shaped = shape_line(font, args.text, 100.0, kerning=not args.no_kerning)
    if shaped.width <= 0:
        print("(nothing to render)")
        return 0

    scale_to_cols = args.cols / shaped.width
    size = 100.0 * scale_to_cols
    ascent = font.ascender / font.units_per_em * size
    descent = -font.descender / font.units_per_em * size

    shaped = shape_line(font, args.text, size, kerning=not args.no_kerning)
    width = max(1, int(math.ceil(shaped.width)))
    # Terminal cells are about twice as tall as they are wide.
    height = max(2, int(math.ceil((ascent + descent) * 2)))

    # Render at double height, then pair up rows so each character cell covers
    # two samples — that cancels the ~1:2 aspect ratio of a terminal cell.
    baseline = ascent * 2
    polygons = _scale_y(text_polygons(font, shaped, size, 0.0, baseline), 2.0, baseline)
    coverage = rasterize(polygons, width, height, samples=9)

    lines = []
    for row in range(0, height - 1, 2):
        chars = []
        for col in range(width):
            value = (coverage[row * width + col] + coverage[(row + 1) * width + col]) / 2.0
            if args.invert:
                value = 1.0 - value
            index = min(len(RAMP) - 1, int(value * len(RAMP)))
            chars.append(RAMP[index])
        lines.append("".join(chars).rstrip())
    print("\n".join(lines))
    return 0


def _scale_y(polygons, factor, baseline):
    return [[(x, baseline + (y - baseline) * factor) for x, y in poly] for poly in polygons]


def cmd_info(args) -> int:
    path = resolve_font(args.font)
    font = TrueTypeFont.load(path, args.face)
    names = font.names()

    print(f"file        {path}")
    print(f"size        {len(font.data):,} bytes")
    print(f"family      {names.get(1, '?')}  |  subfamily {names.get(2, '')}")
    if names.get(4):
        print(f"full name   {names[4]}")
    if names.get(6):
        print(f"postscript  {names[6]}")
    print(f"glyphs      {font.num_glyphs}")
    print(f"unitsPerEm  {font.units_per_em}")
    print(f"bbox        ({font.x_min}, {font.y_min}) .. ({font.x_max}, {font.y_max})")
    print(f"vertical    ascender {font.ascender}  descender {font.descender}  "
          f"lineGap {font.line_gap}")
    print(f"loca format {'short' if font.index_to_loc_format == 0 else 'long'}")
    print(f"cmap        {len(font.cmap)} codepoints "
          f"via platform/encoding {font.cmap_encoding}")

    kern_sources = []
    if font.kern_pairs:
        kern_sources.append(f"kern ({len(font.kern_pairs)} pairs)")
    if font.gpos_subtables:
        explicit = sum(len(s["pairs"]) for s in font.gpos_subtables if s["kind"] == 1)
        classes = sum(1 for s in font.gpos_subtables if s["kind"] == 2)
        kern_sources.append(f"GPOS ({explicit} explicit pairs, {classes} class subtables)")
    print(f"kerning     {', '.join(kern_sources) if kern_sources else 'none'}")

    print(f"tables      {len(font.tables)}")
    for tag in sorted(font.tables):
        offset, length = font.tables[tag]
        print(f"  {tag:<5} offset {offset:>8}  length {length:>9,}")

    samples = ["A", "a", "g", "Ą", "ż", "ó", "€", "→", "あ"]
    covered = [f"{c}={font.glyph_id(c)}" for c in samples if font.glyph_id(c)]
    print(f"probe       {'  '.join(covered) if covered else '(none of the probes are present)'}")

    if args.kern_pairs:
        pairs = [("A", "V"), ("V", "A"), ("T", "o"), ("A", "W"), ("L", "T"), ("P", "a"),
                 ("W", "a"), ("r", "."), ("F", "A"), ("y", ",")]
        print("kern probe")
        for left, right in pairs:
            lg, rg = font.glyph_id(left), font.glyph_id(right)
            if lg and rg:
                value = font.kerning(lg, rg)
                if value:
                    print(f"  {left}{right}  {value:+d} units "
                          f"({value / font.units_per_em:+.3f} em)")
    return 0


# ---------------------------------------------------------------------------
# synthetic font — lets the self-test run without touching the system
# ---------------------------------------------------------------------------


def build_test_font() -> bytes:
    """Assemble a tiny valid TTF in memory: .notdef, a square and a triangle."""
    units_per_em = 1000

    def glyph_header(n_contours, x_min, y_min, x_max, y_max):
        return struct.pack(">hhhhh", n_contours, x_min, y_min, x_max, y_max)

    def simple_glyph(points, x_min, y_min, x_max, y_max):
        """One closed contour of on-curve points, int16 deltas throughout."""
        body = glyph_header(1, x_min, y_min, x_max, y_max)
        body += struct.pack(">H", len(points) - 1)  # endPtsOfContours
        body += struct.pack(">H", 0)                # no instructions
        body += bytes([0x01]) * len(points)         # every point on-curve, no repeats
        prev = 0
        for x, _y in points:
            body += struct.pack(">h", x - prev)
            prev = x
        prev = 0
        for _x, y in points:
            body += struct.pack(">h", y - prev)
            prev = y
        if len(body) % 4:
            body += b"\0" * (4 - len(body) % 4)
        return body

    square = simple_glyph([(100, 100), (900, 100), (900, 900), (100, 900)], 100, 100, 900, 900)
    triangle = simple_glyph([(0, 0), (800, 0), (400, 800)], 0, 0, 800, 800)
    glyf = b"" + square + triangle           # glyph 0 is empty (.notdef)
    offsets = [0, 0, len(square), len(square) + len(triangle)]

    head = (struct.pack(">IIII", 0x00010000, 0x00010000, 0, 0x5F0F3CF5)
            + struct.pack(">HH", 0, units_per_em)
            + b"\0" * 16
            + struct.pack(">hhhh", 0, 0, 900, 900)
            + struct.pack(">HHh", 0, 8, 2)
            + struct.pack(">hh", 0, 0))       # indexToLocFormat = short, glyphDataFormat
    maxp = struct.pack(">IH", 0x00010000, 3) + b"\0" * 26
    hhea = (struct.pack(">I", 0x00010000)
            + struct.pack(">hhh", 800, -200, 0)
            + struct.pack(">HhhH", 1000, 0, 0, 900)
            + struct.pack(">hhh", 1, 0, 0)
            + b"\0" * 8
            + struct.pack(">hH", 0, 3))
    hmtx = (struct.pack(">Hh", 600, 0) + struct.pack(">Hh", 1000, 100)
            + struct.pack(">Hh", 900, 0))
    loca = struct.pack(">4H", *[o // 2 for o in offsets])

    # cmap format 4: 'A' -> 1, 'B' -> 2, plus the mandatory 0xFFFF terminator.
    segments = [(0x41, 0x41, 1 - 0x41), (0x42, 0x42, 2 - 0x42), (0xFFFF, 0xFFFF, 1)]
    seg_count = len(segments)
    sub = struct.pack(">HHHHHHH", 4, 16 + seg_count * 8, 0, seg_count * 2,
                      2 * (2 ** int(math.log2(seg_count))), int(math.log2(seg_count)), 0)
    sub += struct.pack(f">{seg_count}H", *[s[1] for s in segments])
    sub += struct.pack(">H", 0)
    sub += struct.pack(f">{seg_count}H", *[s[0] for s in segments])
    sub += struct.pack(f">{seg_count}h", *[s[2] for s in segments])
    sub += struct.pack(f">{seg_count}H", *([0] * seg_count))
    cmap = struct.pack(">HHHHI", 0, 1, 3, 1, 12) + sub

    name_records = [(3, 1, 0x409, 1, "Kroj Test"), (3, 1, 0x409, 2, "Regular")]
    strings = b""
    records = b""
    for platform, encoding, language, name_id, text in name_records:
        encoded = text.encode("utf-16-be")
        records += struct.pack(">HHHHHH", platform, encoding, language, name_id,
                               len(encoded), len(strings))
        strings += encoded
    name = struct.pack(">HHH", 0, len(name_records), 6 + len(records)) + records + strings

    tables = {"head": head, "maxp": maxp, "hhea": hhea, "hmtx": hmtx,
              "loca": loca, "glyf": glyf, "cmap": cmap, "name": name}

    tags = sorted(tables)
    num_tables = len(tags)
    entry_selector = int(math.log2(num_tables))
    search_range = (2 ** entry_selector) * 16
    header = struct.pack(">IHHHH", 0x00010000, num_tables, search_range, entry_selector,
                         num_tables * 16 - search_range)

    offset = len(header) + num_tables * 16
    directory = b""
    body = b""
    for tag in tags:
        payload = tables[tag]
        directory += struct.pack(">4sIII", tag.encode("latin-1"), 0, offset, len(payload))
        padded = payload + b"\0" * (-len(payload) % 4)
        body += padded
        offset += len(padded)

    return header + directory + body


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------


class Checker:
    def __init__(self):
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  ok   {name}")
        else:
            self.failed.append(name)
            print(f"  FAIL {name}  {detail}")

    def near(self, name: str, value: float, expected: float, tolerance: float) -> None:
        self.check(name, abs(value - expected) <= tolerance,
                   f"got {value:.4f}, expected {expected:.4f} +-{tolerance}")


def cmd_selftest(args) -> int:
    import tempfile

    c = Checker()

    print("span coverage")
    row = [0.0] * 10
    _add_span(row, 2.0, 5.0, 1.0, 10)
    c.near("full pixels", sum(row), 3.0, 1e-9)
    c.check("exact cells", row[2] == 1.0 and row[4] == 1.0 and row[5] == 0.0)
    row = [0.0] * 10
    _add_span(row, 2.25, 2.75, 1.0, 10)
    c.near("sub-pixel span", row[2], 0.5, 1e-9)
    row = [0.0] * 10
    _add_span(row, -3.0, 12.0, 1.0, 10)
    c.near("clipping", sum(row), 10.0, 1e-9)

    print("rasterizer")
    square = [[(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]]
    cov = rasterize(square, 100, 100, samples=15)
    c.near("square area", sum(cov), 6400.0, 6400 * 0.002)
    c.check("interior is solid", cov[50 * 100 + 50] > 0.999)
    c.check("exterior is empty", cov[5 * 100 + 5] == 0.0)

    triangle = [[(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]]
    cov = rasterize(triangle, 100, 100, samples=17)
    c.near("triangle area", sum(cov), 5000.0, 5000 * 0.01)

    # A counter (hole) must vanish under the nonzero rule only if wound the other way.
    outer = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
    inner = [(30.0, 30.0), (30.0, 70.0), (70.0, 70.0), (70.0, 30.0)]  # reversed winding
    cov = rasterize([outer, inner], 100, 100, samples=15)
    c.near("ring area", sum(cov), 6400.0 - 1600.0, 6400 * 0.003)
    c.check("hole is empty", cov[50 * 100 + 50] == 0.0)

    print("bezier flattening")
    contour = [("m", (0.0, 0.0)), ("q", (50.0, 100.0), (100.0, 0.0))]
    poly = flatten_contour(contour, 1.0, 0.0, 0.0, tolerance=0.1)
    apex = min(p[1] for p in poly)  # y is negated into device space
    c.near("quadratic apex", apex, -50.0, 0.6)
    c.check("adaptive subdivision", 6 <= len(poly) <= 40, f"got {len(poly)} points")
    coarse = flatten_contour(contour, 0.05, 0.0, 0.0, tolerance=0.1)
    c.check("small curves use fewer points", len(coarse) < len(poly),
            f"{len(coarse)} vs {len(poly)}")

    print("png round-trip")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe.png")
        canvas = Canvas(7, 5, (0, 0, 0, 255))
        canvas.composite([1.0] * 35, (255, 128, 64, 255))
        write_png(path, canvas.to_rgba_bytes(), 7, 5)
        w, h, depth, colour = read_png_header(path)
        c.check("header", (w, h, depth, colour) == (7, 5, 8, 6), f"got {(w, h, depth, colour)}")
        w, h, pixels = png_idat_pixels(path)
        c.check("pixel count", len(pixels) == 7 * 5 * 4)
        c.check("colour survives", tuple(pixels[:4]) == (255, 128, 64, 255),
                f"got {tuple(pixels[:4])}")

        print("synthetic font")
        font = TrueTypeFont(build_test_font())
        c.check("units per em", font.units_per_em == 1000, f"got {font.units_per_em}")
        c.check("glyph count", font.num_glyphs == 3, f"got {font.num_glyphs}")
        c.check("family name", font.family == "Kroj Test", f"got {font.family!r}")
        c.check("cmap A", font.glyph_id("A") == 1, f"got {font.glyph_id('A')}")
        c.check("cmap B", font.glyph_id("B") == 2, f"got {font.glyph_id('B')}")
        c.check("cmap miss", font.glyph_id("Z") == 0)
        c.check("advance", font.advance(1) == 1000, f"got {font.advance(1)}")
        c.check("lsb", font.left_side_bearing(1) == 100, f"got {font.left_side_bearing(1)}")
        c.check("notdef is blank", font.glyph_contours(0) == [])

        contours = font.glyph_contours(1)
        c.check("square has one contour", len(contours) == 1, f"got {len(contours)}")
        box = contour_bbox(contours)
        c.check("square bbox", box == (100.0, 100.0, 900.0, 900.0), f"got {box}")
        c.check("all points on-curve", all(cmd[0] != "q" for cmd in contours[0]))

        tri = font.glyph_contours(2)
        c.check("triangle bbox", contour_bbox(tri) == (0.0, 0.0, 800.0, 800.0),
                f"got {contour_bbox(tri)}")

        # Rasterize the square glyph at 100px: 0.8em on a side = 80x80 pixels.
        polys = [flatten_contour(k, 100.0 / 1000.0, 0.0, 100.0) for k in contours]
        cov = rasterize(polys, 100, 100, samples=15)
        c.near("glyph raster area", sum(cov), 6400.0, 6400 * 0.005)

        print("layout")
        shaped = shape_line(font, "AB", 100.0)
        c.check("two glyphs shaped", len(shaped.glyphs) == 2)
        c.near("advance sum", shaped.width, 190.0, 1e-6)  # (1000 + 900) * 0.1
        c.check("no kerning table", font.kerning(1, 2) == 0)
        c.check("missing glyph reported", shape_line(font, "Z", 10.0).missing == ["Z"])

        print("end-to-end render")
        out = os.path.join(tmp, "text.png")
        shaped = shape_line(font, "AB", 64.0)
        polygons = text_polygons(font, shaped, 64.0, 8.0, 8.0 + 64.0 * 0.8)
        w = int(shaped.width + 16)
        h = int(64.0 + 16)
        cov = rasterize(polygons, w, h, samples=9)
        canvas = Canvas(w, h, (0, 0, 0, 255))
        canvas.composite(cov, (255, 255, 255, 255))
        write_png(out, canvas.to_rgba_bytes(), w, h)
        c.check("rendered file exists", os.path.getsize(out) > 100)
        c.check("ink was actually laid down", sum(cov) > 1000, f"coverage {sum(cov):.1f}")

    print("system fonts")
    try:
        path = default_font()
    except FontError:
        path = None
        print("  skip (no system font found)")
    if path:
        font = TrueTypeFont.load(path)
        c.check("loads real font", font.num_glyphs > 100, f"{font.num_glyphs} glyphs")
        c.check("cmap is populated", len(font.cmap) > 200, f"{len(font.cmap)} codepoints")
        gid = font.glyph_id("A")
        c.check("has an A", gid > 0)
        contours = font.glyph_contours(gid)
        c.check("A has contours", len(contours) >= 1, f"got {len(contours)}")
        box = contour_bbox(contours)
        c.check("A is a sane size", box is not None and 0 < box[3] <= font.units_per_em * 1.2,
                f"bbox {box}")

        # Ą is composite in essentially every Latin font: A plus a mark.
        gid_ogonek = font.glyph_id("Ą")
        if gid_ogonek:
            composite = font.glyph_contours(gid_ogonek)
            c.check("composite glyph resolves", len(composite) >= 2,
                    f"got {len(composite)} contours")
        else:
            print("  skip (font has no Ą)")

        pairs = [("A", "V"), ("V", "A"), ("T", "o"), ("A", "W"), ("F", "a"), ("y", ".")]
        found = [font.kerning(font.glyph_id(a), font.glyph_id(b)) for a, b in pairs]
        c.check("kerning is wired up", any(v != 0 for v in found),
                f"all zero for {pairs} (font may genuinely lack kerning)")

    print()
    total = c.passed + len(c.failed)
    if c.failed:
        print(f"FAILED {len(c.failed)}/{total}: {', '.join(c.failed)}")
        return 1
    print(f"all {total} checks passed")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kroj",
        description="A from-scratch TrueType rasterizer and glyph inspector (stdlib only).")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_font_args(p, default_needed=True):
        p.add_argument("--font", default="" if default_needed else "",
                       help="path to a .ttf/.ttc, a filename, or a family fragment")
        p.add_argument("--face", type=int, default=0, help="face index inside a .ttc")

    p = sub.add_parser("text", help="render a string to PNG")
    add_font_args(p)
    p.add_argument("--text", required=True, help=r"text to render; \n starts a new line")
    p.add_argument("--size", type=float, default=96.0, help="em size in pixels")
    p.add_argument("--out", default="kroj-text.png")
    p.add_argument("--svg", help="also write vector outlines to this SVG path")
    p.add_argument("--color", default="ink")
    p.add_argument("--bg", default="paper")
    p.add_argument("--pad", type=float, default=24.0)
    p.add_argument("--align", choices=["left", "center", "right"], default="left")
    p.add_argument("--tracking", type=float, default=0.0, help="extra px between glyphs")
    p.add_argument("--line-height", type=float, default=1.0)
    p.add_argument("--samples", type=int, default=15, help="vertical AA sub-scanlines")
    p.add_argument("--no-kerning", action="store_true")
    p.add_argument("--tight", action="store_true", help="crop to the inked bounds")
    p.set_defaults(func=cmd_text)

    p = sub.add_parser("glyph", help="annotated single-glyph diagram")
    add_font_args(p)
    p.add_argument("--char", help="character to inspect")
    p.add_argument("--gid", type=int, help="glyph id to inspect instead of a character")
    p.add_argument("--size", type=float, default=520.0, help="em size in pixels")
    p.add_argument("--out", default="kroj-glyph.png")
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--no-points", action="store_true")
    p.add_argument("--no-fill", action="store_true")
    p.set_defaults(func=cmd_glyph)

    p = sub.add_parser("info", help="dump font metadata")
    add_font_args(p)
    p.add_argument("--kern-pairs", action="store_true", help="probe a few kerning pairs")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("term", help="render text as ASCII art")
    add_font_args(p)
    p.add_argument("--text", required=True)
    p.add_argument("--cols", type=int, default=100)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--no-kerning", action="store_true")
    p.set_defaults(func=cmd_term)

    p = sub.add_parser("selftest", help="verify the pipeline against a synthetic font")
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    if getattr(args, "font", None) == "":
        try:
            args.font = default_font()
        except FontError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    try:
        return args.func(args)
    except FontError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
