"""Generate placeholder resources for the Rose installer."""
import struct
from pathlib import Path

RESOURCES_DIR = Path(__file__).parent / "resources"
RESOURCES_DIR.mkdir(exist_ok=True)


def create_placeholder_ico(filepath: Path, size: int = 32):
    """Create a minimal .ico file with a colored square."""
    # ICO header
    header = struct.pack("<HHH", 0, 1, 1)  # Reserved, Type=ICO, Count=1

    # ICO directory entry
    bmp_size = 40 + size * size * 4  # BITMAPINFOHEADER + BGRA pixels
    entry = struct.pack("<BBBBHHII",
                        size, size,  # Width, Height
                        0,  # Color count
                        0,  # Reserved
                        1,  # Color planes
                        32,  # Bits per pixel
                        bmp_size,  # Size of image data
                        22)  # Offset to image data

    # BITMAPINFOHEADER
    bih = struct.pack("<IiiHHIIiiII",
                      40,  # Header size
                      size,  # Width
                      size * 2,  # Height (doubled for ICO)
                      1,  # Planes
                      32,  # Bits per pixel
                      0,  # Compression
                      bmp_size,  # Image size
                      0, 0, 0, 0)  # Resolution, colors

    # Pixels (rose-colored square)
    pixels = b""
    for y in range(size):
        for x in range(size):
            # Rose accent color #e94560 in BGRA
            pixels += struct.pack("BBBB", 0x60, 0x45, 0xe9, 0xff)

    data = header + entry + bih + pixels
    filepath.write_bytes(data)
    print(f"Created: {filepath}")


def create_placeholder_bmp(filepath: Path, width: int = 164, height: int = 314):
    """Create a minimal BMP file for wizard image."""
    row_size = (width * 3 + 3) & ~3  # Align to 4 bytes
    pixel_data_size = row_size * height
    file_size = 54 + pixel_data_size

    # BMP Header
    bmp_header = struct.pack("<2sIHHI",
                             b"BM",
                             file_size,
                             0, 0,
                             54)

    # DIB Header
    dib_header = struct.pack("<IiiHHIIiiII",
                             40,
                             width,
                             height,
                             1,
                             24,
                             0,
                             pixel_data_size,
                             2835, 2835,
                             0, 0)

    # Pixel data (rose color)
    row = b""
    for x in range(width):
        row += struct.pack("BBB", 0x60, 0x45, 0xe9)  # BGRA -> BGR
    padding = b"\x00" * (row_size - width * 3)
    row_data = row + padding

    pixel_data = row_data * height

    data = bmp_header + dib_header + pixel_data
    filepath.write_bytes(data)
    print(f"Created: {filepath}")


if __name__ == "__main__":
    create_placeholder_ico(RESOURCES_DIR / "rose.ico")
    create_placeholder_bmp(RESOURCES_DIR / "wizard.bmp", 164, 314)
    create_placeholder_bmp(RESOURCES_DIR / "wizard_small.bmp", 55, 58)
    print("All placeholder resources created.")
