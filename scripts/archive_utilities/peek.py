import pyarrow as pa

path = "/home/phinnm/Downloads/casia-hwdb2-line-train-00000-of-00002.arrow"

with pa.memory_map(path) as src:
    try:
        table = pa.ipc.open_stream(src).read_all()
    except pa.ArrowInvalid:
        table = pa.ipc.open_file(src).read_all()

print(table.schema)
print(f"\nrows: {table.num_rows}\n")

for i, row in enumerate(table.slice(0, 5).to_pylist()):
    print(f"--- row {i} ---")
    for k, v in row.items():
        if isinstance(v, dict) and "bytes" in v:      # HF Image column
            v = f"<image, {len(v['bytes'] or b'')} bytes, path={v.get('path')}>"
        elif isinstance(v, (bytes, bytearray)):
            v = f"<{len(v)} bytes>"
        print(f"{k}: {v}")
        import io
from PIL import Image
img = table.slice(0, 1).to_pylist()[0]
col = next(k for k, v in img.items() if isinstance(v, dict) and "bytes" in v)
Image.open(io.BytesIO(img[col]["bytes"])).show()