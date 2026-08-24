import GEOparse

gse = GEOparse.get_GEO(geo="GSE31210", destdir="geo_cache", include_data=False)

# Print characteristics_ch1 for first 10 samples
count = 0
for gsm_id, gsm in gse.gsms.items():
    if count >= 10:
        break
    meta = gsm.metadata or {}
    chars = meta.get("characteristics_ch1", [])
    print(f"\n{gsm_id}:")
    if isinstance(chars, list):
        for i, c in enumerate(chars):
            print(f"  [{i}] {repr(c)[:200]}")
    else:
        print(f"  {repr(chars)[:200]}")
    count += 1
