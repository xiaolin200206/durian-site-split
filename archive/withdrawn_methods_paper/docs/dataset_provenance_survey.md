# Dataset provenance survey

**Purpose.** The paper's method needs a site identifier. Most public crop
disease datasets do not release one. Establishing how many is a supplementary
table, and it answers the reviewer's first question — *why has nobody
measured this before?* The answer is that the data do not permit it.

**Status.** Partially filled from a literature search. Rows marked
`UNVERIFIED` need the dataset downloaded and its files inspected before the
table can be published. Everything below is what the paper or repository
says, not what the files contain, unless noted.

---

## How to complete a row

For each dataset, download it and check four things. Half an hour each.

1. **Filenames.** Do they encode anything beyond a class and an index? A
   site code, a plot number, a date, a plant ID?
2. **Accompanying files.** Is there a CSV, JSON or README mapping images to
   fields, plots, sessions or coordinates?
3. **EXIF.** `exiftool -GPSLatitude -DateTimeOriginal -Model` on twenty
   files. Preprocessing usually strips it; note whether any survives.
4. **The paper's split.** Does it say random, stratified random, or grouped?
   Search the methods for "split", "random", "fold".

Record the answer as **yes** (a grouped split is possible as released),
**partial** (something is recorded but not at a useful granularity — a
country, say, but not a field), or **no**.

```bash
# a quick first pass on any downloaded dataset
find . -name "*.jpg" | head -20                    # filename structure
find . -name "*.csv" -o -name "*.json" | head       # side-car metadata
exiftool -GPSLatitude -DateTimeOriginal -Model "$(find . -name '*.jpg' | head -1)"
```

---

## Table S1 — provenance metadata in public crop disease datasets

| Dataset | Year | Size | Setting | Site metadata | What is recorded | EXIF | Split in source paper |
|---|---|---|---|---|---|---|---|
| PlantVillage | 2015 | 54,305 img, 38 cls | lab, detached leaves on uniform background | **no** | leaf identity implicit (4–7 orientations per leaf); no field or plot | stripped in the common redistributions | random / stratified random |
| PlantDoc | 2020 | 2,598 img, 27 cls | internet-scraped | **no** | nothing; provenance unknown by construction | n/a | random |
| FieldPlant | 2023 | 5,170 img | field, in plantations | `UNVERIFIED` | plantation named in paper; per-image mapping unconfirmed | `UNVERIFIED` | `UNVERIFIED` |
| PlantWild | 2024 | ~50,000 img | internet-scraped | **no** | — | n/a | `UNVERIFIED` |
| PlantSeg | 2024 | 11,458 img | internet-scraped | **no** | — | n/a | `UNVERIFIED` |
| Plant Pathology (FGVC apple) | 2020–21 | ~23,000 img | field, one orchard | `UNVERIFIED` | single orchard; tree ID unconfirmed | `UNVERIFIED` | random (Kaggle) |
| Cassava Leaf Disease | 2019/2020 | ~21,000 img | field, crowdsourced | `UNVERIFIED` | collected by farmers; contributor ID unconfirmed | `UNVERIFIED` | random (Kaggle) |
| BRACOL (coffee) | 2019 | 4,407 img | field/lab | `UNVERIFIED` | | `UNVERIFIED` | `UNVERIFIED` |
| RoCoLe (coffee) | 2019 | 1,560 img | field | `UNVERIFIED` | | `UNVERIFIED` | `UNVERIFIED` |
| DiaMOS Plant (pear) | 2021 | 3,505 img | field, one orchard | `UNVERIFIED` | one orchard, multiple sessions claimed | `UNVERIFIED` | `UNVERIFIED` |
| Citrus (Rauf et al.) | 2019 | 750 img | lab | `UNVERIFIED` | | `UNVERIFIED` | `UNVERIFIED` |
| Durian, Binh Phuoc & Tien Giang | 2025 | 2,595 img, 6 cls | field, 4 orchards | **no** | class name + global index only; four orchards named in the paper, none identifiable in the files; no side-car metadata | stripped, 0/20 sampled; background removed, resized 400x400 | preset random train/test/val across all orchards |
| Durian, Vinh Long | 2025 | 5,451 img, 10 cls | field, 5 orchards | **no** | bare integer filenames; five orchards named in the paper, none identifiable in the files; no side-car metadata | absent, 0/10 sampled; released files are RGBA PNG with XMP `UserComment = Screenshot`, plus 177 JPEG including 224x224-named files | preset Train/Test/Validation across all orchards |
| **Global Wheat Head Detection 2021** | 2021 | 6,515 img | field, 12 countries | **yes** | 47 subdatasets, each one site, one date, one sensor | n/a | **grouped by continent, then by session** |
| DeepWeeds | 2019 | 17,509 img | field, 8 locations | **partial** | eight collection sites named | `UNVERIFIED` | `UNVERIFIED` |
| *This work* | 2026 | 1,314 img | field | **yes** | GPS per image; farm, tree and capture session recoverable | retained, released with GPS removed | grouped by farm; per-site reported |

**Count as it stands:** of the 15 public datasets listed, **one** (GWHD)
demonstrably supports a grouped split as released, **one** (DeepWeeds) does
so partially, **six** demonstrably do not, and **seven** are unverified.
Complete the unverified rows before quoting any proportion.

---

## What the table is for

Three sentences in the Discussion, no more:

> A grouped split requires a site identifier. Of fourteen public crop disease
> datasets surveyed (Table S1), one releases one at a granularity that permits
> it. The one that does — Global Wheat Head Detection — was built for
> domain generalisation and rejects random splitting explicitly; the disease
> detection datasets the field is built on were not, and cannot be
> retrofitted, because the metadata was not recorded at capture.

Do not overclaim beyond this. The table shows that the measurement is hard
to repeat elsewhere. It does not show that the effect exists elsewhere,
which would require doing the experiment on another dataset.

---

## The row worth completing first — done, and the answer was no

**The Vinh Long durian dataset.** Downloaded and inspected 2026-08. It does
not identify which farm an image came from, so the analysis cannot be
repeated on it and the paper's external validity gap stands.

What the inspection produced instead is the strongest row in the table. The
release is described as raw iPhone 14 photographs captured at 0.5-1 m in
natural light. The files are 5,274 RGBA PNGs whose XMP records
`exif:UserComment = Screenshot`, with no camera model, timestamp or GPS
populated in any of ten sampled files, alongside 177 JPEGs including files
named at a 224x224 resolution.

**Write about the files, not about how they came to be.** What can be
demonstrated is what the released metadata contains and what it therefore
permits. Anything about the origin of the images is inference, and the
paper's whole method is to report only what it can recompute.

The Binh Phuoc / Tien Giang dataset was inspected at the same time and is
also a clean **no**: class name plus global index, no side-car file, EXIF
stripped in 20 of 20 sampled images. It is the dataset behind the 64.1
percentage-point drop cited in the Discussion, so its row does double duty.

**Still outstanding: seven UNVERIFIED rows.** Half an hour each, protocol
above.
