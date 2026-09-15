# Submission checklist

Everything that has to be true before the upload button.
Ordered by whether it blocks.

---

## Blocking — the paper cannot go without these

### 1. Verify the six references marked VERIFY

They were located by search and their fields reconstructed. Check each
against the publisher record and correct or drop it. A reviewer who knows
one of these will check it.

| Ref | Entry | What to check |
|---|---|---|
| 2 | `lu2024plantdisease` | authors, journal, volume — PMC11466843 |
| 6 | `david2023gwc` | author list — PMC10795497 |
| 9 | `chaibubneto2019identity` | authors, volume |
| 10 | `brookshire2024eegleakage` | author list |
| 19 | Binh Phuoc / Tien Giang durian dataset | authors — *Data Brief* **61**, 111845 |
| 22 | `ogara2004durian` | the whole entry; ACIAR monograph number |
| 25 | Malaysian AI advisory rollout | see item 3 |

Reference 1 (El Jarroudi, *Nat. Sustain.* **7**, 846–854, 2024) and
reference 18 (Nguyen, *Data Brief* **63**, 112244, 2025) are verified.

### 2. Fill the remaining placeholders

- `[code DOI]` in Data availability and the cover letter — from the Zenodo
  GitHub release
- `[ORCID]` — register at orcid.org if you do not have one, it takes five
  minutes and Nature asks for it at acceptance anyway
- `[Date]` and `[names and affiliations]` in the cover letter

The dataset DOI (10.5281/zenodo.22030623) is filled.

### 3. Decide the deployment framing in the Introduction

The text says several Southeast Asian governments are scaling AI advisory
systems toward national coverage, cited to reference 25, which is a
placeholder. The specific figures found by search — a ten-week pilot with
42 farmers in Perak scaling toward ~110,000 growers — are unverified.

Either confirm from a primary source (a ministry press release, a programme
document, an FAO or World Bank report) and cite that, or cut the specific
claim and keep the framing general. **Do not submit with an unverified
citation to a live national programme.** If in doubt, cut: the argument
does not depend on naming anyone.

### 4. Publish the Zenodo dataset

Draft is prepared; four zips plus README. Check before publishing, because
files cannot be changed afterwards:

- licence is **CC BY-NC 4.0**, not plain CC BY and not the -ND variant
- all four zips uploaded and open correctly
- README.md present and unzipped
- no phone numbers, faces, or WhatsApp screenshots anywhere in the archive

---

## Should be done, does not block

### 5. Ethics determination

Requested from the Institute's Head of Research and Director; pending. The
Methods section carries a bracketed placeholder and the cover letter
discloses the position. If a determination arrives before submission, insert
it. If it arrives after, send it to the editor as an update.

Written consent from both growers is held (WhatsApp, timestamped). **Redact
the phone numbers before that evidence goes anywhere.**

### 6. Move verify.yml into .github/workflows/

Uploaded to the repository root, where GitHub Actions will not run it. Rename
it to `.github/workflows/verify.yml` in the web editor, or delete the CI
claim from the README.

### 7. Suggested reviewers

Nature asks for these. Reasonable directions: the Global Wheat Head
Detection authors (they would recognise the argument immediately); anyone
working on identity confounding in medical imaging; a tropical fruit
pathologist. Avoid anyone you have worked with.

---

## Known weaknesses — do not try to hide these

A reviewer will find them, and the paper is stronger for stating them first.
All are already in the manuscript.

- **Five effective farms.** Every dispersion figure rests on five units. The
  paper says so and calls its own numbers a lower bound.
- **No external validity.** Both public durian datasets were downloaded and
  neither releases a site identifier, so the analysis cannot be repeated on
  independent data. This is reported as a finding rather than hidden as a
  gap.
- **Single crop, single architecture, single resolution.**
- **Farm 6 differs in optics.** 2.22 mm focal length against 6.76 mm
  everywhere else, and it is also the farm that breaks the stage
  association. One farm cannot separate those.
- **Capture-burst identifiers not released.** Derived at analysis time and
  not preserved. Stated in the dataset README.
- **No deployment.** No grower has used this detector.

---

## What is already done

- Main text 5,178 words, within the 5,000-word guideline once Methods,
  abstract, references and captions are excluded; abstract 154 words
- Three main figures, four tables, one supplementary figure
- Related Work section with 25 references
- Supplementary Table 1: provenance survey of fifteen datasets
- Acknowledgements, Author contributions, Competing interests
- 113 quantitative claims verified against released tables by CI
- Dataset deposited with farm and tree identifiers; GPS stripped and
  verified on all 1,314 files
- Written consent from both growers

---

## If Nature Sustainability declines on scope

The content needs no change; only the cover letter does. In order:

1. **Nature Food** — technical validation in food systems is squarely in scope
2. **Plant Phenomics** — published GWHD; would recognise the argument immediately
3. **Computers and Electronics in Agriculture**
4. **Precision Agriculture**
5. A Datasets and Benchmarks track

The sustainability framing is real but thin, and an editor may say so. That
is not a reason to weaken the framing — it is a reason to have the second
venue ready.
