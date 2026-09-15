# Durian leaf disease dataset with farm, tree and capture-session identifiers

1,314 annotated field photographs of durian (*Durio zibethinus*) foliage and
pests, from Peninsular Malaysia and Sabah, Malaysian Borneo.

**This dataset releases a site identifier at three levels — farm, tree and
capture session — so that images from one management unit can be kept out of
a test set.** That is the point of it. Most public crop disease datasets do
not record this, which means a grouped split cannot be constructed from them
however much anyone might want to, and the accuracy figures computed on them
cannot be checked for site leakage. This one can be.

DOI: https://doi.org/10.5281/zenodo.22030622
Licence: CC BY-NC 4.0

---

## What is here

```
peninsula_images.zip   1,033 images, Peninsular Malaysia
sabah_images.zip         281 images, two orchards near Lahad Datu, Sabah
labels.zip               YOLO-format annotations, one .txt per image
metadata.zip             the identifiers and per-farm covariates
README.md                this file
LICENSE.txt
```

### metadata.zip

| File | What it gives you |
|---|---|
| `split_assignment.csv` | per image (560 with a farm): farm, how the farm was attributed, and membership in the random split and the five by-farm folds used in the paper |
| `sabah_tree_ids.csv` | per Sabah image (281): orchard and tree identifier, thirteen trees |
| `peninsula_no_farm.csv` | the 473 peninsular images with no recoverable farm |
| `farm_annotation_scale.csv` | median annotated box area per class per farm |
| `farm_capture_conditions.csv` | per farm: capture hour, solar elevation, midday share, devices, ISO, focal length |

**Capture-burst identifiers are not released.** The paper groups peninsular
images into bursts — consecutive frames less than 15 s apart — as a proxy for
individual trees, but that grouping was derived at analysis time from EXIF
timestamps and the per-image assignment was not preserved. Aggregate per-burst
scores are in the paper's code repository. Farm and tree identifiers, on which
the paper's central comparison rests, are released in full and are sufficient
to reconstruct every split the paper reports.

### Classes

Six, identical in both regions and in the same order:

```
0 Algal              2,312 boxes peninsula / 819 Sabah
1 Leaf_rot             570 / 362
2 Phomopsis          6,627 / 620
3 Psyllid            3,472 / 921
4 Psyllid_damage     2,258 / 666
5 leaf_hopper_damage   134 / 12
```

The class list mixes a fungal genus name with symptom names. This is the
field's usage, not a claim about pathogen identity: an annotator choosing
between `Phomopsis` and `Leaf_rot` from a photograph is not identifying an
organism, and a *Phomopsis* infection can present as leaf rot. The names are
retained because they are the names growers and extension services use and
under which fungicides are registered.

---

## How it was collected

All images were captured hand-held under natural light by one person, and
annotated and quality-controlled by the same person. **No background removal
or colour correction was applied at any stage.** Annotation is lesion-level:
each visible lesion or insect has its own box.

**Images are released at 640 x 640.** That is the resolution and aspect ratio
used for training, and it is the form on which every number in the paper was
computed — which is why it is the form released. The originals are at full
phone resolution (4:3) and are available from the author on request. If your
question needs the originals, ask; if your question is whether the paper's
numbers hold up, these are the files that produced them.

- Peninsular Malaysia (Selangor, Negeri Sembilan, Muar district of Johor),
  December 2025 – July 2026, seven farms.
- Sabah, two orchards near Lahad Datu, 13–16 August 2026, thirteen trees.
- Primary device iPhone 16 Pro Max; an iPhone 13 Pro at three peninsular
  farms. One farm was photographed at a 2.22 mm focal length where the others
  used 6.76 mm — see `farm_capture_conditions.csv` before any analysis that
  depends on apparent target size.

### Farm attribution

Farms were recovered from EXIF GPS by single-link clustering at 1.5 km.
Images from the second handset, which recorded no coordinates, were
attributed by capture-time proximity: on a multi-farm day visits are
sequential, so an image sits in time between GPS-bearing images of the farm
then being visited, and attribution was accepted only within 30 minutes of
the nearest same-day GPS-bearing image. 560 of 1,033 peninsular images were
attributed (427 by GPS, 133 by timing); the attribution method is recorded
per image in `split_assignment.csv`.

**GPS coordinates have been stripped from the released EXIF.** The farm
labels remain, so grouped splits are fully reconstructible; the orchards are
not locatable.

---

## Known limitations

Read these before using the data.

- **473 peninsular images have no recoverable farm** (listed in
  `peninsula_no_farm.csv`). They are released with their annotations, but
  carry no farm label and are excluded from every grouped analysis in the
  paper. They are not a random sample of the corpus: they are an early batch
  transferred by instant messaging, which strips EXIF, plus a second handset
  with location services off.
- **Two farm clusters contain a single image each.** Five farms are of
  usable size.
- **One annotator, no inter-annotator agreement.** Protocol consistency is a
  strength; the absence of a second opinion is a weakness. For two classes we
  expect disagreement would be high. Images whose only annotation was a
  single psyllid egg were withdrawn during quality control, because at the
  magnification needed to see one it was not separable from reflections,
  trichomes and water droplets. Those photographs are retained; only the
  annotations were withdrawn.
- **`leaf_hopper_damage` is not interpretable on its own.** 134 boxes in the
  peninsula from a few sessions on adjacent trees, 12 in Sabah from two
  trees. It is useful as a contrast between evaluation protocols, not as a
  measurement.
- **Two framings occur** — a leaf held on the tree, and a detached leaf laid
  on the palm — both within single farms, and which is which was not
  recorded.
- **Released at training resolution, not as originals.** 640 x 640 square,
  from 4:3 originals. Fine detail on the smallest targets — single psyllid
  eggs, 4-pixel Phomopsis lesions — is limited by this, as it was for the
  models reported in the paper.
- **Disease stage is not recorded.** It is nonetheless present in the data:
  median `Leaf_rot` box area differs 92.8-fold between the extreme farms
  because early and spread presentations share one label. This is a property
  of every crop disease dataset we know of, including this one.

---

## Suggested use

The obvious use is to check whether an accuracy figure survives a grouped
split. `split_assignment.csv` gives the five by-farm folds used in the paper
and the random split they are compared against, over an identical 560-image
pool, so the split rule is the only difference between the two regimes.

Other groupings the identifiers support and we did not use: by tree, by
region, and leave-one-orchard-out within Sabah.

If you use this dataset, please cite the paper. As a courtesy, letting the
author know where it ended up is appreciated but not required.

---

## Licence

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
You may share and adapt the material for non-commercial purposes with
attribution. Commercial use is not permitted.
