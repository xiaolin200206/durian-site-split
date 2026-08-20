> **Status: inserted into the manuscript.** Kept here as the drafting
> source. One sentence was removed before insertion — a reference to a
> shortcut diagnostic following Noyan's design. `scripts/diagnostics/
> shortcut_check.py` runs on the superseded twelve-class taxonomy
> (Pink_disease, Root_disease, Scale_insect), so it does not test the
> model this paper reports. Re-run it on the six-class model or leave
> the claim out.

---

# Related work

*Insert after the Introduction, before Results. Roughly 600 words.*

*Every claim about prior work here comes from a literature search and needs
its citation verified against the publisher record before submission. The
entries are in `docs/references.bib` with the ones needing checks marked.*

---

## Related work

**The concern has been stated; it has not been measured.** A 2024 review of
plant disease recognition datasets identifies random splitting as the field's
dominant practice and names its principal defect: several images taken of one
observation, differing only slightly, can be assigned to training and test
sets, so test performance is overestimated [1]. That is the mechanism this
paper examines. What the review does not do — and what we are not aware of
anyone doing on field-collected disease imagery — is measure how large the
overestimate is, or ask what remains once it is removed. The contribution
here is not the observation that random splitting leaks. It is the
measurement, and what the measurement exposes about the corrected figure.

**A related bias in the same datasets is already quantified.** Noyan trained
a classifier on eight background pixels from PlantVillage images and reached
49.0% accuracy against a 2.6% chance baseline, demonstrating that
label-correlated signal exists outside the leaf [2]. That result concerns
background shortcut rather than site leakage, but it establishes the same
point in a weaker form: aggregate accuracy on these datasets can be produced
by information that has nothing to do with the disease.

**Grouped splitting is already standard in one agricultural vision task.**
The Global Wheat Head Detection dataset organises its images into
subdatasets defined as a consistent set acquired over the same experimental
unit, during the same acquisition session, with the same vector and sensor —
47 such domains in the 2021 release [3,4]. Its authors state the reasoning
directly: although random splitting is the common practice, the competition
aims to test performance on unseen genotypes, environments and observational
conditions, so images are grouped by continent instead [3]. The 2021
competition went further and made validation and test sets entirely disjoint
by session, where the 2020 edition had drawn them from shared subdatasets
[5]. WILDS subsequently adopted the dataset as a domain-generalisation
benchmark in which a domain *is* an acquisition session [6].

The precedent is therefore established in agricultural imaging — for wheat
head counting. It has not reached disease detection, where the datasets the
field is built on do not record the metadata that would make it possible
(Table S1). GWHD reports per-session error distributions but does not treat
the dispersion between sessions as the quantity of interest; we do.

**The failure has a name and a remedy in adjacent fields.** In medical
imaging and physiological signal analysis it is called identity confounding:
a model learns to recognise the subject alongside the diagnostic feature, so
record-wise cross-validation inflates accuracy relative to subject-wise
[7,8]. It has been documented in structural and functional MRI, optical
coherence tomography, accelerometry, voice recordings and EEG, in deep
networks and in random forests alike [9,10]. Patient-level partitioning is
now routine, and its absence is treated as a methodological defect rather
than a design choice [11,12].

Our contribution relative to that literature is not the diagnosis. It is
that the agricultural analogue of a patient is not obvious. A medical dataset
has an unambiguous grouping key. A field dataset could be grouped by leaf,
plant, plot, farm, session, or region, and the choice changes the answer: we
measure a 48.5% inflation at farm level and a considerably larger dispersion
at capture-site level. Which unit is correct depends on what the deployed
system will encounter, and for a handheld tool used at one orchard the
relevant unit is the one the field currently does not record.

---

### Citation keys for the numbered references above

1. `lu2024plantdisease`
2. `noyan2022uncovering`
3. `david2020gwhd`
4. `david2021gwhd`
5. `david2023gwc`
6. `koh2021wilds`
7. `saeb2017needtoconsider`
8. `chaibubneto2019identity`
9. `brookshire2024eegleakage`
10. `tampu2022octleakage`
11. `varoquaux2022machine`
12. `kapoor2023leakage`

---

### A note on how this changes the paper's claims

Adding this section forces two edits elsewhere, and both are improvements:

**The Introduction currently says the problem "is understood in principle"
without saying by whom.** Replace with a direct citation to [1] and the
statement that it has not been quantified on field imagery. A reviewer who
knows that review will otherwise assume you do not.

**Done. The Discussion's first claim was "random splitting overstates
accuracy substantially."** That is now a confirmation of a stated concern,
not a discovery, and should be phrased as such. The novel claims are the
second, third and fourth: that the corrected figure is unstable, that
transfer tracks site coverage rather than annotation volume, and that a
single label spans a 93-fold size range because disease stage is unrecorded.
Leading with the confirmed one weakens the paper.
