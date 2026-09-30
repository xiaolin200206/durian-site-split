# Figure captions and the accompanying text

Paste the captions under each figure. The Results subsection below is new
text for Figure 2, which has no home in the current manuscript.

---

## Figure 1 — `fig1_per_site.pdf`

> **Figure 1 | The same weights, evaluated one site at a time.** Each marker
> is one capture site scored with weights that never saw it; error bars are
> the standard deviation across five training seeds. Left, 34 peninsular
> capture bursts, each scored by the fold whose validation set contains that
> burst's farm. Right, 12 Sabah trees with at least five images, scored by
> models trained only on peninsular data. Solid lines are the pooled figures
> those same weights produce — 0.246 in both cases. The dashed line is what a
> random image split reports for the peninsular data, 0.478. Site scores span
> 0.000 to 0.864 in the peninsula and 0.201 to 0.540 in Sabah; the pooled
> number reports none of that range, and no site score can be predicted from
> it.

**What to say about it in the text:** the two panels have the same pooled
value and visibly different spread, which is the coefficient-of-variation
contrast (0.72 against 0.28) made visual.

---

## Figure 2 — `fig2_aggregation.pdf`

> **Figure 2 | A reported figure is a statement about how many sites it
> averages over.** For each subset size k, 4,000 random subsets of k sites
> were drawn and the mean of their scores recorded; the line is the mean of
> that distribution and the shaded band its 5th-to-95th percentile interval.
> The expected value does not move with k — pooling does not make a model
> better — but the interval within which a particular evaluation can land
> narrows from 0.695 at one peninsular site to 0.106 at twenty. An evaluation
> at a single site is not a noisy estimate of the pooled figure; it is the
> quantity a grower experiences, and the pooled figure discards it.

---

## Figure 3 — `fig3_site_coverage.pdf`

> **Figure 3 | Transfer tracks how many farms a class was photographed at,
> not how many boxes it has.** Grey, AP50 under a random split of the
> peninsular data; blue, AP50 on the held-out Sabah set; a line joins the two
> evaluations of one class. The two classes present at all five farms lose
> nothing or gain. `leaf_hopper_damage`, whose 134 boxes come from a few
> capture sessions at three farms, is the best class in the dataset under a
> random split at 0.980 and reaches 0.057 in Borneo. `Phomopsis` carries
> 6,627 boxes, the most of any class, and loses 36%. Points are offset
> horizontally within each farm count for legibility.

---

## New Results subsection for Figure 2

*Insert after "One site at a time, on both islands".*

### How much is pooled determines how much the figure can move

The two panels of Fig. 1 have the same pooled value and different spread,
which suggests the pooled figure's stability is a function of how many sites
it averages. We tested this directly by resampling: for each subset size k,
we drew 4,000 random subsets of k sites and recorded the distribution of
their mean score (Fig. 2).

The expected value is flat in k. Pooling more sites does not make the model
better, and the mean of any subset size returns the same 0.31 in both
regions. What changes is how far a particular evaluation can fall from it.
In the peninsula, the 90% interval on a reported figure is 0.695 wide at one
site, 0.304 at five, and 0.106 at twenty. In Sabah, whose sites are more
alike, it is 0.339 at one tree and 0.091 at five.

The practical reading is that a single-site evaluation is not a noisy
estimate of a true underlying accuracy. It is a measurement of a different
quantity — performance at that site — and the two coincide only in the limit
where every site is alike. Our data say they are not: even in Sabah, where
thirteen trees sit under one manager in one season, one tree returns 0.201
and another 0.540.

**A caveat on what was resampled.** The curve is the sampling distribution
of the *mean of k site-level scores*, not a pooled mAP recomputed over the
union of k sites' images. The two differ in weighting: a pooled mAP weights
a site by the instances it contributes, an equal-weight mean does not. We
report the equal-weight version because the question is how much a headline
figure depends on which sites were chosen, and because instance counts vary
by an order of magnitude between our sites for reasons unrelated to
performance.

---

## Where the figures go

| Figure | Section | Replaces or supports |
|---|---|---|
| 1 | "One site at a time, on both islands" | supports the CV table; the table can stay |
| 2 | new subsection above, immediately after | nothing — this is new evidence |
| 3 | "Per-class transfer tracks site coverage" | supports the per-class table; keep both |

Three figures and four tables is heavy for a 4,000-word article. If a
reviewer asks for a cut, the per-class table is the one Figure 3 makes
redundant, not the reverse — the table has the box counts, which the figure
does not.
