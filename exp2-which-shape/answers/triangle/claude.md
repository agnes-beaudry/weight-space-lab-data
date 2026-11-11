# Exp 2 · triangle (246a5b58) · Claude Opus 5.5

**Observations**

*Which pixels the network reads (W1):* Every input row for row 0 of the image (pixels 0–11), column 0 and column 11 has weights of about 0.01 or less. Strong first-layer weights only appear for pixels in rows 1–11 and columns 1–10. In row 1, only columns 3–8 are active. For example, row 17 (pixel (1,5)) has 0.389 into unit 19, and row 29 (pixel (2,5)) has 0.629 into unit 19.

*Bottom rows act as horizontal detectors:*
- Row 9 pixels (108–119): unit 44 gets about +0.32 to +0.36 for every pixel, and unit 31 about −0.33.
- Row 10 pixels (121–130): unit 8 gets 0.24, 0.38, 0.44, 0.45, 0.48, 0.48, 0.41; unit 25 gets 0.37 to 0.56 on the left; unit 17 gets about −0.3.
- Row 11 pixels (133–142): unit 38 gets 0.13, 0.21, 0.28, 0.31, 0.32, 0.33, 0.32, 0.29, 0.24, 0.14, peaking in the centre.

*Output bias (b3):*
- Row 0, column 0 and column 11 sit at about −0.4 to −0.5.
- Row 1 is only bright in the middle: (1,5) = −0.143 and (1,6) = −0.105.
- Row 2 widens to columns 4–8 (−0.17 to −0.01). Row 3 covers columns 2–9.
- Rows 4–8 are about 0 across columns 2–9 (e.g. +0.049, +0.063, +0.069).
- Rows 9–11 are fairly bright across the full width (−0.1 to −0.28).
- So the average image is a wedge: a point at the top centre, widening downward.

*Output units (W3):*
- **Unit 13:** a positive diagonal from (1,5) down-left to (8,1): 0.129, 0.328, 0.282, 0.368, 0.450, 0.457, 0.351.
- **Units 20 and 34:** positive diagonals from (1,6)/(1,7) down-right to (8,10), e.g. unit 34 has 0.556, 0.461, 0.619, 0.393. The pixels just inside these lines are about 0.05–0.09, so this looks like an edge moving outward over a filled area.
- **Unit 12:** row 6 is positive (0.06 to 0.30). Row 7 is negative (about −0.07 to −0.14) and rows 8–9 more so (−0.21 to −0.42). Rows 3–4 are slightly positive and row 5 is about 0.
- **Unit 11:** row 8 is positive (0.05 to 0.27) and row 9 is negative (−0.24 to −0.42).
- **Units 21, 49 and 57:** positive bars across row 10, e.g. unit 49 has 0.31, 0.45, 0.49, 0.45, 0.36.
- **Unit 1:** a solid positive patch at rows 2–4, columns 5–8 (0.10 to 0.20). Row 6 is negative and rows 7–8 are negative across the whole width (−0.13 to −0.39).
- **Unit 63 and unit 31** have large biases (0.437 and 0.546) and are probably active most of the time. They make the centre (rows 5–8, columns 3–8) brighter than the edges.

**Reasoning**

- **The shape:** The building blocks are two slanted sides meeting near the top centre and a horizontal bottom edge. The two sides go in opposite directions (down-left and down-right). The bottom edge can sit at row 6, 8, 9, 10 or 11. Together with the wedge-shaped average image, this describes an upward-pointing triangle whose base height and width vary.
- **Filled, not an outline:**
  - When unit 12 or 11 moves the base, whole rows below the new base go dark. The rows above it (interior) stay at about 0, meaning they are on in both cases.
  - Unit 1 darkens rows 6–8 across their full width.
  - The diagonal edge units push pixels on at the edge without darkening the inside.
  - The usual average image is bright in the middle.

  All of these fit a solid triangle. A thin outline would only change the base row and the two side pixels.
- **Ruled out:**
  - Digits and letters such as "A", 4, 2 or 7: there is no top bar, no curves, and the bottom row is bright in the centre.
  - Thin strokes or lines: the interior goes dark as a block.
  - Circles: the top is a sharp point, not a rounded edge.
  - Shapes touching the frame: row 0 and columns 0 and 11 are never used.

**Images:** A single solid white triangle on a black background, pointing up. The tip is near the top centre (row 1–2, columns 4–7). The sides spread downward to a flat, horizontal base, usually 6–10 pixels wide, in a row between about 6 and 11 (often low, at rows 9–11). Size, slope and horizontal position vary a little. Nothing is drawn in row 0 or in the leftmost and rightmost columns.

**Example:**
```
............
.....##.....
.....##.....
....####....
....####....
...######...
...######...
..########..
..########..
.##########.
.##########.
............
```

**Confidence:** 45%
