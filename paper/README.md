# SCL manuscript handoff

Fixed paper title:

**Duality and Complementarity of Vector and Interpolation-Inspired Barrier Certificates for Safety Verification: Toward Reduced Conservatism and Complexity**

The computational repository is organized around this story, but the article body is intentionally the next editing phase. The manuscript revision must cleanly separate:

1. degree-preserving structural duality for the aligned globally scaled path subclass;
2. the rotation-side degree separation favoring cyclic VBC coupling;
3. the ImplicationGap1D separation favoring implication-style IBC propagation; and
4. complexity consequences stated as transparent optimization-size considerations rather than universal runtime claims.

During the next paper pass, replace any legacy one-sided degree-reduction or proof-reuse framing with the duality/complementarity narrative in `../docs/COMPLEMENTARITY.md`. Do not alter the frozen computational evidence merely to match prose.

Standard `elsarticle`, TikZ and BibTeX are used. See `AUTHOR_CHECKLIST.md` before sending the revised manuscript to Majid and Vishnu.

Generated `main.pdf`/`main.bbl` files are build artifacts rather than canonical source. CI compiles the exact revision; the paper pass should regenerate them after the manuscript text is revised.
