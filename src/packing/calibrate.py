"""Gate: measure detection rate per corruption at group_size = 1.

Purpose is twofold and both matter before any packing sweep is paid for:

1. **Assign detection class by measurement, not intuition.** The catalogue
   records which operations are expected to be detectable without
   comprehension. An operation whose detection rate at group_size = 1 is at
   ceiling cannot exhibit degradation under packing, and pooling it with
   operations that can would dilute a real effect toward the null.
2. **Set severity.** An operation at ceiling needs less severity; one at floor
   needs more, or removal.

Operations are then partitioned:

- ``test`` -- detection inside the measurable band; carries the primary estimand.
- ``ceiling`` -- above the band; retained as a POSITIVE CONTROL. If a
  ceiling-bound operation stops being detected at high group size, the judge has
  ceased reading rather than degraded gradually, which is a distinct failure mode.
- ``floor`` -- at or near chance; uninformative, excluded, and reported as
  excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Detection at group_size=1 must leave room to fall without sitting at chance.
BAND_LOW = 0.70
BAND_HIGH = 0.90

# Discrimination is accuracy CONDITIONAL on the judge expressing a preference:
# hits / (hits + wrong-direction). It separates the two failure modes that a raw
# detection rate conflates.
#
# Measured in the first calibration run: wrong-direction rates were ~0 across all
# twenty operations while tie rates reached 0.95. Every low-detection operation was
# therefore INSENSITIVE, not inaccurate -- the judge saw no difference rather than
# judging it backwards. Treating those as "near chance" would have excluded
# operations whose severity simply needed raising. Chance would show ~50%
# wrong-direction; these showed 0-10%.
MIN_DISCRIMINATION = 0.80
# Discrimination is a ratio over DECIDED cases only. Below this count it is noise
# and must not be reported as evidence. Measured: `jargon` produced discrimination
# figures of 0.25 / 1.00 / 0.00 / 0.67 / 0.67 across a severity ladder on 1-2
# decided cases per rung -- an apparent direction inversion that was pure sampling
# variation on top of a 90% tie rate.
MIN_DECIDED = 8
# Above this share of ties the operation is too subtle to measure at its current
# severity.
MAX_TIE_RATE = 0.35


@dataclass(frozen=True)
class Verdict:
    corruption: str
    criterion: str
    hypothesised_class: str
    n: int
    detected: int
    wrong: int
    tied: int
    role: str
    note: str

    @property
    def rate(self) -> float:
        """Detection rate: hits / n. Conflates insensitivity with error."""
        return self.detected / self.n if self.n else 0.0

    @property
    def tie_rate(self) -> float:
        return self.tied / self.n if self.n else 0.0

    @property
    def discrimination(self) -> float:
        """Accuracy given that a preference was expressed. 1.0 = never backwards."""
        decided = self.detected + self.wrong
        return self.detected / decided if decided else float("nan")

    def as_row(self) -> dict[str, Any]:
        return {
            "corruption": self.corruption,
            "criterion": self.criterion,
            "hypothesised": self.hypothesised_class,
            "n": self.n,
            "detection_rate": round(self.rate, 3),
            "tie_rate": round(self.tie_rate, 3),
            "discrimination": round(self.discrimination, 3)
            if self.detected + self.wrong
            else None,
            "role": self.role,
            "note": self.note,
        }


def classify(
    rate: float, tie_rate: float, discrimination: float, n: int
) -> tuple[str, str]:
    """Assign role from measured behaviour, distinguishing the failure modes.

    Order matters. A backwards judge is a different problem from a blind one, and
    only the first is a genuine accuracy failure.
    """
    if n < 10:
        return "insufficient", f"n={n} too small to classify"

    # With too few decided cases, discrimination cannot distinguish a backwards
    # judge from an unmeasured one; the operation is simply mute.
    return_mute = f"only {int(round(rate * n))} decided of {n}; no signal to judge"
    if tie_rate > MAX_TIE_RATE and int(round((1 - tie_rate) * n)) < MIN_DECIDED:
        return "mute", return_mute

    decided = discrimination == discrimination  # not NaN
    if decided and discrimination < MIN_DISCRIMINATION:
        return "unreliable", (
            f"judges backwards when it commits (discrimination "
            f"{discrimination:.2f}); genuine accuracy failure"
        )
    if tie_rate > MAX_TIE_RATE:
        return "insensitive", (
            f"{tie_rate:.0%} ties with discrimination "
            f"{discrimination:.2f} -- too subtle at this severity, raise it"
        )
    if rate > BAND_HIGH:
        return "ceiling", "positive control; cannot degrade, detects disengagement"
    if rate < BAND_LOW:
        return "test", "slightly below band; usable, consider raising severity"
    return "test", "in band"


def verdict_for(
    corruption_key: str,
    criterion: str,
    hypothesised_class: str,
    detected: int,
    wrong: int,
    tied: int,
) -> Verdict:
    n = detected + wrong + tied
    rate = detected / n if n else 0.0
    tie_rate = tied / n if n else 0.0
    decided = detected + wrong
    discrimination = detected / decided if decided else float("nan")
    role, note = classify(rate, tie_rate, discrimination, n)
    return Verdict(
        corruption=corruption_key,
        criterion=criterion,
        hypothesised_class=hypothesised_class,
        n=n,
        detected=detected,
        wrong=wrong,
        tied=tied,
        role=role,
        note=note,
    )


def agreement_with_hypothesis(verdicts: list[Verdict]) -> dict[str, Any]:
    """Did the measured roles match the hypothesised detection classes?

    The prediction is that `surface` operations land at ceiling and `semantic`
    operations land in band. Where measurement contradicts the hypothesis, the
    measurement governs and the discrepancy is reported.
    """
    rows = [v for v in verdicts if v.role != "insufficient"]
    if not rows:
        return {"n": 0}
    surface_ceiling = sum(
        1 for v in rows if v.hypothesised_class == "surface" and v.role == "ceiling"
    )
    insensitive = [v.as_row() for v in rows if v.role == "insensitive"]
    unreliable = [v.as_row() for v in rows if v.role == "unreliable"]
    surface_n = sum(1 for v in rows if v.hypothesised_class == "surface")
    semantic_test = sum(
        1 for v in rows if v.hypothesised_class == "semantic" and v.role == "test"
    )
    semantic_n = sum(1 for v in rows if v.hypothesised_class == "semantic")
    return {
        "n": len(rows),
        "insensitive_needing_severity": insensitive,
        "unreliable": unreliable,
        "surface_at_ceiling": f"{surface_ceiling}/{surface_n}",
        "semantic_in_test_band": f"{semantic_test}/{semantic_n}",
        "discrepancies": [
            v.as_row()
            for v in rows
            if (v.hypothesised_class == "surface" and v.role != "ceiling")
            or (v.hypothesised_class == "semantic" and v.role == "ceiling")
        ],
    }


def window_report(pairs: list[tuple[int | None, Verdict]]) -> str:
    """Locate each operation's usable severity window.

    A window requires, at one severity: tie rate at or below ``MAX_TIE_RATE`` so
    the judge registers the defect, and discrimination at or above
    ``MIN_DISCRIMINATION`` on at least ``MIN_DECIDED`` decided cases so the
    asserted direction is supported by more than noise.

    A ladder is necessary rather than simply raising severity, because severity is
    not monotone in validity. Three distinct failure modes are distinguished, and
    they imply different remedies:

    - **mute** -- almost every judgement is a tie at every severity. The operation
      does not move the targeted score at all. No remedy; the operation does not
      measure its criterion.
    - **capped** -- directionally sound but tie rate never falls far enough,
      because severity is bounded by the text (a response with three numerals
      cannot absorb eight numeral edits). Usable with reduced power.
    - **inverting** -- adequately sensitive but discrimination collapses on a
      sufficient number of decided cases: heavier corruption genuinely reverses the
      asserted direction. The operation is invalid.
    """
    by_op: dict[str, list[tuple[int | None, Verdict]]] = {}
    for sev, verdict in pairs:
        by_op.setdefault(verdict.corruption, []).append((sev, verdict))

    lines = [
        "",
        f"Severity windows (tie <= {MAX_TIE_RATE:.2f}, discrimination >= "
        f"{MIN_DISCRIMINATION:.2f} on >= {MIN_DECIDED} decided cases)",
        "",
    ]
    for op in sorted(by_op):
        rungs = sorted(by_op[op], key=lambda sv: sv[0] or 0)
        usable = []
        for sev, v in rungs:
            decided = v.detected + v.wrong
            if (
                v.tie_rate <= MAX_TIE_RATE
                and decided >= MIN_DECIDED
                and v.discrimination >= MIN_DISCRIMINATION
            ):
                usable.append(sev)

        trace = "  ".join(
            f"s{sev}:{v.rate:.2f}/{v.tie_rate:.2f}/"
            + (
                f"{v.discrimination:.2f}"
                if (v.detected + v.wrong) >= MIN_DECIDED
                else f"({v.detected + v.wrong}n)"
            )
            for sev, v in rungs
        )

        if usable:
            best = max(
                (sv for sv in rungs if sv[0] in usable),
                key=lambda sv: (sv[1].rate, -sv[1].tie_rate),
            )
            lines.append(
                f"  {op:<14} WINDOW {usable}  best s{best[0]}   [{trace}]"
            )
            continue

        # No window: classify why, since the remedies differ.
        enough = [(sev, v) for sev, v in rungs if (v.detected + v.wrong) >= MIN_DECIDED]
        if not enough:
            verdict = "MUTE (never moves the score; operation does not measure it)"
        elif all(v.discrimination >= MIN_DISCRIMINATION for _s, v in enough):
            verdict = "CAPPED (direction sound, sensitivity bounded by the text)"
        else:
            verdict = "INVERTING (direction reverses under load; invalid)"
        lines.append(f"  {op:<14} NO WINDOW -- {verdict}   [{trace}]")

    lines.append("")
    lines.append(
        "  format: s<severity>:detection/tie-rate/discrimination "
        f"(or (Nn) where decided cases < {MIN_DECIDED})"
    )
    return "\n".join(lines)
