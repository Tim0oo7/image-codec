"""Combinations of feature families, in context order."""

import features_line
import features_match


def line_match80(m):
    return features_line.line(m) + features_match.match80(m)


def match80_line(m):
    return features_match.match80(m) + features_line.line(m)


def match120(m):
    return list(features_match.last_occurrence(m, 120))


def match80_line_hp(m):
    return features_match.match80(m) + features_line.line(m)[:2]


def line_match48(m):
    return features_line.line(m) + features_match.match48(m)


def line_match32(m):
    return features_line.line(m) + features_match.match32(m)


def line_match48_80(m):
    return features_line.line(m) + features_match.match48(m) + features_match.match80(m)
