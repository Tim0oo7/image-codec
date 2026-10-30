"""Harness cost functions for CTW with a trained starting table (see ctw_prior.py)."""

from pathlib import Path

import ctw_prior

CORPUS = Path(__file__).resolve().parents[2] / "corpus"


def _t(name):
    return str(CORPUS / f"table_{name}.npy")


def base_t12_s4(m):
    return ctw_prior.cost(m, "none@24", _t("base12"), 12, 4)


def base_t16_s4(m):
    return ctw_prior.cost(m, "none@24", _t("base16"), 16, 4)


def line_t12_s4(m):
    return ctw_prior.cost(m, "features_line:line@24/2", _t("line12"), 12, 4)


def line_t16_s4(m):
    return ctw_prior.cost(m, "features_line:line@24/2", _t("line16"), 16, 4)
