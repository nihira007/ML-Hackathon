"""Metric spatial utilities (per-trip local tangent plane, ENU, meters).

Why ENU and not raw degrees: one degree of longitude is ~0.6 of a degree of latitude at
UK latitudes, so Euclidean distance in degrees is anisotropic and wrong. A per-trip ENU frame
centred on the trip's mean lat/lon is exact to well below GPS noise for city-scale trips,
needs no external dependency (pyproj is optional), and keeps every distance in meters.
"""
from __future__ import annotations

import numpy as np

_A = 6378137.0                 # WGS84 semi-major axis (m)
_E2 = 6.69437999014e-3         # WGS84 first eccentricity squared
EARTH_R = 6371008.8            # mean Earth radius for haversine (m)


class ENUProjector:
    """Geodetic (lon, lat, deg) -> local East/North (m) around a fixed origin (WGS84 exact)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        lam, phi = np.radians(self.lon0), np.radians(self.lat0)
        self._sl, self._cl, self._sp, self._cp = np.sin(lam), np.cos(lam), np.sin(phi), np.cos(phi)
        self._ecef0 = self._to_ecef(self.lon0, self.lat0)

    @staticmethod
    def _to_ecef(lon, lat):
        lam, phi = np.radians(lon), np.radians(lat)
        n = _A / np.sqrt(1.0 - _E2 * np.sin(phi) ** 2)
        x = n * np.cos(phi) * np.cos(lam)
        y = n * np.cos(phi) * np.sin(lam)
        z = n * (1.0 - _E2) * np.sin(phi)
        return np.array([x, y, z]) if np.ndim(lon) == 0 else np.stack([x, y, z], axis=-1)

    @classmethod
    def from_track(cls, track: np.ndarray) -> "ENUProjector":
        return cls(float(np.mean(track[:, 0])), float(np.mean(track[:, 1])))

    def to_enu(self, lon, lat) -> np.ndarray:
        """-> (N, 2) array of east, north meters."""
        d = self._to_ecef(np.asarray(lon, float), np.asarray(lat, float)) - self._ecef0
        e = -self._sl * d[..., 0] + self._cl * d[..., 1]
        n = -self._sp * self._cl * d[..., 0] - self._sp * self._sl * d[..., 1] + self._cp * d[..., 2]
        return np.stack([e, n], axis=-1)

    def project(self, lonlat: np.ndarray) -> np.ndarray:
        return self.to_enu(lonlat[:, 0], lonlat[:, 1])


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in meters (vectorised)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def point_to_segment(p, a, b):
    """Perpendicular (clamped) distance from point(s) p to segment(s) a->b, plus fraction t in [0, 1].

    Arguments broadcast against each other; last axis is (x, y). Degenerate (zero length)
    segments give t = 0 and the distance to ``a``.
    """
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    ab2 = np.sum(ab * ab, axis=-1)
    safe = np.where(ab2 > 0, ab2, 1.0)
    t = np.clip(np.sum((p - a) * ab, axis=-1) / safe, 0.0, 1.0)
    t = np.where(ab2 > 0, t, 0.0)
    proj = a + t[..., None] * ab
    d = np.linalg.norm(p - proj, axis=-1)
    return d, t


def bearing_deg(dx, dy):
    """Compass bearing in [0, 360): 0 = north, 90 = east (from ENU deltas)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def segment_bearing(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    return bearing_deg(d[..., 0], d[..., 1])


def angular_diff_deg(b1, b2):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(b1, float) - np.asarray(b2, float)) % 360.0)
    return np.where(d > 180.0, 360.0 - d, d)
