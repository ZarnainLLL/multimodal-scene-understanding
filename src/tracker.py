# Centroid-based tracker: matches detections across frames using Euclidean distance.

import math
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, max_disappeared=5, max_distance=0.2):
        self.next_id = 0
        self.objects = OrderedDict()     # id -> (class, centroid)
        self.disappeared = OrderedDict() # id -> frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        if not detections:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]; del self.disappeared[oid]
            return []

        new_centroids = [(d[1][0], d[1][1]) for d in detections]

        if not self.objects:
            return [self._register(cls, c, bbox) for (cls, bbox), c in zip(detections, new_centroids)]

        eids = list(self.objects)
        ecs = [self.objects[oid][1] for oid in eids]

        pairs = sorted(
            [(i, j, math.dist(ecs[i], new_centroids[j]))
             for i in range(len(eids)) for j in range(len(new_centroids))],
            key=lambda x: x[2]
        )

        matched_e, matched_n, results = set(), set(), []
        for ei, ni, dist in pairs:
            if ei in matched_e or ni in matched_n or dist > self.max_distance:
                continue
            oid = eids[ei]
            if self.objects[oid][0] != detections[ni][0]:
                continue
            self.objects[oid] = (detections[ni][0], new_centroids[ni])
            self.disappeared[oid] = 0
            results.append((oid, detections[ni][0], detections[ni][1]))
            matched_e.add(ei); matched_n.add(ni)

        for ei, oid in enumerate(eids):
            if ei not in matched_e:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]; del self.disappeared[oid]

        for ni, (cls, bbox) in enumerate(detections):
            if ni not in matched_n:
                results.append(self._register(cls, new_centroids[ni], bbox))

        return results

    def _register(self, cls, centroid, bbox):
        oid = self.next_id
        self.objects[oid] = (cls, centroid)
        self.disappeared[oid] = 0
        self.next_id += 1
        return (oid, cls, bbox)