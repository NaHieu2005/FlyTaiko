"""Native-size 1000x300 RGB gameplay input matching the web Taiko lane.

Only the rendered pixels reach the retina. No note/time fields are fed to the
policy or motor neurons; those fields are used solely to draw the game image.
"""
import numpy as np


class HighResolutionTaikoPixels:
    width, height = 1000, 300
    hit_x, hit_y = 150, 150
    speed_scale = 850 / 448

    def __init__(self, spinner_palette='blue'):
        if spinner_palette not in ('blue', 'purple'):
            raise ValueError('Unknown spinner palette')
        # Keep the historical blue palette for published v23 checkpoints.
        # The purple variant needs its own sensory cache and trained decoder.
        self.spinner_palette = spinner_palette
        self.yy, self.xx = np.mgrid[:self.height, :self.width]

    def background(self):
        image = np.empty((self.height, self.width, 3), dtype=np.uint8)
        image[:] = [16, 21, 29]
        image[97:203] = [25, 35, 51]
        image[97:99] = [61, 83, 110]
        image[201:203] = [61, 83, 110]
        self._ring(image, self.hit_x, self.hit_y, 48, 4, [242, 246, 255])
        return image

    def _disc(self, image, cx, cy, radius, fill, border, border_width=3):
        left = max(0, int(np.floor(cx - radius - 2)))
        right = min(self.width, int(np.ceil(cx + radius + 3)))
        top = max(0, int(np.floor(cy - radius - 2)))
        bottom = min(self.height, int(np.ceil(cy + radius + 3)))
        if left >= right or top >= bottom:
            return
        patch = image[top:bottom, left:right]
        distance = np.sqrt((self.xx[top:bottom, left:right] - cx) ** 2 +
                           (self.yy[top:bottom, left:right] - cy) ** 2)
        inside = distance <= radius - border_width
        edge = (distance > radius - border_width) & (distance <= radius)
        patch[inside] = fill
        patch[edge] = border
        fringe = (distance > radius) & (distance < radius + 1)
        if np.any(fringe):
            alpha = (radius + 1 - distance[fringe])[:, None]
            patch[fringe] = np.clip(patch[fringe] * (1 - alpha) +
                                    np.asarray(border) * alpha, 0, 255).astype(np.uint8)

    def _ring(self, image, cx, cy, radius, width, color):
        left = max(0, int(cx - radius - width - 1))
        right = min(self.width, int(cx + radius + width + 2))
        top = max(0, int(cy - radius - width - 1))
        bottom = min(self.height, int(cy + radius + width + 2))
        patch = image[top:bottom, left:right]
        distance = np.sqrt((self.xx[top:bottom, left:right] - cx) ** 2 +
                           (self.yy[top:bottom, left:right] - cy) ** 2)
        patch[np.abs(distance - radius) <= width / 2] = color

    def circle(self, image, note, x):
        big = note.is_big
        color = [79, 165, 228] if note.is_kat else [236, 89, 98]
        inner = [49, 116, 174] if note.is_kat else [184, 58, 70]
        radius = 42 if big else 28
        self._disc(image, x, self.hit_y, radius, color,
                   [255, 242, 176] if big else [234, 243, 252], 5 if big else 3)
        self._disc(image, x, self.hit_y, radius * .56, inner, inner, 0)

    def frame(self, game, show_keys=False):
        image = self.background()
        now = game.current_time_ms
        if not hasattr(game, '_highres_hits'):
            game._highres_hits = game.hit_notes
            game._highres_times = np.asarray([n.time_ms for n in game.hit_notes])
            game._highres_long = [n for n in game.beatmap.notes if not n.is_hit]
            game._highres_min_speed = min((n.scroll_px_per_ms for n in game.hit_notes), default=.1)
        min_speed = max(1e-6, game._highres_min_speed * self.speed_scale)
        lo = max(game.next_note_idx, int(np.searchsorted(game._highres_times, now - 200 / min_speed)))
        hi = int(np.searchsorted(game._highres_times, now + 950 / min_speed, side='right'))
        for note in game._highres_long:
            speed = note.scroll_px_per_ms * self.speed_scale
            x = self.hit_x + (note.time_ms - now) * speed
            end = self.hit_x + (note.end_time_ms - now) * speed
            if end < 70 or x > self.width + 60:
                continue
            if note.note_type == 'drumroll':
                left, right = max(self.hit_x, int(x)), min(self.width, int(end))
                if right > left:
                    image[124:176, left:right] = [213, 171, 55]
                    image[124:127, left:right] = [255, 225, 155]
                    image[173:176, left:right] = [255, 225, 155]
            else:
                centre = self.hit_x if now >= note.time_ms else x
                outer, inner = (([145, 65, 215], [87, 34, 145])
                                if self.spinner_palette == 'purple' else
                                ([79, 165, 228], [49, 116, 174]))
                self._disc(image, centre, self.hit_y, 62, outer, [234, 243, 252], 4)
                self._disc(image, centre, self.hit_y, 34, inner, inner, 0)
                for obj in game.long:
                    if obj['note'] is note:
                        fraction = obj['hits'] / max(1, note.required_hits)
                        image[219:226, self.hit_x:self.hit_x + int(100 * fraction)] = [228, 164, 255]
                        break
        # Earlier notes draw last, as in the web lane, so close stacks remain
        # readable to the extent the pixels contain information.
        for note in reversed(game._highres_hits[lo:hi]):
            x = self.hit_x + (note.time_ms - now) * note.scroll_px_per_ms * self.speed_scale
            if -50 <= x <= self.width + 50:
                self.circle(image, note, x)
        return image
