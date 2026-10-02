"""Training-only label regions aligned with Great windows and key release.

Never imported by a live policy. Timestamps are exclusively supervised labels.
"""
import numpy as np
from flytaiko.neural_game import taiko_windows

CATEGORY={'don':1,'kat':2,'don_big':3,'kat_big':4}

def circle_regions(row,pulse_labels,step=8.):
    result=np.array(pulse_labels,dtype=np.int8,copy=True)
    circles=sorted((n for n in row['notes'] if n['type'] in CATEGORY),key=lambda n:n['t'])
    longs=[n for n in row['notes'] if n['type'] not in CATEGORY]
    half=min(24.,taiko_windows(row['metadata'].get('overall_difficulty',5))['great']-step/2)
    # Frame zero observes at -8ms and is judged at 0ms, not +8ms.
    times=np.arange(len(result))*step
    for i,n in enumerate(circles):
        before=circles[i-1]['t'] if i else -float('inf')
        after=circles[i+1]['t'] if i+1<len(circles) else float('inf')
        left=max(n['t']-half,(before+n['t'])/2+step/2)
        right=min(n['t']+half,(after+n['t'])/2-step/2)
        allowed=(times>=left)&(times<=right)
        # The evaluator routes long-object presses first; do not label a circle
        # press in a frame that would actually be consumed by a drumroll/swell.
        for long in longs:
            end=long['end_t']+(long.get('tick_spacing_ms',0)/2 if long['type']=='drumroll' else 0)
            allowed&=~((times>=long['t'])&(times<=end))
        result[allowed]=CATEGORY[n['type']]
    return result
