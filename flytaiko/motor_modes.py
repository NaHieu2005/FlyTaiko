"""Learned measured-motor categories and explicit actuator repeat semantics.

No beatmap or game timestamp enters this interface. Simulation step duration
only integrates the actuator refractory interval, like an ordinary key repeat.
This engineered actuator is NOT an additional measured MaleCNS neuron.
"""
import numpy as np
from torch import nn
from flytaiko.motor_policy import MotorPolicy,KeyInterface
from flytaiko.circle_region_labels import CATEGORY

class ModeMotorPolicy(MotorPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.net[-1]=nn.Linear(128,7)

class ModeKeyInterface(KeyInterface):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.repeat_elapsed=1e9;self.mode=0;self.spinner_color=0
        self.armed_category=0;self.armed_ms=0.;self.last_circle_category=0

    def decide(self,probabilities,delta_ms=8.):
        p=np.asarray(probabilities);self.repeat_elapsed+=delta_ms
        category=int(np.argmax(p));confidence=float(p[category])
        if category in (5,6) and confidence>=self.threshold:
            self.mode=category;self.latched=False;self.armed_category=0;self.armed_ms=0.
            if self.repeat_elapsed<16:return 0
            self.repeat_elapsed=0
            color='don' if category==5 or not self.spinner_color else 'kat'
            if category==6:self.spinner_color=1-self.spinner_color
            action=(1 if color=='don' else 3)+int(self.right[color]);self.right[color]=not self.right[color]
            return action
        self.mode=0
        circle=np.concatenate(([p[0]+p[5]+p[6]],p[1:5]))
        hit=1-float(circle[0]);circle_category=int(np.argmax(circle[1:]))+1
        if hit<self.release:
            self.latched=False;self.armed_category=0;self.armed_ms=0.
            return 0
        if hit<self.threshold:
            self.armed_category=0;self.armed_ms=0.
            return 0
        if circle_category!=self.armed_category:
            self.armed_category=circle_category;self.armed_ms=0.
        else:self.armed_ms+=delta_ms
        # The learned class becomes positive up to 12ms before a circle.
        # Requiring a second consistent observation targets the note centre,
        # and rejects one-frame noise without any map-clock/label oracle.
        if self.armed_ms<8.:return 0
        if self.latched and circle_category!=self.last_circle_category and self.repeat_elapsed>=16.:
            # A changed color can re-arm even when two dense notes leave only
            # one low-confidence frame. Same-color repeats still need release.
            self.latched=False
        action=super().decide(circle)
        if action:
            self.last_circle_category=circle_category;self.repeat_elapsed=0.
        return action

def mode_training_label(game):
    """Training-only labels. A live policy never calls this function."""
    t=game.current_time_ms;step=game.step_ms
    for obj in game.long:
        n=obj['note'];end=n.end_time_ms+(n.tick_spacing_ms/2 if n.note_type=='drumroll' else 0)
        if obj['finished'] or not n.time_ms<=t<=end:continue
        if n.note_type=='drumroll':return 5
        if obj['hits']<n.required_hits:return 6
    notes=game.hit_notes
    i=np.searchsorted([n.time_ms for n in notes],t)
    for index in (i-1,i):
        if not 0<=index<len(notes):continue
        n=notes[index];half=min(12.,game.windows['great']-step/2)
        left=n.time_ms-half;right=n.time_ms+half
        if index:left=max(left,(notes[index-1].time_ms+n.time_ms)/2+4)
        if index+1<len(notes):right=min(right,(notes[index+1].time_ms+n.time_ms)/2-4)
        if left<=t<=right:return CATEGORY[n.note_type]
    return 0

def fast_long_teacher(game,default):
    """Training-only spinner trajectories matched to the physical actuator."""
    for obj in game.long:
        n=obj['note'];t=game.current_time_ms
        if n.note_type=='swell' and not obj['finished'] and n.time_ms<=t<=n.end_time_ms and obj['hits']<n.required_hits:
            if t-game.last_press<16:return 0
            color='don' if obj['hits']%2==0 else 'kat'
            return (2 if game.don_alt else 1) if color=='don' else (4 if game.kat_alt else 3)
    return default(game)
