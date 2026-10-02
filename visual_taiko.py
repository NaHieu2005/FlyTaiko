"""All-object taiko evaluator and RGB gameplay observation renderer.

Rendering is an offscreen gameplay viewport, not a note-feature encoder.
Policy input is pixels only. Expert timestamps are training labels only.
"""
import numpy as np
from pathlib import Path
import hashlib
from taiko.parser import TaikoNote, TaikoBeatmap, BeatmapMetadata
from neural_game import Game, metrics

def beatmap_from_json(row):
    if row.get('source') and any('scroll_px_per_ms' not in n for n in row['notes']):
        from taiko.parser import parse_osu_text
        data=Path(row['source']).read_bytes()
        if row.get('source_sha256') and hashlib.sha256(data).hexdigest()!=row['source_sha256']:
            raise ValueError('Native SV source hash mismatch')
        original=parse_osu_text(data.decode('utf-8-sig'))
        if original is None:raise ValueError('SV source is not native taiko')
        speeds={(n.time_ms,n.note_type):n.scroll_px_per_ms for n in original.notes}
        offset=row.get('excerpt_start_ms',0)
        for n in row['notes']:
            n['scroll_px_per_ms']=speeds[(n['t']+offset,n['type'])]
    metadata=BeatmapMetadata(**{k:v for k,v in row['metadata'].items()
                                 if k in BeatmapMetadata.__dataclass_fields__})
    notes=[TaikoNote(float(n['t']),n['type'],float(n.get('end_t',0)),
                    float(n.get('tick_spacing_ms',0)),int(n.get('required_hits',0)),
                    bool(n.get('strong',False)),float(n.get('scroll_px_per_ms',0))) for n in row['notes']]
    for n in notes:
        if not n.is_hit and (n.end_time_ms<=n.time_ms or
           (n.note_type=='drumroll' and n.tick_spacing_ms<=0) or
           (n.note_type=='swell' and n.required_hits<1)):
            raise ValueError('Long object missing duration/ticks; original .osu is required')
    return TaikoBeatmap(metadata,sorted(notes,key=lambda n:n.time_ms))

class VisualGame(Game):
    def __init__(self,beatmap,step_ms=8):
        self.long=[];self.long_results=[];self.last_action=0;self.last_press=-99999.
        super().__init__(beatmap,step_ms)
        for i,n in enumerate(beatmap.notes):
            if n.is_hit:continue
            ticks=np.arange(n.time_ms,n.end_time_ms+n.tick_spacing_ms/2,n.tick_spacing_ms) if n.note_type=='drumroll' else []
            self.long.append({'note':n,'index':i,'ticks':ticks,'hit_ticks':set(),
                              'hits':0,'last_color':None,'finished':False})

    @property
    def is_done(self):
        return self.next_note_idx>=len(self.hit_notes) and all(o['finished'] for o in self.long)

    def expire(self):
        while self.next_note_idx<len(self.hit_notes) and self.current_time_ms>self.hit_notes[self.next_note_idx].time_ms+self.windows['miss']:
            self.record('miss',0,None,'unhit')
        for obj in self.long:
            n=obj['note'];end=n.end_time_ms+(n.tick_spacing_ms/2 if n.note_type=='drumroll' else 0)
            if not obj['finished'] and self.current_time_ms>end:
                obj['finished']=True
                self.long_results.append({'object_index':obj['index'],'type':n.note_type,
                    'hits':obj['hits'],'required':len(obj['ticks']) if n.note_type=='drumroll' else n.required_hits,
                    'completed':obj['hits']>=(len(obj['ticks']) if n.note_type=='drumroll' else n.required_hits)})

    def hit(self,action):
        if not action:return
        if action not in range(1,7):raise ValueError('Invalid action')
        self.expire();self.last_action=action;self.last_press=self.current_time_ms
        for obj in self.long:
            n=obj['note'];t=self.current_time_ms
            if obj['finished'] or t<n.time_ms:continue
            if n.note_type=='swell' and t<=n.end_time_ms and obj['hits']<n.required_hits:
                self.update_hand(action)
                color='don' if action in (1,2,5) else 'kat'
                valid=color!=obj['last_color']
                if valid:obj['hits']+=1;obj['last_color']=color
                self.actions.append({'time_ms':t,'model_time_ms':t,'action_id':action,
                                     'judgment':'swell_tick' if valid else 'ignore','object_index':obj['index']})
                return
            if n.note_type=='drumroll' and t<=n.end_time_ms+n.tick_spacing_ms/2:
                self.update_hand(action)
                # A press may hit at most one unjudged drumroll tick.
                available=[i for i,v in enumerate(obj['ticks']) if i not in obj['hit_ticks'] and abs(t-v)<=n.tick_spacing_ms/2]
                valid=bool(available)
                if valid:
                    tick=min(available,key=lambda i:abs(t-obj['ticks'][i]));obj['hit_ticks'].add(tick);obj['hits']+=1
                self.actions.append({'time_ms':t,'model_time_ms':t,'action_id':action,
                                     'judgment':'drumroll_tick' if valid else 'ignore','object_index':obj['index']})
                return
        if self.next_note_idx>=len(self.hit_notes):
            self.update_hand(action)
            self.false_hits+=1
            self.actions.append({'time_ms':self.current_time_ms,'model_time_ms':self.current_time_ms,
                                 'action_id':action,'judgment':'ignore'})
            return
        super().hit(action)

    def update_hand(self,action):
        if action in (1,2,3,4):
            color='don' if action<3 else 'kat';hand=action in (2,4)
            self.alt_violations+=int(self.last_hand[color]==hand)
            self.last_hand[color]=hand
            if color=='don':self.don_alt=not hand
            else:self.kat_alt=not hand

    def expert(self,anticipation_ms=0):
        if self.next_note_idx<len(self.hit_notes):
            action=super().expert(anticipation_ms)
            if action:return action
        for obj in self.long:
            n=obj['note'];t=self.current_time_ms
            if obj['finished'] or t<n.time_ms:continue
            if n.note_type=='drumroll':
                if any(i not in obj['hit_ticks'] and abs(t-v)<=self.step_ms/2 for i,v in enumerate(obj['ticks'])):
                    return 2 if self.don_alt else 1
            elif t<=n.end_time_ms and obj['hits']<n.required_hits:
                # Teacher presses at 16ms cadence. Runtime never reads this schedule.
                if t-self.last_press>=16:
                    return (2 if self.don_alt else 1) if obj['hits']%2==0 else (4 if self.kat_alt else 3)
        return 0

def all_metrics(games):
    result=metrics(games);long=[o for g in games for o in g.long_results]
    events=[e for g in games for e in g.events]
    result['size_match_rate']=sum(e.get('size_correct',False) for e in events)/max(1,len(events))
    for kind in ('drumroll','swell'):
        rows=[r for r in long if r['type']==kind]
        result[kind]={'objects':len(rows),'completed':sum(r['completed'] for r in rows),
                      'hits':sum(r['hits'] for r in rows),'required':sum(r['required'] for r in rows),
                      'coverage':sum(r['hits'] for r in rows)/max(1,sum(r['required'] for r in rows))}
    return result

class GameplayPixels:
    width,height=512,96
    visible_ms=1200.
    def __init__(self):
        self.yy,self.xx=np.mgrid[:self.height,:self.width]

    def background(self):
        rgb=np.zeros((self.height,self.width,3),dtype=np.uint8);rgb[:]=[8,12,20]
        rgb[46:51]=[28,35,45];rgb[:,62:65]=[180,180,180]
        return rgb

    def circle(self,rgb,n,x,y):
        radius=14 if n.is_big else 9
        left=max(0,int(x-radius));right=min(self.width,int(x+radius)+2)
        top=max(0,y-radius);bottom=min(self.height,y+radius+1)
        if right<=left:return
        patch=rgb[top:bottom,left:right]
        distances=(self.xx[top:bottom,left:right]-x)**2+(self.yy[top:bottom,left:right]-y)**2
        patch[distances<=radius**2]=[245,65,70] if n.is_don else [45,155,245]
        patch[distances<(radius-3)**2]=[130,30,35] if n.is_don else [20,75,130]

    def frame(self,game,show_keys=False):
        rgb=self.background()
        now=game.current_time_ms;speed=(self.width-64)/self.visible_ms
        # Binary-search circles in view instead of scanning thousands of objects
        # on every frame. Long objects remain visible while their tail is on screen.
        if not hasattr(game,'_render_hits'):
            game._render_hits=[n for n in game.beatmap.notes if n.is_hit]
            game._render_times=np.array([n.time_ms for n in game._render_hits])
            game._render_long=[n for n in game.beatmap.notes if not n.is_hit]
        # Slow-SV notes can enter much earlier; a fixed time window hides them.
        if not hasattr(game,'_render_min_speed'):
            game._render_min_speed=min((n.scroll_px_per_ms or speed for n in game._render_hits),default=speed)
        lo=np.searchsorted(game._render_times,now-24/game._render_min_speed)
        hi=np.searchsorted(game._render_times,now+(self.width-40)/game._render_min_speed,side='right')
        for n in game._render_hits[lo:hi]+game._render_long:
            note_speed=n.scroll_px_per_ms or speed
            x=64+(n.time_ms-now)*note_speed;y=48
            if n.is_hit:
                if x<40 or x>self.width+16:continue
                self.circle(rgb,n,x,y)
            else:
                end=64+(n.end_time_ms-now)*note_speed
                if end<48 or x>self.width:continue
                left=max(64,int(x));right=min(self.width,int(end))
                if n.note_type=='drumroll':
                    rgb[37:60,left:right]=[230,180,25]
                    for tick in np.arange(n.time_ms,n.end_time_ms+n.tick_spacing_ms/2,n.tick_spacing_ms):
                        tx=int(64+(tick-now)*note_speed)
                        if 64<=tx<self.width:rgb[43:54,max(0,tx-2):min(self.width,tx+3)]=[255,250,180]
                else:
                    rgb[30:66,left:right]=[145,65,215]
                    # Visible spinner progress; sourced from evaluator, not future labels.
        # One canonical active-spinner indicator; completed overlapping swells
        # must not paint a full bar over a later, still-unfinished spinner.
        spinners=[o for o in game.long if o['note'].note_type=='swell' and
                  o['note'].time_ms<=now<=o['note'].end_time_ms]
        if spinners:
            obj=next((o for o in spinners if o['hits']<o['note'].required_hits),spinners[-1])
            fraction=obj['hits']/obj['note'].required_hits
            # Actual player-progress feedback: a completed swell turns green.
            # A one-pixel length change is otherwise invisible at retinal-column
            # resolution for large required-hit counts. No future labels or
            # teacher actions enter this displayed completion cue.
            color=[80,245,130] if obj['hits']>=obj['note'].required_hits else [235,220,255]
            rgb[69:74,64:64+int(100*fraction)]=color
        # Key lights are UI telemetry, not retinal observations. Teacher actions
        # otherwise create an avoidable teacher/student feedback distribution shift.
        if show_keys and now-game.last_press<24:
            action=game.last_action
            for i in range(4):
                if action==i+1 or (action==5 and i<2) or (action==6 and i>=2):
                    rgb[80:94,10+i*12:20+i*12]=[255,245,130]
        return rgb


class DefaultSkinGameplayPixels(GameplayPixels):
    """Deterministic 512x96 version of the web player's default note style.

    It changes pixels only, never beatmap timing, SV, judgments or teacher labels.
    A new retina cache and decoder training are required before this can be used
    for inference. The existing checkpoint must keep the legacy renderer.
    """
    def background(self):
        rgb=np.zeros((self.height,self.width,3),dtype=np.uint8);rgb[:]=[18,21,29]
        rgb[35:62]=[34,37,48]
        rgb[:,62:65]=[235,239,245]
        return rgb

    def circle(self,rgb,n,x,y):
        # The web canvas uses a 30/45px radius and 3px white stroke at 1000x300.
        # Preserve its ellipse when mapped to this 512x96 observation viewport.
        rx,ry=(24.,14.) if n.is_big else (16.,10.)
        left=max(0,int(x-rx-2));right=min(self.width,int(x+rx+3))
        top=max(0,int(y-ry-2));bottom=min(self.height,int(y+ry+3))
        if right<=left or bottom<=top:return
        patch=rgb[top:bottom,left:right]
        dx=(self.xx[top:bottom,left:right]-x)/rx
        dy=(self.yy[top:bottom,left:right]-y)/ry
        distance=dx*dx+dy*dy
        patch[distance<=1.]=[240,90,90] if n.is_don else [90,180,240]
        # A light anti-aliased edge survives reduction to retinal columns.
        patch[(distance>.87)&(distance<=1.08)]=[245,247,250]
