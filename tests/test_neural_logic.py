import numpy as np
import pytest
import torch
from scipy import sparse
from flytaiko.neural_system import NeuralConfig, NeuralLIF, NeuralRuntime, JsonBeatmap
from flytaiko.neural_game import Game, Observation, PROFILES, metrics


def test_directed_synapse_propagates_from_pre_to_post_only():
    adjacency=sparse.csr_matrix(([1.],([0],[1])),shape=(2,2))
    lif=NeuralLIF(adjacency,NeuralConfig(synaptic_gain=30),device='cpu')
    lif.spikes[0,0]=1
    lif.step(torch.zeros((2,1)))
    assert lif.voltage[1,0]>-65
    assert lif.voltage[0,0]==-65


def test_lif_batch_columns_equal_independent_simulations():
    adjacency=sparse.csr_matrix(([1.,-.2],([0,1],[1,0])),shape=(2,2))
    cfg=NeuralConfig(synaptic_gain=30)
    batched=NeuralLIF(adjacency,cfg,batch=2,device='cpu')
    independent=[NeuralLIF(adjacency,cfg,device='cpu') for _ in range(2)]
    current=torch.tensor([[60.,90.],[0.,0.]])
    for _ in range(60):
        batched.step(current)
        for i,one in enumerate(independent):one.step(current[:,i:i+1])
    for i,one in enumerate(independent):
        torch.testing.assert_close(batched.voltage[:,i:i+1],one.voltage)
        torch.testing.assert_close(batched.rates[:,i:i+1],one.rates)


def make_game(types=('don','don','kat','kat','don_big','kat_big'),times=None):
    return Game(JsonBeatmap({'metadata':{'title':'test'},'notes':[
        {'t':times[i] if times else 320+i*27,'type':kind} for i,kind in enumerate(types)]}))


def test_expert_reads_do_not_change_alternation_and_fast_notes_are_hittable():
    game=make_game();actions=[]
    while not game.is_done:
        label=game.expert()
        assert label==game.expert()
        if label:actions.append(label)
        game.hit(label);game.advance(game.step_ms)
    assert actions==[1,2,3,4,5,6]
    assert metrics([game])['miss']==0
    assert metrics([game])['p95_abs_error_ms']<=4


def test_big_note_single_hit_keeps_judgment_but_no_full_hit_bonus():
    game=make_game(('don_big','kat'),[0,100])
    game.hit(1);game.advance(500)
    result=metrics([game])
    assert result['great']==1 and result['miss']==1 and result['total_notes']==2
    assert result['accuracy_by_type']['don_big']==1
    assert result['accuracy_by_type']['kat']==0
    assert result['big_note_full_hit_rate']==0


def test_piecewise_hit_windows_match_official_taiko_ranges():
    from flytaiko.neural_game import taiko_windows
    assert taiko_windows(7)=={'great':28.5,'good':67.5,'miss':84.5}
    assert taiko_windows(10)=={'great':19.5,'good':49.5,'miss':69.5}


def test_future_hidden_notes_cannot_be_revealed_by_noise():
    runtime=NeuralRuntime(NeuralConfig(neurons=1000,visible_ms=1200),device='cpu')
    game=make_game(('don',),[5000])
    obs=Observation(dict(PROFILES['combined'],noise=10000),np.random.default_rng(3))
    assert obs.read(runtime,game)['upcoming_notes'][0][1]=='none'


def test_same_color_hand_repeats_count_even_for_missed_actions():
    game=make_game(('don','don'),[0,100])
    game.hit(1);game.advance(100);game.hit(1)
    assert metrics([game])['full_alt_violations']==1


def test_synaptic_readout_changes_with_stimulus_and_disappears_without_edges():
    cfg=NeuralConfig(neurons=1000)
    runtime=NeuralRuntime(cfg,batch=2,device='cpu')
    states=[{'upcoming_notes':[(4,'don')]+[(99999,'none')]*3,'don_alt':False,'kat_alt':False},
            {'upcoming_notes':[(4,'kat')]+[(99999,'none')]*3,'don_alt':True,'kat_alt':True}]
    for _ in range(20):features=runtime.features(states)
    assert features.isfinite().all()
    assert torch.mean(abs(features[0]-features[1]))>.001
    runtime.lif.W=torch.sparse_coo_tensor(runtime.lif.W.indices(),torch.zeros_like(runtime.lif.W.values()),
                                        runtime.lif.W.shape).coalesce()
    runtime.lif.reset()
    for _ in range(20):disconnected=runtime.features(states)
    torch.testing.assert_close(disconnected[0],disconnected[1])


def test_t_zero_note_has_a_first_frame_decision():
    game=make_game(('don',),[0])
    game.advance(8)
    assert game.expert()==1
    game.hit(1)
    assert metrics([game])['great']==1
