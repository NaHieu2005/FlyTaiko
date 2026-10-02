from flytaiko.replay_osz_worker import background_member, kiai_intervals


def test_background_member():
    text = '[Events]\n0,0,"art/bg.jpg",0,0\n'
    assert background_member(text, 'song/map.osu', ['song/art/bg.jpg']) == 'song/art/bg.jpg'
    assert background_member('0,0,"../private.png",0,0', 'map.osu', ['../private.png']) is None
    assert background_member('0,0,"video.mp4",0,0', 'map.osu', ['video.mp4']) is None


def test_kiai_intervals():
    assert kiai_intervals('[TimingPoints]\n0,500,4,1,0,100,1,0\n1000,-100,4,1,0,100,0,1\n2000,-100,4,1,0,100,0,0\n3000,500,4,1,0,100,1,1', 4000) == [[1000,2000],[3000,4000]]
