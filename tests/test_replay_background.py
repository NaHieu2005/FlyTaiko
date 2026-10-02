from flytaiko.replay_osz_worker import background_member


def test_background_member():
    text = '[Events]\n0,0,"art/bg.jpg",0,0\n'
    assert background_member(text, 'song/map.osu', ['song/art/bg.jpg']) == 'song/art/bg.jpg'
    assert background_member('0,0,"../private.png",0,0', 'map.osu', ['../private.png']) is None
    assert background_member('0,0,"video.mp4",0,0', 'map.osu', ['video.mp4']) is None
