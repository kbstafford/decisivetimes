from one.api import ONE

one = ONE(base_url='https://openalyx.internationalbrainlab.org')
eid, *_ = one.search(project='brainwide', datasets='spikes.times.npy')
one.load_collection(eid, 'alf', download_only=True)


