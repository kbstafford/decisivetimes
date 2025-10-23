# Save this as ibl_connect.py and run it with: python ibl_connect.py
from one.api import ONE
import ibllib.io.video as vidio

one = ONE()
eid = '4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a'
label = 'body' # 'left', 'right' or 'body'

# Find url of video data to stream
url = vidio.url_from_eid(eid, one=one)[label]

# Load video timestamps
ts = one.load_dataset(eid, f'*{label}Camera.times*', collection='alf')

# Find the frame closest to 1000s into data
import numpy as np
frame_n = np.searchsorted(ts, 1000)

# Stream the data
frame = vidio.get_video_frame(url, frame_n)
