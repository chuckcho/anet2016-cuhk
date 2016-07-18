from config import ANET_CFG
import sys
sys.path.append(ANET_CFG.DENSE_FLOW_ROOT+'/build')
from libpydenseflow import TVL1FlowExtractor
import numpy as np
import glob
import os

class FlowExtractor(object):
    def __init__(self, dev_id, bound=20):
        TVL1FlowExtractor.set_device(dev_id)
        self._et = TVL1FlowExtractor(bound)
    def extract_flow(self, frame_list, new_size=None):
        """
        This function extracts the optical flow and interleave x and y channels
        :param frame_list:
        :return:
        """
        frame_size = frame_list[0].shape[:2]
        rst = self._et.extract_flow([x.tostring() for x in frame_list], frame_size[1], frame_size[0])
        n_out = len(rst)
        if new_size is None:
            ret = np.zeros((n_out*2, frame_size[0], frame_size[1]))
            for i in xrange(n_out):
                ret[2*i, :] = np.fromstring(rst[i][0], dtype='uint8').reshape(frame_size)
                ret[2*i+1, :] = np.fromstring(rst[i][1], dtype='uint8').reshape(frame_size)
        else:
            import cv2
            ret = np.zeros((n_out*2, new_size[1], new_size[0]))
            for i in xrange(n_out):
                ret[2*i, :] = cv2.resize(np.fromstring(rst[i][0], dtype='uint8').reshape(frame_size), new_size)
                ret[2*i+1, :] = cv2.resize(np.fromstring(rst[i][1], dtype='uint8').reshape(frame_size), new_size)
        return ret


if __name__ == "__main__":

    import cv2
    dextro_video_basedir = '/media/6TB2/Videos/dextro-benchmark'

    # 5 optical flow images (X/Y) -> 10 images per reference image
    length = 6

    # stride (default: sampling ~every second given 30fps)
    interval = 30

    # resize frame_size
    new_size = (340,256)

    f = FlowExtractor(0)

    all_video_iter = glob.glob(os.path.join(dextro_video_basedir, '*.*'))
    num_videos = len(all_video_iter)
    for count, video_file in enumerate(all_video_iter):
        print "[Info] Processing video_file={} ({}/{})".format(
                video_file,
                count,
                num_videos,
                )
        video_dir, ext = os.path.splitext(video_file)
        video_id = os.path.basename(video_dir)
        if not os.path.isdir(video_dir):
            print "[Warning] video_dir={} doesn't exist. Skipping...".format(
                    video_dir)
            continue
        frames = glob.glob(os.path.join(video_dir, '*.jpg'))
        num_frames = len(frames)
        if not num_frames:
            print("[Warning] video_dir={} doesn't contain any extracted frames."
                  "Skipping...".format(video_dir))
            continue
        print "[Info] num_frames={}".format(num_frames)

        num_iters = (num_frames - length) / 30 + 1
        if not num_iters:
            print("[Warning] video_dir={} contains only {} extracted "
                  "frames, not enough. Skipping...".format(
                          video_dir,
                          num_frames
                          )
                  )
            continue
        for iter_id in range(num_iters):
            start_frame = iter_id * interval
            end_frame = iter_id * interval + length - 1
            print("[Info] Processing iter_id={} (frames {}~{})".format(
                    iter_id,
                    start_frame,
                    end_frame,
                    )
                  )

            frm_stack = []
            for frame_num in range(start_frame, end_frame + 1):
                img_filename = os.path.join(
                        video_dir, '{0}.{1:06d}.jpg'.format(
                                video_id,
                                frame_num + 1 # 1-based image names
                                )
                        )
                #print "[Debug] Reading img={}".format(img_filename)
                frm = cv2.imread(img_filename)
                if new_size is not None:
                    frm = cv2.resize(frm, new_size)
                frm_stack.append(frm)
            # extract flow
            flow_stack = f.extract_flow(frm_stack)
            if flow_stack.shape[0] == (length - 1) * 2:
                print "[Info] Optical flow images extracted successfully."
                # save flow images
                # reversing: ret = np.zeros((n_out*2, frame_size[0], frame_size[1]))
                for frm_count in range(flow_stack.shape[0]):
                    if frm_count % 2 == 0:
                        flow_direction = 'x'
                    else:
                        flow_direction = 'y'
                    frame_pair_name = "{0:06d}_{1:06d}".format(
                            start_frame,
                            start_frame + frm_count/2 + 1
                            )

                    optical_flow_dirname = os.path.join(
                            video_dir.rstrip('/') + '_flow'
                            )
                    optical_flow_out_filename = os.path.join(
                            optical_flow_dirname,
                            '{}.{}_{}.jpg'.format(
                                    video_id, frame_pair_name, flow_direction
                                    )
                            )

                    if os.path.exists(optical_flow_out_filename) and \
                       os.stat(optical_flow_out_filename).st_size:
                        print("[Info] This output={} has already been saved. "
                              "Skipping...")
                        continue

                    # mkdir if needed
                    if not os.path.exists(optical_flow_dirname):
                        os.makedirs(optical_flow_dirname)

                    print "[Debug] Saving output={}...".format(optical_flow_out_filename)
                    cv2.imwrite(
                            optical_flow_out_filename, flow_stack[frm_count]
                            )

    print "-" * 79
    print "[Info] All done!"
