from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
from modelscope.hub.snapshot_download import snapshot_download


def download_model_to_local(name_model, local_dir,  revision=None):
    if revision:
        model_dir = snapshot_download(name_model, local_dir, revision=revision)
    model_dir = snapshot_download(name_model, local_dir)
    return model_dir

def t2v_pipeline(pipe_t2v, text_input, output_path, max_frames = 16):
    # pipe_t2v = pipeline(task, model_dir)
    input_test = {
        'text' : text_input,
        'max_frames' : max_frames
    }
    
    output_video = pipe_t2v(input_test, output_video=output_path)[OutputKeys.OUTPUT_VIDEO]
    return output_path

def v2v_pipeline_enhance(pipe_v2v, video_path, text_input, output_path, max_frames = 16):
    # pipe = pipeline(task, model_dir)
    input_test = {
        'video_path' : video_path,
        'text' : text_input,
        'max_frames' : max_frames
    }
    
    output_video = pipe_v2v(input_test, output_video=output_path)[OutputKeys.OUTPUT_VIDEO]
    return output_path

def main(local_dir, text_input, output_path, max_frames = 16):
    t2v_model_dir = download_model_to_local('damo/text-to-video-synthesis', local_dir)
    v2v_model_dir = download_model_to_local('damo/Video-to-Video', local_dir, 'v1.1.0')
    
    t2v_pipeline = pipeline('text-to-video-synthesis',model=  t2v_model_dir)
    v2v_pipeline = pipeline('video-to-video', model = v2v_model_dir, model_revision = 'v1.1.0', device='cuda:0')
    
    t2v_path = t2v_pipeline(t2v_pipeline, text_input, output_path, max_frames = 16)
    v2v_path = v2v_pipeline_enhance(v2v_pipeline, t2v_path, text_input, output_path, max_frames = 16)
    
    return v2v_path
    
if __name__ == '__main__':
    local_dir = './'
    text_input = ''
    output_path = ''
    result = main(local_dir, text_input, output_path, max_frames = 16)