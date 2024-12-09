import os
import argparse
import json
from time import perf_counter
from datetime import datetime
from model.pred_func import *
from model.config import load_config
from pytubefix import YouTube
import io
import tempfile

config = load_config()
print('CONFIG')
print(config)

def single_vid(
    ed_weight, vae_weight, path, net=None, fp16=False
):
    result = set_result()
    r = 0
    f = 0
    count = 0
    
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    video = YouTube(path)
    video_stream = video.streams.filter(res="360p", file_extension="mp4").first()
    buffer = io.BytesIO()
    video_stream.stream_to_buffer(buffer)
    buffer.seek(0)

    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
        temp_file.write(buffer.read())
        temp_file.flush()

        try:
            if is_video(temp_file.name):
                result, accuracy, count, pred = predict(
                    temp_file.name,
                    model,
                    fp16,
                    result,
                    "uncategorized",
                    count,
                )
                f, r = (f + 1, r) if "FAKE" == real_or_fake(pred[0]) else (f, r + 1)
                print(
                    f"Prediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}"
                )
            else:
                print(f"Invalid video file: {path}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return pred[1]


def predict(
    vid,
    model,
    fp16,
    result,
    klass,
    count=0,
    accuracy=-1,
    correct_label="unknown",
    compression=None,
):
    count += 1
    print(f"\n\n{str(count)} Loading... {vid}")

    df = df_face(vid)  # extract face from the frames
    if fp16:
        df.half()
    y, y_val = (
        pred_vid(df, model)
        if len(df) >= 1
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )
    result = store_result(
        result, os.path.basename(vid), y, y_val, klass, correct_label, compression
    )

    if accuracy > -1:
        if correct_label == real_or_fake(y):
            accuracy += 1
        print(
            f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}"
        )

    return result, accuracy, count, [y, y_val]


def gen_parser():
    parser = argparse.ArgumentParser("GenConViT prediction")
    parser.add_argument("--p", type=str, help="video or image path")
    parser.add_argument(
        "--f", type=int, help="number of frames to process for prediction"
    )
    parser.add_argument(
        "--d", type=str, help="dataset type, dfdc, faceforensics, timit, celeb"
    )
    parser.add_argument(
        "--s", help="model size type: tiny, large.",
    )
    parser.add_argument(
        "--e", nargs='?', const='genconvit_ed_inference', default='genconvit_ed_inference', help="weight for ed.",
    )
    parser.add_argument(
        "--v", '--value', nargs='?', const='genconvit_vae_inference', default='genconvit_vae_inference', help="weight for vae.",
    )
    
    parser.add_argument("--fp16", type=str, help="half precision support")

    args = parser.parse_args()
    path = args.p
    num_frames = args.f if args.f else 15
    dataset = args.d if args.d else "other"
    fp16 = True if args.fp16 else False

    net = 'genconvit'
    ed_weight = 'genconvit_ed_inference'
    vae_weight = 'genconvit_vae_inference'

    if args.e and args.v:
        ed_weight = args.e
        vae_weight = args.v
    elif args.e:
        net = 'ed'
        ed_weight = args.e
    elif args.v:
        net = 'vae'
        vae_weight = args.v
    
        
    print(f'\nUsing {net}\n')  
    

    if args.s:
        if args.s in ['tiny', 'large']:
            config["model"]["backbone"] = f"convnext_{args.s}"
            config["model"]["embedder"] = f"swin_{args.s}_patch4_window7_224"
            config["model"]["type"] = args.s
    
    return path, dataset, num_frames, net, fp16, ed_weight, vae_weight


def main():
    path, dataset, num_frames, net, fp16, ed_weight, vae_weight = gen_parser()

    fake_prob = single_vid(ed_weight, vae_weight, path, net, fp16)

    print(f'fake prob is: {fake_prob}')

if __name__ == "__main__":
    main()
