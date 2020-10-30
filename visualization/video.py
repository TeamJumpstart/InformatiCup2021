from imageio_ffmpeg import write_frames


def render_video(file_name, frames, width, height, fps=20):
    # Initilize ffmpeg writer
    writer = write_frames(file_name, (width, height), fps=fps, codec="libx264", quality=8)
    writer.send(None)  # seed the generator

    for frame in frames:
        writer.send(frame)

    writer.close()
