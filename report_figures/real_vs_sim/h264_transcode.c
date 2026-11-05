// Re-encode one clip to browser-playable H.264 MP4 (the recorded clips are MPEG-4 Part 2, which browsers do not play).
// Build: gcc -O2 h264_transcode.c -o h264_transcode $(pkg-config --cflags --libs libavformat libavcodec libavutil libswscale)
// Usage: h264_transcode IN OUT WIDTH CRF
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <stdio.h>
#include <stdlib.h>
#define CK(x) do { int _r = (x); if (_r < 0) { fprintf(stderr, "failed: %s (%d)\n", #x, _r); return 1; } } while (0)
static int flush_out(AVCodecContext *e, AVFormatContext *o, AVStream *s) {
    AVPacket *p = av_packet_alloc(); int r;
    while ((r = avcodec_receive_packet(e, p)) == 0) {
        av_packet_rescale_ts(p, e->time_base, s->time_base); p->stream_index = s->index;
        if (av_interleaved_write_frame(o, p) < 0) return -1;
    }
    av_packet_free(&p); return 0;
}
int main(int c, char **v) {
    if (c != 5) { fprintf(stderr, "usage: IN OUT WIDTH CRF\n"); return 2; }
    int W = atoi(v[3]);
    AVFormatContext *in = NULL, *out = NULL;
    CK(avformat_open_input(&in, v[1], NULL, NULL)); CK(avformat_find_stream_info(in, NULL));
    int si = av_find_best_stream(in, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0); CK(si);
    AVStream *is = in->streams[si];
    AVCodec *dc = avcodec_find_decoder(is->codecpar->codec_id);
    AVCodecContext *d = avcodec_alloc_context3(dc);
    CK(avcodec_parameters_to_context(d, is->codecpar)); CK(avcodec_open2(d, dc, NULL));
    int H = (d->height * W / d->width) & ~1; W &= ~1;
    AVCodec *ec = avcodec_find_encoder_by_name("libx264"); if (!ec) { fprintf(stderr, "no libx264\n"); return 1; }
    AVCodecContext *e = avcodec_alloc_context3(ec);
    e->width = W; e->height = H; e->pix_fmt = AV_PIX_FMT_YUV420P; e->time_base = (AVRational){1, 30}; e->framerate = (AVRational){30, 1};
    e->gop_size = 15; e->max_b_frames = 2; e->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    av_opt_set(e->priv_data, "preset", "slow", 0); av_opt_set(e->priv_data, "crf", v[4], 0); av_opt_set(e->priv_data, "profile", "main", 0);
    CK(avcodec_open2(e, ec, NULL));
    CK(avformat_alloc_output_context2(&out, NULL, NULL, v[2]));
    AVStream *os = avformat_new_stream(out, NULL); CK(avcodec_parameters_from_context(os->codecpar, e)); os->time_base = e->time_base;
    CK(avio_open(&out->pb, v[2], AVIO_FLAG_WRITE));
    AVDictionary *mo = NULL; av_dict_set(&mo, "movflags", "+faststart", 0); CK(avformat_write_header(out, &mo));
    struct SwsContext *sw = sws_getContext(d->width, d->height, d->pix_fmt, W, H, AV_PIX_FMT_YUV420P, SWS_AREA, NULL, NULL, NULL);
    AVFrame *f = av_frame_alloc(), *g = av_frame_alloc(); g->format = AV_PIX_FMT_YUV420P; g->width = W; g->height = H; CK(av_frame_get_buffer(g, 32));
    AVPacket *pk = av_packet_alloc(); int64_t n = 0; int eof = 0;
    while (!eof) {
        int r = av_read_frame(in, pk);
        if (r < 0) { eof = 1; avcodec_send_packet(d, NULL); }
        else if (pk->stream_index != si) { av_packet_unref(pk); continue; }
        else { avcodec_send_packet(d, pk); av_packet_unref(pk); }
        while (avcodec_receive_frame(d, f) == 0) {
            CK(av_frame_make_writable(g));
            sws_scale(sw, (const uint8_t * const *)f->data, f->linesize, 0, d->height, g->data, g->linesize);
            g->pts = n++;  // frame index at 30 fps: the same timeline the sync used
            CK(avcodec_send_frame(e, g)); CK(flush_out(e, out, os));
        }
    }
    avcodec_send_frame(e, NULL); CK(flush_out(e, out, os));
    CK(av_write_trailer(out)); avio_closep(&out->pb);
    printf("%lld\n", (long long)n);
    return 0;
}
