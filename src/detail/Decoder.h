#pragma once

#include <string>

#include "PictureSequenceImpl.h"

class AVPacket;
#ifdef HAVE_AVSTREAM_CODECPAR
class AVCodecParameters;
using CodecParameters = AVCodecParameters;
#else
class AVCodecContext;
using CodecParameters = AVCodecContext;
#endif

namespace NVVL {

class FrameReq;
class PictureSequence;

namespace detail {

class Logger;

struct FrameReq {
    std::string filename;
    int frame;
    int count;
    bool stream;
};

class CUStream {
  public:
    CUStream(bool default_stream);
    ~CUStream();
    CUStream(const CUStream&) = delete;
    CUStream& operator=(const CUStream&) = delete;
    CUStream(CUStream&&);
    CUStream& operator=(CUStream&&);
    void create(bool default_stream);
    operator cudaStream_t();

public:
    bool created_;
    cudaStream_t stream_;
};

class Decoder {
  public:
    Decoder();
    Decoder(int device_id, Logger& logger,
            const CodecParameters* codecpar);
    Decoder(const Decoder&) = default;
    Decoder(Decoder&&) = default;
    Decoder& operator=(const Decoder&) = default;
    Decoder& operator=(Decoder&&) = default;
    virtual ~Decoder() = default;

    int decode_packet(AVPacket* pkt);

    virtual void reset(const CodecParameters *codecpar){}

    virtual void reset_flag(){}

    virtual void push_req(FrameReq req);

    virtual void receive_frames(PictureSequence& sequence);

    virtual void finish();

    virtual void set_time_base(AVRational time_base);

    virtual void set_frame_base(AVRational frame_base);

    virtual void set_max_send_frame(int max_num);

  protected:
    virtual int decode_av_packet(AVPacket* pkt);

    void record_sequence_event_(PictureSequence& sequence);
    void record_sequence_end_event_(PictureSequence& sequence);
    void use_default_stream();

    // We're buddies with PictureSequence so we can forward a visitor
    // on to it's private implementation.
    template<typename Visitor>
    void foreach_layer(PictureSequence& sequence, const Visitor& visitor) {
        sequence.pImpl->foreach_layer(visitor);
    }

    const int device_id_;
    const CodecParameters* codecpar_;
    AVMediaType codec_type_;

    detail::Logger& log_;
public:
    CUStream stream_;
};


}
}
