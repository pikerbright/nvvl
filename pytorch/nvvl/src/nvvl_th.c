#include <THC/THC.h>
#include "VideoLoader.h"

extern THCState *state;

int nvvl_sequence_stream_wait_th(PictureSequenceHandle sequence) {
    return nvvl_sequence_stream_wait(sequence, THCState_getCurrentStream(state));
}
