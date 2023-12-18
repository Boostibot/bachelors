#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code

//
//      cudaPrintfInit
//
//      Call this once to initialise the printf system. If the output
//      file or buffer size needs to be changed, call cudaPrintfEnd()
//      before re-calling cudaPrintfInit().
//
//      The default size for the buffer is 1 megabyte. For CUDA
//      architecture 1.1 and above, the buffer is filled linearly and
//      is completely used;     however for architecture 1.0, the buffer
//      is divided into as many segments are there are threads, even
//      if some threads do not call cuPrintf().
//
//      Arguments:
//              bufferLen - Length, in bytes, of total space to reserve
//                          (in device global memory) for output.
//
//      Returns:
//              cudaSuccess if all is well.
//
extern "C" cudaError_t cudaPrintfInit(size_t bufferLen_or_zero);   // 1-meg - that's enough for 4096 printfs by all threads put together

//
//      cudaPrintfEnd
//
//      Cleans up all memories allocated by cudaPrintfInit().
//      Call this at exit, or before calling cudaPrintfInit() again.
//
extern "C" void cudaPrintfEnd();

//
//      cudaPrintfDisplay
//
//      Dumps the contents of the output buffer to the specified
//      file pointer. If the output pointer is not specified,
//      the default "stdout" is used.
//
//      Arguments:
//              outputFP     - A file pointer to an output stream.
//              showThreadID - If "true", output strings are prefixed
//                             by "[blockid, threadid] " at output.
//
//      Returns:
//              cudaSuccess if all is well.
//
extern "C" cudaError_t cudaPrintfDisplay(void *outputFP_or_zero, bool showThreadID);