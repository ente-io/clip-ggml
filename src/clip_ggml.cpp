#include "clip_ggml.h"
#include <string>

FFI_PLUGIN_EXPORT char * test(char * input) {
  // std::string sample = "test works";
  // char *result = new char[sample.size() + 1];
  // strcpy(result, sample.c_str());
  // return result;
  return input;
}