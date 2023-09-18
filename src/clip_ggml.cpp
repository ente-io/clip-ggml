#include <string>
#include "clip.cpp/clip.h"

extern "C"
{
  char * test(char * input) {
    // std::string sample = "test works";
    // char *result = new char[sample.size() + 1];
    // strcpy(result, sample.c_str());
    // return result;
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();
    std:string time = to_string(t_main_start_us);
    char *result = new char[sample.size() + 1];
    strcpy(result, sample.c_str());
    return result;
  }


}
