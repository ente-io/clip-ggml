#include <string>
#include "clip.cpp/clip.h"
#include <vector>


std::string model = "models/openai_clip-vit-base-patch32.ggmlv0.f16.bin";
std::string image_path;
std::string text;

char *str_to_charp(std::string s) {
  char *result = new char[s.size() + 1];
  strcpy(result, s.c_str());
  return result;
}

extern "C"
{
  char * test(char * input) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();
    char *result = str_to_charp(std::to_string(t_main_start_us));
    
    auto ctx = clip_model_load(model.c_str(), 1);
    if (!ctx){
      std::string error_message = "Model not loaded";
      return str_to_charp(error_message);
    }
    return result;
  }

  


}
