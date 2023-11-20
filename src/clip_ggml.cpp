#include <string>
#include "clip.cpp/clip.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <cstring>


char *str_to_charp(std::string s)
{
  char *result = new char[s.size() + 1];
  strcpy(result, s.c_str());
  return result;
}

extern "C"
{
  struct clip_ctx *img_ctx;

  char *load_image_model(char *image_model_path)
  {
    img_ctx = clip_model_load(image_model_path, 1);
    if (!img_ctx)
    {
      std::string error_message = "Image model not loaded";
      return str_to_charp(error_message);
    }
    return str_to_charp("ok");
  }
}