#include <string>
#include "clip.cpp/clip.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <cstring>

char *model = "models/openai_clip-vit-base-patch32.ggmlv0.f16.bin";
char *image_path = "models/red_apple.jpg";
char *text = "an apple";
char *image_load_failure = "Failed to load the image";
char *image_preprocess_failure = "Failed to preprocess the image";
char *image_encode_failure = "Failed to encode the image";
char *text_encode_failure = "Failed to encode the text";

char *str_to_charp(std::string s)
{
  char *result = new char[s.size() + 1];
  strcpy(result, s.c_str());
  return result;
}

extern "C"
{
  struct clip_ctx *ctx;

  char *load_model(char *model_path)
  {
    ctx = clip_model_load(model_path, 1);
    if (!ctx)
    {
      std::string error_message = "Model not loaded";
      return str_to_charp(error_message);
    }
    return str_to_charp("ok");
  }

  char *run_inference(char *dart_image_path, char *dart_text)
  {
    if (!ctx)
    {
      std::string error_message = "Model not loaded";
      return str_to_charp(error_message);
    }

    int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;
    struct clip_image_u8 *img0 = make_clip_image_u8();
    if (!clip_image_load_from_file(dart_image_path, img0))
    {
      fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, image_path);
      return image_load_failure;
    }

    struct clip_image_f32 *img_res = make_clip_image_f32();
    if (!clip_image_preprocess(ctx, img0, img_res))
    {
      fprintf(stderr, "%s: failed to preprocess image\n", __func__);
      return image_preprocess_failure;
    }

    float img_vec[vec_dim];
    if (!clip_image_encode(ctx, 4, img_res, img_vec, true))
    {
      fprintf(stderr, "%s: failed to encode image\n", __func__);
      return image_encode_failure;
    }

    struct clip_tokens tokens = clip_tokenize(ctx, dart_text);

    float txt_vec[vec_dim];
    if (!clip_text_encode(ctx, 4, &tokens, txt_vec, true))
    {
      fprintf(stderr, "%s: failed to encode text\n", __func__);
      return text_encode_failure;
    }

    float score = clip_similarity_score(img_vec, txt_vec, vec_dim);
    printf("Similarity score = %2.3f\n", score);
    return str_to_charp(std::to_string(score));
  }

  char *can_read_file(const char *file_path)
  {
    std::ifstream file(file_path);
    if (file.is_open())
    {
      file.close();
      return str_to_charp("true");
    }
    std::cerr << "Error: " << std::strerror(errno) << std::endl;
    return str_to_charp("false");
  }
}

int main()
{
  return 1;
}