#include <string>
#include "clip.cpp/clip.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <cstring>
#include "json.hpp"

using json = nlohmann::json;

struct clip_image_u8_batch make_clip_image_u8_batch(std::vector<clip_image_u8> & images) {
    struct clip_image_u8_batch batch;
    batch.data = images.data();
    batch.size = images.size();
    return batch;
}

// Constructor-like function
struct clip_image_f32_batch make_clip_image_f32_batch(std::vector<clip_image_f32> & images) {
    struct clip_image_f32_batch batch;
    batch.data = images.data();
    batch.size = images.size();
    return batch;
}

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

char *jsonToChar(json jsonData)
{
    std::string result = jsonData.dump();
    char *ch = new char[result.size() + 1];
    strcpy(ch, result.c_str());
    return ch;
}

std::string arrayToArrayString(float * embedding, int length, int start = 0)
{
  std::string embedding_string = "[";

  for (int i = start; i < start + length; i++) {
    embedding_string += std::to_string(embedding[i]) + ",";
  }
  embedding_string.pop_back();
  embedding_string += "]";
  return embedding_string;
}

extern "C"
{
  struct clip_ctx *ctx;
  float cached_img_vec[512];

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

  char *create_image_embedding(char *dart_image_path)
  {
    json result;
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

    // Creating result JSON
    result["vec_dim"] = std::to_string(vec_dim);
    result["embedding"] = arrayToArrayString(img_vec, vec_dim);

    for (int i = 0; i < sizeof(img_vec) / sizeof(img_vec[0]); i++)
    {
      cached_img_vec[i] = img_vec[i];
    }

    return jsonToChar(result);
  }

  char *batch_image_embeddings(char *body)
  {

    if (!ctx)
    {
      std::string error_message = "Model not loaded";
      return str_to_charp(error_message);
    }

    const size_t n_threads = 4;
    json jsonBody = json::parse(body);
    int batch_size = jsonBody["batch_size"];
    int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;

    std::vector<clip_image_u8> img_inputs(batch_size);
    std::vector<clip_image_f32> imgs_resized(batch_size);
    float img_vecs[vec_dim * batch_size];

    for (int i = 0; i < batch_size; i++) {
      char *image_path = str_to_charp(jsonBody["image_paths"][i]);
      if (!clip_image_load_from_file(image_path, &img_inputs[i])) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, image_path);
        return image_load_failure;
      }
    }

    clip_image_u8_batch img_inputs_batch = make_clip_image_u8_batch(img_inputs);
    clip_image_f32_batch imgs_resized_batch = make_clip_image_f32_batch(imgs_resized);

    
    clip_image_batch_preprocess(ctx, n_threads, &img_inputs_batch, &imgs_resized_batch);
    clip_image_batch_encode(ctx, n_threads, &imgs_resized_batch, img_vecs, true);

    json result;
    result["vec_dim"] = std::to_string(vec_dim);
    for (int i = 0; i < batch_size; i++) {
      result["embedding_" + std::to_string(i)] = arrayToArrayString(img_vecs, vec_dim, vec_dim * i);
    }
    return jsonToChar(result);
  }

  char *create_text_embedding(char *dart_text)
  {
    json result;
    if (!ctx)
    {
      std::string error_message = "Model not loaded";
      return str_to_charp(error_message);
    }

    int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;
    struct clip_tokens tokens = clip_tokenize(ctx, dart_text);

    float txt_vec[vec_dim];
    if (!clip_text_encode(ctx, 4, &tokens, txt_vec, true))
    {
      fprintf(stderr, "%s: failed to encode text\n", __func__);
      return text_encode_failure;
    }

    // Creating result JSON
    result["vec_dim"] = std::to_string(vec_dim);
    result["embedding"] = arrayToArrayString(txt_vec, vec_dim);

    return jsonToChar(result);
  }

  char *get_score(char *image_embedding, char *text_embedding, int vec_dim)
  {
    // TODO: handle errors from this function
    float image_embedding_array[vec_dim];
    float text_embedding_array[vec_dim];

    // TODO: Assert length of image_embedding and text_embedding to be same

    char image_array[std::strlen(image_embedding) + 1];
    char text_array[std::strlen(text_embedding) + 1];

    for (int i = 0; i < std::strlen(image_embedding) + 1; i++) {
      image_array[i] = image_embedding[i];
      text_array[i] = text_embedding[i];
    }
    char *image_embedding_char_array = strtok(image_array, "[,]");
    char *text_embedding_char_array = strtok(text_array, "[,]");

    int i = 0;
    while (image_embedding_char_array) {
      image_embedding_array[i++] = std::stof(image_embedding_char_array);
      image_embedding_char_array = strtok(NULL, "[,]");
    }

    i = 0;
    while (text_embedding_char_array) {
      text_embedding_array[i++] = std::stof(text_embedding_char_array);
      text_embedding_char_array = strtok(NULL, "[,]");
    }
    return str_to_charp(std::to_string(clip_similarity_score(image_embedding_array, text_embedding_array, vec_dim)));
  }

  char *run_inference(char *dart_text)
  {
    if (!ctx)
    {
      std::string error_message = "Model not loaded";
      return str_to_charp(error_message);
    }

    int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;
    struct clip_tokens tokens = clip_tokenize(ctx, dart_text);

    float txt_vec[vec_dim];
    if (!clip_text_encode(ctx, 4, &tokens, txt_vec, true))
    {
      fprintf(stderr, "%s: failed to encode text\n", __func__);
      return text_encode_failure;
    }

    float score = clip_similarity_score(cached_img_vec, txt_vec, vec_dim);
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