#include <string>
#include "clip.cpp/clip.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <cstring>
#include "json.hpp"

using json = nlohmann::json;


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

std::string arrayToArrayString(float * embedding, int length)
{
  std::string embedding_string = "[";

  for (int i = 0; i < length; i++) {
    embedding_string += std::to_string(embedding[i]) + ",";
  }
  embedding_string.pop_back();
  embedding_string += "]";
  return embedding_string;
}

// Source: https://gist.github.com/gustavorv86/c5fe4f279258ac0cfcebad88bb6acee2
// void string_split(char * string, char sep, char *** r_array_string, int * r_size) {
// 	int i, k, len, size;
// 	char ** array_string;
	
// 	// Number of substrings
// 	size = 1, len = strlen(string);
// 	for(i = 0; i < len; i++) {
// 		if(string[i] == sep) {
// 			size++;
// 		}
// 	}
	
// 	array_string = malloc(size * sizeof(char*));
	
// 	i=0, k=0;
// 	array_string[k++] = string; // Save the first substring pointer
// 	// Split 'string' into substrings with \0 character
// 	while(k < size) {
// 		if(string[i++] == sep) {
// 			string[i-1] = '\0'; // Set end of substring
// 			array_string[k++] = (string+i); // Save the next substring pointer
// 		}
// 	}
// 	*r_array_string = array_string;
// 	*r_size = size;
// 	return;
// }

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

  // char *get_score(char *image_embedding, char *text_embedding, int vec_dim)
  // {
  //   // TODO: handle errors from this function
  //   float image_embedding_array[vec_dim];
  //   float text_embedding_array[vec_dim];

  //   char **image_char_array, **text_char_array;
  //   int size;
  //   string_split(image_embedding, ',', &image_char_array, &size);
  //   string_split(text_embedding, ',', &text_char_array, &size);
  //   for (int i = 0; i < vec_dim; i++) {
  //     image_embedding_array[i] = std::stof(image_char_array[i]);
  //     text_embedding_array[i] = std::stof(text_char_array[i]);
  //   }
  //   return str_to_charp(std::to_string(clip_similarity_score(image_embedding_array, text_embedding_array, vec_dim)));
  // }

  char *test_json(char *body) {
    json jsonBody = json::parse(body);
    std::string a = jsonBody["embedding"];
    return str_to_charp(a);
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