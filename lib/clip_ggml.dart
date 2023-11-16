import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart';

typedef load_model_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> model_path);

typedef create_image_embedding_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> image_path);

typedef preprocess_image_embedding_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> image_path);

typedef create_batch_image_embedding_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> request);

typedef create_text_embedding_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> text);

typedef get_score_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> image_embedding,
    ffi.Pointer<Utf8> text_embedding,
    ffi.Int32 vec_dim);

typedef get_score_response = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> image_embedding,
    ffi.Pointer<Utf8> text_embedding,
    int vec_dim);

class CLIP {
  static const clipLib = "libclip_ggml.so";

  static final _clip = openLib();

  static ffi.DynamicLibrary openLib() {
    if (Platform.isIOS || Platform.isMacOS) {
      return ffi.DynamicLibrary.process();
    } else {
      return ffi.DynamicLibrary.open(clipLib);
    }
  }

  static loadImageModel(String modelPath) {
    var res = _clip
        .lookupFunction<load_model_request, load_model_request>(
            "load_image_model")
        .call(modelPath.toNativeUtf8());
    return res.toDartString();
  }

  static loadTextModel(String modelPath) {
    var res = _clip
        .lookupFunction<load_model_request, load_model_request>(
            "load_text_model")
        .call(modelPath.toNativeUtf8());
    return res.toDartString();
  }

  static List<double> createImageEmbedding(String imagePath) {
    final res = _clip
        .lookupFunction<create_image_embedding_request,
            create_image_embedding_request>("create_image_embedding")
        .call(imagePath.toNativeUtf8());
    return List<double>.from(
        jsonDecode(jsonDecode(res.toDartString())["embedding"]) as List);
  }

  static String preprocessImage(String imagePath) {
    final res = _clip
        .lookupFunction<preprocess_image_embedding_request,
            preprocess_image_embedding_request>("preprocess_image")
        .call(imagePath.toNativeUtf8());
    return res.toDartString();
  }

  static List<List<double>> createBatchImageEmbedding(List<String> imagePaths) {
    final args = <String, dynamic>{};
    args["batch_size"] = imagePaths.length;
    args["image_paths"] = imagePaths;
    final res = _clip
        .lookupFunction<create_batch_image_embedding_request,
            create_batch_image_embedding_request>("batch_image_embeddings")
        .call(jsonEncode(args).toNativeUtf8());
    final result = jsonDecode(res.toDartString());
    final List<List<double>> embeddings = [];
    for (int i = 0; i < imagePaths.length; i++) {
      embeddings
          .add(List<double>.from(jsonDecode(result[i.toString()]) as List));
    }
    return embeddings;
  }

  static List<double> createTextEmbedding(String text) {
    final res = _clip
        .lookupFunction<create_text_embedding_request,
            create_text_embedding_request>("create_text_embedding")
        .call(text.toNativeUtf8());
    return List<double>.from(
        jsonDecode(jsonDecode(res.toDartString())["embedding"]) as List);
  }

  static double computeScore(
      List<double> imageEmbedding, List<double> textEmbedding) {
    assert(imageEmbedding.length == textEmbedding.length,
        "The two embeddings should have the same length");
    double score = 0;
    for (int index = 0; index < imageEmbedding.length; index++) {
      score += imageEmbedding[index] * textEmbedding[index];
    }
    return score;
  }
}
