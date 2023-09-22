import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart';

typedef load_model_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> model_path);

typedef create_image_embedding_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> image_path);

typedef create_text_embedding_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> text);

typedef inference_request = ffi.Pointer<Utf8> Function(ffi.Pointer<Utf8> text);

typedef get_score_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> image_embedding,
    ffi.Pointer<Utf8> text_embedding,
    ffi.Int32 vec_dim);

typedef get_score_response = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> image_embedding,
    ffi.Pointer<Utf8> text_embedding,
    int vec_dim);

class CLIP {
  final String clipLib = "libclip_ggml.so";

  ffi.DynamicLibrary openLib() {
    if (Platform.isIOS || Platform.isMacOS) {
      return ffi.DynamicLibrary.process();
    } else {
      return ffi.DynamicLibrary.open(clipLib);
    }
  }

  String loadModel(String modelPath) {
    var res = openLib()
        .lookupFunction<load_model_request, load_model_request>("load_model")
        .call(modelPath.toNativeUtf8());
    return res.toDartString();
  }

  List<double> createImageEmbedding(String imagePath) {
    final res = openLib()
        .lookupFunction<create_image_embedding_request,
            create_image_embedding_request>("create_image_embedding")
        .call(imagePath.toNativeUtf8());
    return List<double>.from(
        jsonDecode(jsonDecode(res.toDartString())["embedding"]) as List);
  }

  List<double> createTextEmbedding(String text) {
    final res = openLib()
        .lookupFunction<create_text_embedding_request,
            create_text_embedding_request>("create_text_embedding")
        .call(text.toNativeUtf8());
    return List<double>.from(
        jsonDecode(jsonDecode(res.toDartString())["embedding"]) as List);
  }

  String runInference(String text) {
    var res = openLib()
        .lookupFunction<inference_request, inference_request>("run_inference")
        .call(text.toNativeUtf8());
    return res.toDartString();
  }

  double computeScore(List<double> imageEmbedding, List<double> textEmbedding) {
    assert(imageEmbedding.length == textEmbedding.length,
        "The two embeddings should have the same length");
    double score = 0;
    for (int index = 0; index < imageEmbedding.length; index++) {
      score += imageEmbedding[index] * textEmbedding[index];
    }
    return score;
  }
}
