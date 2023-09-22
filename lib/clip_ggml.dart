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
    ffi.Pointer<Utf8> model_string = modelPath.toNativeUtf8();
    var res = openLib()
        .lookupFunction<load_model_request, load_model_request>("load_model")
        .call(model_string);
    return res.toDartString();
  }

  String createImageEmbedding(String imagePath) {
    ffi.Pointer<Utf8> image_string = imagePath.toNativeUtf8();
    var res = openLib()
        .lookupFunction<create_image_embedding_request,
            create_image_embedding_request>("create_image_embedding")
        .call(image_string);
    return res.toDartString();
  }

  String createTextEmbedding(String text) {
    ffi.Pointer<Utf8> text_string = text.toNativeUtf8();
    var res = openLib()
        .lookupFunction<create_text_embedding_request,
            create_text_embedding_request>("create_text_embedding")
        .call(text_string);
    return res.toDartString();
  }

  String runInference(String text) {
    ffi.Pointer<Utf8> text_string = text.toNativeUtf8();
    var res = openLib()
        .lookupFunction<inference_request, inference_request>("run_inference")
        .call(text_string);
    return res.toDartString();
  }

  String getScore(String image_embedding, String text_embedding, int vec_dim) {
    ffi.Pointer<Utf8> image_embedding_string = image_embedding.toNativeUtf8();
    ffi.Pointer<Utf8> text_embedding_string = text_embedding.toNativeUtf8();
    var res = openLib()
        .lookupFunction<get_score_request, get_score_response>("get_score")
        .call(image_embedding_string, text_embedding_string, vec_dim);
    return res.toDartString();
  }
}
